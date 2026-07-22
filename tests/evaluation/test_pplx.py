"""Tests for the opt-in perplexity scorer (no live network -- httpx.MockTransport)."""

import json
import math
import unittest

import httpx

from paperscale.evaluation import pplx
from paperscale.evaluation.runs import PageText

_LOGPROB = -0.5


def _fake_handler(request: httpx.Request) -> httpx.Response:
    """Fake vLLM /v1/completions.

    Tokenizes the prompt into fixed 4-char chunks whose decoded_tokens
    reconstruct the prompt. The FIRST entry is null (no logprob, no
    decoded_token -- its span is inferred by the scorer). Every other entry is a
    dict ``{token_id: {"logprob", "decoded_token", "rank": 0}}``.
    """
    body = json.loads(request.content)
    prompt = body["prompt"]
    chunks = [prompt[i : i + 4] for i in range(0, len(prompt), 4)]
    plps = []
    for idx, ch in enumerate(chunks):
        if idx == 0:
            plps.append(None)
        else:
            plps.append({"7": {"logprob": _LOGPROB, "decoded_token": ch, "rank": 0}})
    return httpx.Response(200, json={"choices": [{"prompt_logprobs": plps}]})


def _overrun_handler(request: httpx.Request) -> httpx.Response:
    """Like _fake_handler but each decoded_token is padded, so the decoded chars
    overrun the prompt -- the scorer should warn rather than silently misalign."""
    body = json.loads(request.content)
    prompt = body["prompt"]
    chunks = [prompt[i : i + 4] for i in range(0, len(prompt), 4)]
    plps = [None] + [
        {"7": {"logprob": _LOGPROB, "decoded_token": ch + "XXXXX", "rank": 0}} for ch in chunks[1:]
    ]
    return httpx.Response(200, json={"choices": [{"prompt_logprobs": plps}]})


def _client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_fake_handler))


class ScoreRunPplxTest(unittest.TestCase):
    def test_offset_mapping_excludes_null_token_and_runs_both_passes(self):
        # raw joined = "hello\nworld" (len 11); 4-char tokens:
        #   "hell"[0,4) null   "o\nwo"[4,8) -> page1 (4 < 6)   "rld"[8,11) -> page2
        # boundaries: page1@0, page2@6. Null "hell" (page1) is excluded from sums.
        pages = [
            PageText("m", "/d.pdf", 1, "hello"),
            PageText("m", "/d.pdf", 2, "world"),
        ]
        with _client() as c:
            out = pplx.score_run_pplx(
                pages, pplx_url="http://vllm", pplx_model="q", client=c
            )
        rows = out["/d.pdf"]
        self.assertEqual(len(rows), 2)

        (doc1, pg1, n_r1, s_r1, ppl_r1, n_c1, s_c1, ppl_c1) = rows[0]
        (doc2, pg2, n_r2, s_r2, ppl_r2, n_c2, s_c2, ppl_c2) = rows[1]
        self.assertEqual((doc1, pg1), ("/d.pdf", 1))
        self.assertEqual((doc2, pg2), ("/d.pdf", 2))
        # exactly one scored token per page; null token did NOT count toward page1
        self.assertEqual(n_r1, 1)
        self.assertEqual(n_r2, 1)
        self.assertAlmostEqual(s_r1, _LOGPROB)
        self.assertAlmostEqual(s_r2, _LOGPROB)
        # both passes populated ("hello"/"world" are real words -> corrected == raw)
        self.assertEqual(n_c1, 1)
        self.assertEqual(n_c2, 1)
        self.assertAlmostEqual(s_c1, _LOGPROB)

    def test_warns_when_decoded_tokens_overrun_prompt(self):
        pages = [PageText("m", "/d.pdf", 1, "hello"), PageText("m", "/d.pdf", 2, "world")]
        client = httpx.Client(transport=httpx.MockTransport(_overrun_handler))
        with client, self.assertWarns(UserWarning):
            pplx.score_run_pplx(pages, pplx_url="http://x", pplx_model="q", client=client)

    def test_ppl_is_exp_neg_mean_logprob(self):
        pages = [PageText("m", "/d.pdf", 1, "hello"), PageText("m", "/d.pdf", 2, "world")]
        with _client() as c:
            rows = pplx.score_run_pplx(
                pages, pplx_url="http://x", pplx_model="q", client=c
            )["/d.pdf"]
        for (_doc, _pg, n, s, ppl, *_rest) in rows:
            self.assertAlmostEqual(ppl, math.exp(-s / n))

    def test_chunking_stitches_per_page_results(self):
        # Force one chunk per page and confirm results match the single-prompt path.
        pages = [PageText("m", "/d.pdf", 1, "hello"), PageText("m", "/d.pdf", 2, "world")]
        orig_cap, orig_cpt = pplx._MAX_TOKENS_PER_CHUNK, pplx._CHARS_PER_TOKEN
        pplx._MAX_TOKENS_PER_CHUNK, pplx._CHARS_PER_TOKEN = 0, 1
        try:
            with _client() as c:
                rows = pplx.score_run_pplx(
                    pages, pplx_url="http://x", pplx_model="q", client=c
                )["/d.pdf"]
        finally:
            pplx._MAX_TOKENS_PER_CHUNK, pplx._CHARS_PER_TOKEN = orig_cap, orig_cpt
        self.assertEqual([(r[1], r[2]) for r in rows], [(1, 1), (2, 1)])


class CorrectTextTest(unittest.TestCase):
    def test_corrects_misspelling_preserves_numbers_and_punct(self):
        sym = pplx.build_dictionary()
        # "wrold" is an OCR-style garble absent from the 50k list (unlike common
        # web misspellings such as "teh", which wordfreq actually contains).
        out = pplx.correct_text("wrold cat 42, ok!", sym)
        self.assertIn("world", out)
        self.assertNotIn("wrold", out)
        self.assertIn("cat", out)  # already correct -> untouched
        self.assertIn("42", out)  # number preserved
        self.assertIn(",", out)  # punctuation preserved
        self.assertTrue(out.endswith("!"))


class BuildDictionaryTest(unittest.TestCase):
    def test_lazy_import_and_extra_words(self):
        sym = pplx.build_dictionary(frozenset({"zzzq"}))
        self.assertIn("the", sym.words)
        self.assertIn("zzzq", sym.words)


if __name__ == "__main__":
    unittest.main()
