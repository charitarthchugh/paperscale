"""Tests for the opt-in perplexity scorer.

Offset/aggregation tests stub :func:`paperscale.evaluation.pplx.apost` (the same
seam the OCR path posts through). One test drives the real socket transport
against a throwaway HTTP server to prove requests actually overlap.
"""

import json
import math
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from paperscale.evaluation import pplx
from paperscale.evaluation.runs import PageText

_LOGPROB = -0.5

# "wrold" is an OCR-style garble the 50k dictionary corrects to "world", so a page
# containing it is guaranteed to differ between the raw and corrected passes --
# i.e. it really does cost two requests (see the clean-document dedup test).
_DIRTY = "wrold hello"


def _plps_for(prompt: str, pad: str = "") -> list:
    """Tokenize a prompt into fixed 4-char chunks whose decoded_tokens rebuild it.

    The FIRST entry is null (no logprob, no decoded_token -- its span is inferred
    by the scorer). ``pad`` lengthens each decoded_token to force an overrun.
    """
    chunks = [prompt[i : i + 4] for i in range(0, len(prompt), 4)]
    return [None] + [
        {"7": {"logprob": _LOGPROB, "decoded_token": c + pad, "rank": 0}} for c in chunks[1:]
    ]


def _fake_apost(pad: str = "", delay: float = 0.0, tracker: dict | None = None):
    """Build a stand-in for pplx.apost returning (status_code, body_bytes)."""

    async def apost(url, json_data, api_key=None):
        if tracker is not None:
            tracker["inflight"] += 1
            tracker["peak"] = max(tracker["peak"], tracker["inflight"])
            tracker["count"] += 1
        if delay:
            import asyncio

            await asyncio.sleep(delay)
        if tracker is not None:
            tracker["inflight"] -= 1
        body = json.dumps(
            {"choices": [{"prompt_logprobs": _plps_for(json_data["prompt"], pad)}]}
        ).encode()
        return 200, body

    return apost


class _StubApost:
    """Context manager swapping pplx.apost for a stub."""

    def __init__(self, **kwargs):
        self.stub = _fake_apost(**kwargs)

    def __enter__(self):
        self.orig = pplx.apost
        pplx.apost = self.stub
        return self

    def __exit__(self, *exc):
        pplx.apost = self.orig
        return False


_SYM = None


def _sym():
    """Build the 50k-word dictionary once -- it costs ~1.3s per construction."""
    global _SYM
    if _SYM is None:
        _SYM = pplx.build_dictionary()
    return _SYM


def _score(pages, **kwargs):
    kwargs.setdefault("sym", _sym())
    return pplx.score_run_pplx(pages, pplx_url="http://vllm", pplx_model="q", **kwargs)


class ScoreRunPplxTest(unittest.TestCase):
    def test_offset_mapping_excludes_null_token_and_runs_both_passes(self):
        # raw joined = "hello\nworld" (len 11); 4-char tokens:
        #   "hell"[0,4) null   "o\nwo"[4,8) -> page1 (4 < 6)   "rld"[8,11) -> page2
        # boundaries: page1@0, page2@6. Null "hell" (page1) is excluded from sums.
        pages = [
            PageText("m", "/d.pdf", 1, "hello"),
            PageText("m", "/d.pdf", 2, "world"),
        ]
        with _StubApost():
            rows = _score(pages)["/d.pdf"]
        self.assertEqual(len(rows), 2)

        (doc1, pg1, n_r1, s_r1, _ppl_r1, n_c1, s_c1, _ppl_c1) = rows[0]
        (doc2, pg2, n_r2, s_r2, _ppl_r2, n_c2, _s_c2, _ppl_c2) = rows[1]
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
        with _StubApost(pad="XXXXX"), self.assertWarns(UserWarning):
            _score(pages)

    def test_ppl_is_exp_neg_mean_logprob(self):
        pages = [PageText("m", "/d.pdf", 1, "hello"), PageText("m", "/d.pdf", 2, "world")]
        with _StubApost():
            rows = _score(pages)["/d.pdf"]
        for (_doc, _pg, n, s, ppl, *_rest) in rows:
            self.assertAlmostEqual(ppl, math.exp(-s / n))

    def test_chunking_stitches_per_page_results(self):
        # Force one chunk per page and confirm results match the single-prompt path.
        pages = [PageText("m", "/d.pdf", 1, "hello"), PageText("m", "/d.pdf", 2, "world")]
        orig_cpt = pplx._CHARS_PER_TOKEN
        pplx._CHARS_PER_TOKEN = 1
        try:
            with _StubApost():
                rows = _score(pages, chunk_tokens=0)["/d.pdf"]
        finally:
            pplx._CHARS_PER_TOKEN = orig_cpt
        self.assertEqual([(r[1], r[2]) for r in rows], [(1, 1), (2, 1)])

    def test_chunk_tokens_splits_into_separate_requests(self):
        pages = [PageText("m", "/d.pdf", pg, _DIRTY) for pg in range(1, 5)]
        tracker = {"inflight": 0, "peak": 0, "count": 0}
        with _StubApost(tracker=tracker):
            _score(pages, chunk_tokens=1_000_000)
        one_chunk = tracker["count"]
        tracker.update(inflight=0, peak=0, count=0)
        with _StubApost(tracker=tracker):
            _score(pages, chunk_tokens=0)  # every page its own chunk
        self.assertEqual(one_chunk, 2)  # raw + corrected
        self.assertEqual(tracker["count"], 8)  # 4 pages x 2 passes

    def test_both_passes_of_a_document_are_in_flight_together(self):
        """A single document must not serialize raw-then-corrected."""
        pages = [PageText("m", "/solo.pdf", 1, _DIRTY)]
        tracker = {"inflight": 0, "peak": 0, "count": 0}
        with _StubApost(delay=0.05, tracker=tracker):
            _score(pages, concurrency=8)
        self.assertEqual(tracker["count"], 2)
        self.assertEqual(tracker["peak"], 2)

    def test_concurrency_is_capped_by_the_semaphore(self):
        pages = [PageText("m", f"/d{d}.pdf", 1, _DIRTY) for d in range(32)]
        tracker = {"inflight": 0, "peak": 0, "count": 0}
        with _StubApost(delay=0.02, tracker=tracker):
            _score(pages, concurrency=6)
        self.assertEqual(tracker["count"], 64)  # 32 docs x 2 passes
        self.assertLessEqual(tracker["peak"], 6)
        self.assertGreater(tracker["peak"], 1)

    def test_clean_document_skips_the_duplicate_corrected_pass(self):
        """No corrections => corrected prompt == raw prompt => don't pay for it twice."""
        clean = [PageText("m", "/clean.pdf", 1, "the quick brown fox")]
        tracker = {"inflight": 0, "peak": 0, "count": 0}
        with _StubApost(tracker=tracker):
            rows = _score(clean)["/clean.pdf"]
        self.assertEqual(tracker["count"], 1)  # one request, not two
        # corrected columns still populated, and identical to raw
        (_doc, _pg, n_r, s_r, ppl_r, n_c, s_c, ppl_c) = rows[0]
        self.assertEqual((n_r, s_r, ppl_r), (n_c, s_c, ppl_c))
        self.assertGreater(n_r, 0)

    def test_dirty_document_still_runs_both_passes(self):
        dirty = [PageText("m", "/dirty.pdf", 1, "teh qiuck brwn fx jmps")]
        tracker = {"inflight": 0, "peak": 0, "count": 0}
        with _StubApost(tracker=tracker):
            rows = _score(dirty)["/dirty.pdf"]
        self.assertEqual(tracker["count"], 2)
        self.assertEqual(len(rows), 1)

    def test_a_failing_document_is_skipped_not_fatal(self):
        """One doomed doc must not cost the other 199,999 their progress."""
        pages = [PageText("m", f"/d{d}.pdf", 1, _DIRTY) for d in range(4)]
        orig = pplx.apost
        good = _fake_apost()

        async def flaky(url, json_data, api_key=None):
            if "/d2.pdf" in json_data["prompt"] or json_data["prompt"].startswith("POISON"):
                return 500, b'{"error": "boom"}'
            return await good(url, json_data, api_key)

        # Poison exactly one document by making its text recognizable.
        pages[2] = PageText("m", "/d2.pdf", 1, "POISON " + _DIRTY)
        pplx.apost = flaky
        orig_attempts = pplx._MAX_ATTEMPTS
        pplx._MAX_ATTEMPTS = 1  # skip the real backoff sleeps
        seen = []
        try:
            with self.assertLogs("paperscale.evaluation.pplx", level="ERROR"):
                out = _score(pages, concurrency=4, progress=lambda doc: seen.append(doc))
        finally:
            pplx.apost = orig
            pplx._MAX_ATTEMPTS = orig_attempts

        self.assertNotIn("/d2.pdf", out)
        self.assertEqual(sorted(out), ["/d0.pdf", "/d1.pdf", "/d3.pdf"])
        # progress still advances for the skipped doc so the bar reaches its total
        self.assertEqual(len(seen), 4)

    def test_on_doc_and_progress_fire_once_per_document(self):
        pages = [PageText("m", f"/d{d}.pdf", pg, _DIRTY) for d in range(5) for pg in (1, 2)]
        seen, advanced = [], []
        with _StubApost():
            out = _score(
                pages,
                concurrency=4,
                on_doc=lambda doc, rows: seen.append(doc),
                progress=lambda doc: advanced.append(doc),
            )
        self.assertEqual(len(out), 5)
        self.assertEqual(sorted(seen), sorted(out))
        self.assertEqual(len(advanced), 5)


# --- real socket transport ------------------------------------------------


class _CountingHandler(BaseHTTPRequestHandler):
    """Minimal /v1/completions that records peak concurrent requests."""

    state = {"inflight": 0, "peak": 0, "count": 0}
    lock = threading.Lock()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.state["inflight"] += 1
            self.state["peak"] = max(self.state["peak"], self.state["inflight"])
            self.state["count"] += 1
        time.sleep(0.05)  # stand in for server-side prefill
        with self.lock:
            self.state["inflight"] -= 1
        payload = json.dumps(
            {"choices": [{"prompt_logprobs": _plps_for(body["prompt"])}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):  # noqa: A002 -- base class signature
        pass


class RealTransportConcurrencyTest(unittest.TestCase):
    """Drives the actual apost socket path, not a stub."""

    def test_requests_overlap_up_to_the_configured_cap(self):
        _CountingHandler.state = {"inflight": 0, "peak": 0, "count": 0}
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            pages = [PageText("m", f"/d{d}.pdf", 1, _DIRTY) for d in range(32)]
            elapsed = time.perf_counter()
            pplx.score_run_pplx(
                pages, pplx_url=url, pplx_model="q", concurrency=16, sym=_sym()
            )
            elapsed = time.perf_counter() - elapsed
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(_CountingHandler.state["count"], 64)  # 32 docs x 2 passes
        # 64 requests x 50ms serialized would be 3.2s; at 16-way overlap, ~0.2s.
        self.assertGreaterEqual(_CountingHandler.state["peak"], 8)
        self.assertLess(elapsed, 1.5)


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
