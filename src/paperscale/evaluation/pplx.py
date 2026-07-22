"""Opt-in reference-free perplexity scorer for OCR page text.

Scores each page's text -- both raw and dictionary-corrected -- through an
external vLLM ``/v1/completions`` endpoint using ``prompt_logprobs``. The
raw-minus-corrected perplexity gap isolates surface-typo noise (spelling
slips a dictionary can fix) from residual incoherence the model still can't
predict after correction.

vLLM response shape (``choices[0].prompt_logprobs``): a list with one entry per
prompt token, in order. The FIRST entry is ``null`` -- there is no logprob for
the first token and, crucially, no ``decoded_token`` for it either. Each
non-null entry is a dict mapping ``token-id-string -> {"logprob": float,
"decoded_token": str, "rank": int, ...}``. With ``prompt_logprobs: 0`` the actual
prompt token is the ``rank == 0`` entry (usually the only one).

Char-offset -> page mapping: we walk a running char cursor over the non-null
tokens' ``decoded_token`` strings and assign each token to the page whose
``[start, next_start)`` span contains the token's START offset. The first
(null) token has no ``decoded_token``, so we infer its char span as
``len(prompt) - sum(len(decoded) for non-null tokens)`` -- tokens tile the whole
prompt, so the leftover prefix length is exactly the first token's span. That
keeps every subsequent offset aligned without needing the null token's text.
The null token is excluded from logprob sums but still accounts for its span.
(Assumes only index 0 is null, per the vLLM contract; any other null entry
would lump its span into the leading offset.)
"""

from __future__ import annotations

import math
import warnings

import httpx

from paperscale.evaluation.runs import PageText
from paperscale.evaluation.spell import build_dictionary, correct_text

__all__ = ["build_dictionary", "correct_text", "score_run_pplx"]

_CHARS_PER_TOKEN = 3
_MAX_TOKENS_PER_CHUNK = 32_000


def _ppl(n_tokens: int, sum_logprob: float) -> float:
    """Perplexity ``exp(-mean_logprob)``. NaN when no tokens scored."""
    if n_tokens == 0:
        return float("nan")
    return math.exp(-sum_logprob / n_tokens)


def _pick(entry: dict) -> tuple[float, str]:
    """From one prompt_logprobs entry, take the rank-0 token (or the sole one)."""
    chosen = None
    for v in entry.values():
        if v.get("rank") == 0:
            chosen = v
            break
    if chosen is None:
        chosen = next(iter(entry.values()))
    return chosen["logprob"], chosen["decoded_token"]


def _assign_page(boundaries: list[tuple[int, int]], offset: int) -> int:
    """Page number whose ``[start, next_start)`` span contains ``offset``."""
    pn = boundaries[0][0]
    for page, start in boundaries:
        if start <= offset:
            pn = page
        else:
            break
    return pn


def _chunk_pages(items: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Split pages into sequential chunks (at page boundaries) under the token cap.

    The first page of each chunk after the first loses cross-page conditioning.
    """
    chunks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_chars = 0
    for page, text in items:
        add = len(text) + 1  # + inter-page "\n"
        if cur and (cur_chars + add) // _CHARS_PER_TOKEN > _MAX_TOKENS_PER_CHUNK:
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append((page, text))
        cur_chars += add
    if cur:
        chunks.append(cur)
    return chunks


def _prompt_logprobs(client: httpx.Client, url: str, model: str, prompt: str) -> list:
    resp = client.post(
        f"{url}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 1, "prompt_logprobs": 0},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["prompt_logprobs"]


def _score_pass(
    items: list[tuple[int, str]], client: httpx.Client, url: str, model: str
) -> dict[int, tuple[int, float]]:
    """Score one pass (raw or corrected) -> {page: (n_tokens, sum_logprob)}."""
    acc: dict[int, list] = {page: [0, 0.0] for page, _ in items}
    for chunk in _chunk_pages(items):
        # Join chunk pages with "\n", tracking each page's start offset.
        boundaries: list[tuple[int, int]] = []
        off = 0
        for page, text in chunk:
            boundaries.append((page, off))
            off += len(text) + 1
        prompt = "\n".join(text for _, text in chunk)

        plps = _prompt_logprobs(client, url, model, prompt)
        toks = [_pick(e) for e in plps if e is not None]
        decoded_len = sum(len(dec) for _, dec in toks)
        if decoded_len > len(prompt):
            # Tokens should tile the prompt exactly (null-token span = len(prompt) - decoded_len >= 0).
            # A tokenizer whose decoded_token strings carry marker glyphs (SentencePiece "_", BPE "G")
            # can overrun, making the leading offset negative -> page attribution shifts. Surface it
            # instead of silently clamping; validate against the real --pplx-model before trusting numbers.
            warnings.warn(
                f"pplx: decoded tokens ({decoded_len} chars) overrun the prompt ({len(prompt)} chars); "
                "per-page perplexity attribution may be miscalibrated for this model's tokenizer.",
                stacklevel=2,
            )
        cursor = max(0, len(prompt) - decoded_len)  # span of the skipped null token
        for logprob, dec in toks:
            page = _assign_page(boundaries, cursor)
            acc[page][0] += 1
            acc[page][1] += logprob
            cursor += len(dec)
    return {page: (n, s) for page, (n, s) in acc.items()}


def score_run_pplx(
    pages: list[PageText],
    *,
    pplx_url: str,
    pplx_model: str,
    extra_words: frozenset[str] = frozenset(),
    client: "httpx.Client | None" = None,
    progress=None,
    sym=None,
) -> dict[str, list[tuple]]:
    """Score every page raw and dictionary-corrected; return DB row tuples per doc.

    Row order (positional -- the DB layer relies on it)::

        (doc, page, n_tokens_raw, sum_logprob_raw, ppl_raw,
         n_tokens_corrected, sum_logprob_corrected, ppl_corrected)
    """
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=httpx.Timeout(600.0))
    # sym may be supplied by the caller (the correction metric already built one); else built lazily.
    try:
        by_doc: dict[str, list[PageText]] = {}
        for p in pages:
            by_doc.setdefault(p.doc, []).append(p)

        result: dict[str, list[tuple]] = {}
        for doc, plist in by_doc.items():
            plist = sorted(plist, key=lambda p: p.page)
            raw = _score_pass([(p.page, p.text) for p in plist], client, pplx_url, pplx_model)

            if sym is None:
                sym = build_dictionary(extra_words)
            corr = _score_pass(
                [(p.page, correct_text(p.text, sym)) for p in plist], client, pplx_url, pplx_model
            )

            rows = []
            for p in plist:
                n_r, s_r = raw[p.page]
                n_c, s_c = corr[p.page]
                rows.append(
                    (doc, p.page, n_r, s_r, _ppl(n_r, s_r), n_c, s_c, _ppl(n_c, s_c))
                )
            result[doc] = rows
            if progress is not None:
                progress(doc)
        return result
    finally:
        if own_client:
            client.close()
