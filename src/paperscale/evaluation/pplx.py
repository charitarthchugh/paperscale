"""Opt-in reference-free perplexity scorer for OCR page text.

Scores each page's text -- both raw and dictionary-corrected -- through an
external vLLM ``/v1/completions`` endpoint using ``prompt_logprobs``. The
raw-minus-corrected perplexity gap isolates surface-typo noise (spelling
slips a dictionary can fix) from residual incoherence the model still can't
predict after correction.

Concurrency mirrors the OCR path (:mod:`paperscale.pipeline`): a single asyncio
event loop, requests sent through the very same raw-socket
:func:`~paperscale.pipeline.apost` (httpx/aiohttp connection pools deadlock at
this request volume), and one module-level :class:`asyncio.BoundedSemaphore`
capping in-flight requests -- the analogue of ``max_concurrent_requests_limit``.

The unit of parallelism is **one chunk request**, not one document. A document's
raw and corrected passes, and every chunk within them, are issued concurrently,
so a short document never idles a slot waiting on its own second pass. The
semaphore -- not the shape of any document -- decides how many requests the
server sees at once.

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

import asyncio
import errno
import json
import logging
import math
import warnings

from paperscale.evaluation.runs import PageText
from paperscale.evaluation.spell import build_dictionary, correct_text
from paperscale.pipeline import apost

logger = logging.getLogger(__name__)

__all__ = ["build_dictionary", "correct_text", "score_run_pplx", "score_run_pplx_async"]

_CHARS_PER_TOKEN = 3
_MAX_TOKENS_PER_CHUNK = 32_000

# Retry budget for one chunk request. Mirrors try_single_page_with_backoff:
# fd exhaustion is a soft drop (retried forever, capped delay, does not consume
# an attempt), connection errors get bounded exponential backoff, and a bad
# response body is retried a few times before the document is given up on.
_MAX_ATTEMPTS = 5
_MAX_CONN_BACKOFF_ATTEMPTS = 6
_FD_MAX_BACKOFF_SECONDS = 30

# In-flight request cap shared by every coroutine. Rebuilt per run inside the
# event loop by score_run_pplx_async (a Semaphore binds to the running loop).
_request_limit = asyncio.BoundedSemaphore(1)


class PplxRequestError(RuntimeError):
    """A chunk request came back unusable (bad status, or malformed body)."""


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


def _chunk_pages(
    items: list[tuple[int, str]], max_tokens: int | None = None
) -> list[list[tuple[int, str]]]:
    """Split pages into sequential chunks (at page boundaries) under the token cap.

    The first page of each chunk after the first loses cross-page conditioning.
    """
    cap = _MAX_TOKENS_PER_CHUNK if max_tokens is None else max_tokens
    chunks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_chars = 0
    for page, text in items:
        add = len(text) + 1  # + inter-page "\n"
        if cur and (cur_chars + add) // _CHARS_PER_TOKEN > cap:
            chunks.append(cur)
            cur, cur_chars = [], 0
        cur.append((page, text))
        cur_chars += add
    if cur:
        chunks.append(cur)
    return chunks


async def _post_logprobs(url: str, model: str, prompt: str, api_key: str | None) -> list:
    """One ``/v1/completions`` call -> the raw ``prompt_logprobs`` list.

    The semaphore wraps only the send, exactly as ``try_single_page`` does, so
    response parsing never holds a slot the server could be using.
    """
    payload = {"model": model, "prompt": prompt, "max_tokens": 1, "prompt_logprobs": 0}
    async with _request_limit:
        status_code, body = await apost(
            f"{url.rstrip('/')}/v1/completions", json_data=payload, api_key=api_key
        )
    if status_code != 200:
        snippet = (body or b"")[:300]
        raise PplxRequestError(f"vLLM returned {status_code}: {snippet!r}")
    try:
        return json.loads(body)["choices"][0]["prompt_logprobs"]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise PplxRequestError(f"malformed vLLM response: {type(e).__name__}: {e}") from e


async def _post_logprobs_with_backoff(
    url: str, model: str, prompt: str, api_key: str | None
) -> list:
    """Retry a chunk request through transient failures; raise once the budget is spent.

    File-descriptor exhaustion (EMFILE/ENFILE) is a soft drop -- backed off and
    retried indefinitely without consuming an attempt, since it self-resolves as
    in-flight sockets close. Connection errors get bounded exponential backoff.
    """
    attempt = conn_backoff = fd_backoff = 0
    while True:
        try:
            return await _post_logprobs(url, model, prompt, api_key)
        except asyncio.CancelledError:
            raise
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE):
                delay = min(2**fd_backoff, _FD_MAX_BACKOFF_SECONDS)
                fd_backoff += 1
                logger.warning(
                    f"pplx: out of file descriptors (errno {e.errno}); retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                continue
            conn_backoff += 1
            if conn_backoff > _MAX_CONN_BACKOFF_ATTEMPTS:
                raise
            delay = min(10 * 2 ** (conn_backoff - 1), 120)
            logger.warning(
                f"pplx: connection error ({type(e).__name__}: {e}); backoff "
                f"{conn_backoff}/{_MAX_CONN_BACKOFF_ATTEMPTS}, sleeping {delay}s"
            )
            await asyncio.sleep(delay)
        except (PplxRequestError, asyncio.TimeoutError) as e:
            attempt += 1
            if attempt >= _MAX_ATTEMPTS:
                raise
            delay = min(2**attempt, 30)
            logger.warning(
                f"pplx: request failed ({type(e).__name__}: {e}); "
                f"attempt {attempt}/{_MAX_ATTEMPTS}, sleeping {delay}s"
            )
            await asyncio.sleep(delay)


async def _score_chunk(
    chunk: list[tuple[int, str]], url: str, model: str, api_key: str | None
) -> list[tuple[int, int, float]]:
    """Score one chunk -> ``[(page, n_tokens, sum_logprob), ...]``."""
    # Join chunk pages with "\n", tracking each page's start offset.
    boundaries: list[tuple[int, int]] = []
    off = 0
    for page, text in chunk:
        boundaries.append((page, off))
        off += len(text) + 1
    prompt = "\n".join(text for _, text in chunk)

    plps = await _post_logprobs_with_backoff(url, model, prompt, api_key)
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

    acc: dict[int, list] = {page: [0, 0.0] for page, _ in chunk}
    cursor = max(0, len(prompt) - decoded_len)  # span of the skipped null token
    # `cursor` only moves forward and `boundaries` ascends, so one shared walk
    # replaces a per-token scan of every boundary (was O(tokens x pages)).
    bi = 0
    for logprob, dec in toks:
        while bi + 1 < len(boundaries) and boundaries[bi + 1][1] <= cursor:
            bi += 1
        page = boundaries[bi][0]
        acc[page][0] += 1
        acc[page][1] += logprob
        cursor += len(dec)
    return [(page, n, s) for page, (n, s) in acc.items()]


def _corrected_items(plist: list[PageText], sym) -> list[tuple[int, str]]:
    """Dictionary-correct every page of a document (pure CPU, no I/O)."""
    return [(p.page, correct_text(p.text, sym)) for p in plist]


async def _score_doc(
    doc: str,
    plist: list[PageText],
    *,
    url: str,
    model: str,
    sym,
    api_key: str | None,
    chunk_tokens: int | None,
) -> list[tuple]:
    """Score one document (both passes) -> DB row tuples."""
    plist = sorted(plist, key=lambda p: p.page)

    # SymSpell is pure Python and holds the GIL. A thread does not make it
    # faster, but it lets the interpreter interleave (5ms switch interval) so a
    # long document's correction cannot stall every in-flight request behind it.
    corr_items = await asyncio.to_thread(_corrected_items, plist, sym)

    raw_items = [(p.page, p.text) for p in plist]
    raw_chunks = _chunk_pages(raw_items, chunk_tokens)

    # When the dictionary changed nothing, the corrected pass would send prompts
    # byte-identical to the raw ones. Reusing the raw numbers is exact (same
    # prompt => same logprobs, the server is deterministic at temperature 0 for
    # prompt scoring) and halves GPU work on every already-clean document.
    unchanged = corr_items == raw_items
    corr_chunks = [] if unchanged else _chunk_pages(corr_items, chunk_tokens)

    # Both passes go out at once; the semaphore decides what is actually in flight.
    results = await asyncio.gather(
        *(_score_chunk(c, url, model, api_key) for c in raw_chunks),
        *(_score_chunk(c, url, model, api_key) for c in corr_chunks),
    )

    raw: dict[int, tuple[int, float]] = {}
    corr: dict[int, tuple[int, float]] = {}
    for i, rows in enumerate(results):
        target = raw if i < len(raw_chunks) else corr
        for page, n, s in rows:
            prev_n, prev_s = target.get(page, (0, 0.0))
            target[page] = (prev_n + n, prev_s + s)
    if unchanged:
        corr = dict(raw)

    out = []
    for p in plist:
        n_r, s_r = raw.get(p.page, (0, 0.0))
        n_c, s_c = corr.get(p.page, (0, 0.0))
        out.append((doc, p.page, n_r, s_r, _ppl(n_r, s_r), n_c, s_c, _ppl(n_c, s_c)))
    return out


def _handle_doc_failure(doc: str, exc: Exception) -> None:
    """One document exhausted its retry budget: log it and let the run continue.

    The document is simply never written to the pplx table, so the existing
    resume path (``EvalDB.pplx_done_docs``) retries it on the next invocation.
    Aborting instead would throw away every hour of progress made so far over a
    corpus this size, and the failure is already durable in the log.
    """
    logger.error(f"pplx: giving up on {doc} after retries ({type(exc).__name__}: {exc}); "
                 "left unscored -- a later run will retry it")


async def score_run_pplx_async(
    pages: list[PageText],
    *,
    pplx_url: str,
    pplx_model: str,
    extra_words: frozenset[str] = frozenset(),
    sym=None,
    progress=None,
    concurrency: int = 64,
    chunk_tokens: int | None = None,
    api_key: str | None = None,
    on_doc=None,
) -> dict[str, list[tuple]]:
    """Async core of :func:`score_run_pplx`. See there for the contract."""
    global _request_limit
    _request_limit = asyncio.BoundedSemaphore(max(1, concurrency))

    # sym may be supplied by the caller (the correction metric already built one).
    if sym is None:
        sym = build_dictionary(extra_words)

    by_doc: dict[str, list[PageText]] = {}
    for p in pages:
        by_doc.setdefault(p.doc, []).append(p)

    queue: asyncio.Queue = asyncio.Queue()
    for item in by_doc.items():
        queue.put_nowait(item)

    result: dict[str, list[tuple]] = {}
    failed: list[str] = []

    async def doc_worker() -> None:
        while True:
            try:
                doc, plist = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                rows = await _score_doc(
                    doc,
                    plist,
                    url=pplx_url,
                    model=pplx_model,
                    sym=sym,
                    api_key=api_key,
                    chunk_tokens=chunk_tokens,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 -- policy lives in one place
                _handle_doc_failure(doc, e)
                failed.append(doc)
                # Still advance progress: the phase total counted this doc, and a
                # bar that never reaches its total reads as a hang.
                if progress is not None:
                    progress(doc)
                continue
            result[doc] = rows
            # on_doc/progress run on the event-loop thread only (never inside a
            # gather child), so the DB handle and TUI state stay single-threaded.
            if on_doc is not None:
                on_doc(doc, rows)
            if progress is not None:
                progress(doc)

    # One worker per request slot: each has >=1 chunk in flight or queued behind
    # the semaphore, so the server always has a full batch to schedule.
    n_workers = max(1, min(concurrency, len(by_doc)))
    await asyncio.gather(*(doc_worker() for _ in range(n_workers)))
    if failed:
        logger.error(
            f"pplx: {len(failed)}/{len(by_doc)} documents left unscored after retries "
            f"(first few: {failed[:5]}); re-run to retry them."
        )
    return result


def score_run_pplx(
    pages: list[PageText],
    *,
    pplx_url: str,
    pplx_model: str,
    extra_words: frozenset[str] = frozenset(),
    sym=None,
    progress=None,
    concurrency: int = 64,
    chunk_tokens: int | None = None,
    api_key: str | None = None,
    on_doc=None,
) -> dict[str, list[tuple]]:
    """Score every page raw and dictionary-corrected; return DB row tuples per doc.

    Row order (positional -- the DB layer relies on it)::

        (doc, page, n_tokens_raw, sum_logprob_raw, ppl_raw,
         n_tokens_corrected, sum_logprob_corrected, ppl_corrected)

    ``concurrency``: chunk requests in flight against the vLLM server at once.
    ``chunk_tokens``: approximate token cap per request (default
    ``_MAX_TOKENS_PER_CHUNK``); smaller prompts let vLLM co-schedule far more of
    them, at the cost of cross-page conditioning at each new chunk boundary.
    ``on_doc``: optional callable(doc, rows) fired as each doc completes -- the
    streaming-write sink. Both ``on_doc`` and ``progress`` are invoked only from
    the event-loop thread, so DB handles and TUI state stay single-threaded.
    """
    return asyncio.run(
        score_run_pplx_async(
            pages,
            pplx_url=pplx_url,
            pplx_model=pplx_model,
            extra_words=extra_words,
            sym=sym,
            progress=progress,
            concurrency=concurrency,
            chunk_tokens=chunk_tokens,
            api_key=api_key,
            on_doc=on_doc,
        )
    )
