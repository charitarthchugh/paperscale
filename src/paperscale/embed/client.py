"""The HTTP client for one vLLM OpenAI-compatible embedding server.

Three routes, three different dispositions on failure:

* ``POST /v1/embeddings`` -- the GPU work, bounded by ``--concurrency``.
* ``POST /tokenize`` -- exact token counts for the chunker (design 5.2), served
  CPU-side in the API server process and bounded separately (design 12.4).
* ``GET /v1/models`` -- the served model id and ``max_model_len``, asked once at
  startup before the reporter exists (design 12.1 steps 4 and 10).

`embed` mirrors :mod:`paperscale.evaluation.pplx`, not the OCR path. It is the same
workload -- prefill-only against vLLM -- so it inherits pplx's three-axis backoff and
its raw-socket transport (httpx/aiohttp connection pools deadlock at this request
volume; see pplx's module docstring). It departs in one way that matters: it
**raises** instead of calling ``sys.exit(1)`` from inside a worker, and the exception
type says how far the failure reaches. ``try_single_page_with_backoff`` on the OCR
side has one axis, an uncapped delay that reaches 85 minutes by attempt 10, and
``sys.exit(1)``; none of those three transfer.

The wire format is not negotiable and is the sharpest correctness point in this
module -- see :func:`EmbedClient._send_embeddings` and design 16.2.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import logging
import random
import re
import urllib.error
import urllib.request

from paperscale.pipeline import apost

logger = logging.getLogger(__name__)

__all__ = ["EmbedClient", "EmbedRequestError", "ServerGoneError", "TerminalDocumentError"]

# Retry shape, design 12.6. The connection axis is the only one with jitter, and
# the only one whose exhaustion ends the Invocation rather than the Document.
#
# Both budgets count *attempts*, not retries-after-the-first: design 12.6's table
# reads "connection error | 6" and "bad response / timeout | --max-request-retries
# 8", so six failed connections give up and eight failed responses give up. The two
# axes therefore test the same way (``>=``).
#
# The name carries the same fact, because the old one did not. Read as a count of
# *backoffs* -- which the guard and the warning line both did -- six sleeps need a
# seventh send, and `ServerGoneError` was left saying "after 6 connection attempts"
# about seven. Read as attempts, the guard, both messages and the table agree: six
# sends, five sleeps.
_MAX_CONN_ATTEMPTS = 6
_CONN_MAX_BACKOFF_SECONDS = 120
_FD_MAX_BACKOFF_SECONDS = 30
_RESPONSE_MAX_BACKOFF_SECONDS = 30

# One GET at startup; a hung socket there would hang the whole Invocation before
# it prints anything, which reads as a broken command rather than a dead server.
_MODELS_TIMEOUT_SECONDS = 30

# The three wire fields of design 12.5/16.2. Named constants because two of them
# exist purely to refuse a server-side default, so a future edit that "simplifies"
# them away has to delete something with a name and a comment attached.
_ENCODING_FORMAT = "base64"
_EMBED_DTYPE = "float32"
_ENDIANNESS = "little"

# The decode dtype, spelled out. Never a bare ``np.float32``: that is the *client's*
# native order, so on a big-endian host it would agree with a byte-reversed payload
# instead of catching it.
_DECODE_DTYPE = "<f4"

# vLLM's context-overflow 400 says so in prose and in no other field. Design 12.6
# calls this a bug signal rather than a routine outcome -- Chunks are sized from an
# untruncated token count precisely so it cannot happen -- so the end-of-run report
# counts it apart from ordinary failures, and that needs it distinguishable here.
_CONTEXT_OVERFLOW = re.compile(rb"maximum context length|longer than the maximum|maximum_model_len|max_model_len", re.IGNORECASE)


class EmbedRequestError(RuntimeError):
    """A request came back unusable -- bad status, malformed body, or a timeout.

    Raised once the bad-response axis is spent. Terminal for the **Document**: the
    Invocation continues, because #30's own framing is that one bad PDF must never
    end a run.
    """


class TerminalDocumentError(RuntimeError):
    """A response no retry can change: a 400 context overflow, or a 413.

    Retrying a client error only spends wall-clock to get the same answer, so this
    escapes the backoff loop untouched. ``oversize`` marks the context-overflow
    case for design 12.7's separate counter; a 413 is a body-size limit somewhere in
    the HTTP path, not a context overflow, so it does not set it.
    """

    def __init__(self, message: str, *, oversize: bool = False) -> None:
        super().__init__(message)
        self.oversize = oversize


class ServerGoneError(RuntimeError):
    """The connection axis is spent: the server is not there (design 17.1).

    Terminal for the **Invocation**, and deliberately not a Document failure. A dead
    server would otherwise burn through the whole corpus at six connection attempts
    each, marking every Document failed -- and a Document recorded failed is a
    Document Resume will retry, so the damage would outlive the Invocation.
    """


def _get_sync(url: str, api_key: str | None) -> tuple[int, bytes]:
    """Blocking GET -> ``(status_code, body)``, matching :func:`apost`'s return shape."""
    request = urllib.request.Request(url, method="GET")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=_MODELS_TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        # HTTPError is an OSError subclass. Without this arm a 404 from a server that
        # is plainly alive would fall through to the connection axis and be reported
        # as a dead server after six pointless backoffs.
        return e.code, e.read()


async def _aget(url: str, api_key: str | None = None) -> tuple[int, bytes]:
    """One GET, off the event-loop thread.

    ``pipeline.apost`` is POST-only and ``/v1/models`` is a GET, so the injected
    ``post`` seam cannot carry it. The hand-rolled socket transport exists because
    connection pools deadlock at OCR request *volume*; this route is one call per
    Invocation, so stdlib urllib in a worker thread is the cheaper answer than a
    second HTTP parser to keep correct.
    """
    return await asyncio.to_thread(_get_sync, url, api_key)


class EmbedClient:
    """Speaks to one embedding server; owns the retry policy and the two rate bounds.

    Two **separate** semaphores, not one shared pool (design 12.4). ``concurrency``
    bounds ``/v1/embeddings`` only. ``/tokenize`` gets ``(concurrency * 3) // 2``
    of its own -- 96 at the default 64 -- because tokenize never reaches the GPU and
    sharing slots with it would idle the engine during CPU-side work, while it is
    still HTTP against the same API server process and so is not free either.
    Integer arithmetic rather than ``1.5 *`` keeps it exact for odd inputs (65 -> 97)
    and degrades sensibly at the bottom (1 -> 1).

    ``outstanding`` counts in-flight ``/v1/embeddings`` requests and nothing else --
    the panel's in-flight row reads it against ``vllm:num_requests_waiting``. Folding
    tokenize in would inflate it with traffic that never enters the engine scheduler
    and therefore cannot cause the queue it is being compared against.

    ``native_dim`` is the width every payload must have. It is latched from the first
    response if the caller has not pinned it, but design 12.1 step 8 pins it from the
    Adapter before the probe: a server with a different size of the same family
    loaded produces vectors that slice, normalize and store perfectly, and nothing
    downstream ever notices (design 3.6).

    ``post`` is the injection seam for tests; the default is the raw-socket
    :func:`paperscale.pipeline.apost` that the OCR and pplx paths already use.
    """

    def __init__(
        self,
        url: str,
        *,
        model: str = "",
        api_key: str | None = None,
        concurrency: int = 64,
        max_request_retries: int = 8,
        post=None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.concurrency = max(1, concurrency)
        self.tokenize_concurrency = max(1, (self.concurrency * 3) // 2)
        self.max_request_retries = max(1, max_request_retries)
        self.outstanding = 0
        self.retrying = 0
        self.native_dim: int | None = None
        self._post = apost if post is None else post
        self._loop = None
        self._embed_limit: asyncio.BoundedSemaphore | None = None
        self._tokenize_limit: asyncio.BoundedSemaphore | None = None

    def _limits(self) -> tuple[asyncio.BoundedSemaphore, asyncio.BoundedSemaphore]:
        """Bind both semaphores to the loop that is actually running.

        An asyncio primitive binds to the first loop that awaits it and refuses a
        second, and the client is constructed during startup, before ``asyncio.run``
        opens one. pplx rebuilds its module-level semaphore per run for exactly this
        reason; rebuilding on a loop change also keeps one client usable across two
        runs instead of dying on the second.
        """
        loop = asyncio.get_running_loop()
        embed_limit, tokenize_limit = self._embed_limit, self._tokenize_limit
        if self._loop is not loop or embed_limit is None or tokenize_limit is None:
            self._loop = loop
            embed_limit = asyncio.BoundedSemaphore(self.concurrency)
            tokenize_limit = asyncio.BoundedSemaphore(self.tokenize_concurrency)
            self._embed_limit, self._tokenize_limit = embed_limit, tokenize_limit
        return embed_limit, tokenize_limit

    async def _with_backoff(self, what: str, send):
        """Retry one request across three independent budgets (design 12.6).

        The axes do not share a counter, because they do not share a cause: fd
        exhaustion is a client-side resource shortage that self-resolves as in-flight
        sockets close, a connection error is the server, and a bad body is this one
        request. Spending a shared budget on the first would let a burst of EMFILE
        starve a request that never actually reached the server.

        The response arm is written **above** the ``OSError`` arm on purpose: since
        3.11 the builtin ``TimeoutError`` -- which ``asyncio.TimeoutError`` aliases --
        is an ``OSError`` subclass, so the other order routes every timeout onto the
        connection axis and, at six failures, ends the Invocation for what design
        12.6 classes as a per-request fault. ``evaluation/pplx.py`` carried exactly
        that ordering until it was corrected against this one.
        """
        attempt = conn_attempt = fd_backoff = 0
        while True:
            try:
                return await send()
            except asyncio.CancelledError:
                raise
            except TerminalDocumentError:
                # 400 context overflow and 413: no axis, no delay -- the answer will
                # not change, and the Document is the thing that failed.
                raise
            except (EmbedRequestError, asyncio.TimeoutError) as e:
                attempt += 1
                if attempt >= self.max_request_retries:
                    raise EmbedRequestError(f"embed: {what} failed {attempt} times ({type(e).__name__}: {e}); giving up on this request") from e
                delay = min(2**attempt, _RESPONSE_MAX_BACKOFF_SECONDS)
                # No jitter here, deliberately. This axis fails per request for
                # per-request reasons, so there is no herd to break up and the extra
                # variance would only slow the retry down.
                logger.warning(f"embed: {what} failed ({type(e).__name__}: {e}); attempt {attempt}/{self.max_request_retries}, sleeping {delay}s")
                await self._backoff(delay)
            except OSError as e:
                if e.errno in (errno.EMFILE, errno.ENFILE):
                    delay = min(2**fd_backoff, _FD_MAX_BACKOFF_SECONDS)
                    fd_backoff += 1
                    logger.warning(f"embed: out of file descriptors (errno {e.errno}); retrying {what} in {delay}s")
                    await self._backoff(delay)
                    continue
                conn_attempt += 1
                if conn_attempt >= _MAX_CONN_ATTEMPTS:
                    raise ServerGoneError(
                        f"embed: cannot reach {self.url} for {what} after {_MAX_CONN_ATTEMPTS} connection attempts "
                        f"({type(e).__name__}: {e}); stopping the invocation rather than failing every remaining document"
                    ) from e
                # Full jitter -- new to paperscale, and load-bearing at this
                # concurrency. A server restart fails all 64 in-flight requests from a
                # single cause; unjittered, all 64 wake in the same instant and
                # stampede a server that is still loading weights.
                delay = random.uniform(0, min(10 * 2 ** (conn_attempt - 1), _CONN_MAX_BACKOFF_SECONDS))
                logger.warning(f"embed: connection error ({type(e).__name__}: {e}); attempt {conn_attempt}/{_MAX_CONN_ATTEMPTS}, sleeping {delay:.1f}s")
                await self._backoff(delay)

    async def _backoff(self, delay: float) -> None:
        """Sleep out one retry axis, counted in ``retrying`` for as long as it lasts.

        Design 13.2 asks for a gauge, not a tally: the row answers "how many requests
        are asleep right now", so it has to fall back to zero when the server recovers.
        A tally would only ever climb, and a panel row that never falls stops carrying
        information about the present.

        All three axes count. A request waiting on a file descriptor is as stalled as
        one waiting on a server that has gone; the operator watching the panel wants
        the number of requests that are not moving, not a taxonomy of why.
        """
        self.retrying += 1
        try:
            await asyncio.sleep(delay)
        finally:
            self.retrying -= 1

    async def models(self) -> tuple[str, int]:
        """``GET /v1/models`` -> ``(served_model_id, max_model_len)``.

        Both numbers are preconditions: the id is what the panel header and the
        ``model_id`` provenance record (never the string the operator typed, which
        may name a symlink or an alias), and ``max_model_len`` is the only hard upper
        bound on the context length (design 4.2).
        """
        served, max_model_len = await self._with_backoff("GET /v1/models", self._send_models)
        if not self.model:
            # Latch it so no request can go out naming a model the server does not
            # serve. Only fires when the caller left it unset, so an explicit
            # ``--embed-model``-derived id is never overwritten.
            self.model = served
        return served, max_model_len

    async def _send_models(self) -> tuple[str, int]:
        status, raw = await _aget(f"{self.url}/v1/models", self.api_key)
        if status != 200:
            raise EmbedRequestError(f"embed: GET /v1/models returned {status}: {(raw or b'')[:300]!r}")
        try:
            card = json.loads(raw)["data"][0]
            served = card["id"]
            max_model_len = card["max_model_len"]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise EmbedRequestError(f"embed: malformed /v1/models response: {type(e).__name__}: {e}") from e
        if not served or not isinstance(max_model_len, (int, float)):
            raise EmbedRequestError(f"embed: /v1/models gave id={served!r} max_model_len={max_model_len!r}; both are required at startup")
        return served, int(max_model_len)

    async def tokenize(self, text: str) -> int:
        """``POST /tokenize`` -> the exact token count for the model actually loaded.

        Asking the server is what keeps the count honest: there is no second
        tokenizer to drift from the server's, and the whole chunking design rests on
        vLLM *erroring* on overflow rather than truncating, so a guessed count is
        unsafe rather than merely imprecise (design 5.2, 12.6).

        ``add_special_tokens`` is deliberately not sent -- the server's own default is
        what ``/v1/embeddings`` will apply to the same text, and pinning it here could
        only make the two disagree.
        """
        return await self._with_backoff("POST /tokenize", lambda: self._send_tokenize(text))

    async def _send_tokenize(self, text: str) -> int:
        body = {"model": self.model, "prompt": text}
        _, limit = self._limits()
        # Not counted in ``outstanding``: this route is handled in the API server
        # process and never enters the engine scheduler.
        async with limit:
            status, raw = await self._post(f"{self.url}/tokenize", json_data=body, api_key=self.api_key)
        if status == 413:
            # Design 12.6 makes `413` terminal for the Document without a retry, and 5.2 puts
            # `/tokenize` under that same taxonomy. Left on the retryable branch it repeats the
            # shape of the `/v1/tokenize` 404 this file carried until `ddce48d`: an answer that
            # cannot change, spending the whole budget to arrive back where it started.
            raise TerminalDocumentError(f"embed: /tokenize returned 413: {(raw or b'')[:300]!r}")
        if status != 200:
            raise EmbedRequestError(f"embed: /tokenize returned {status}: {(raw or b'')[:300]!r}")
        try:
            parsed = json.loads(raw)
        except ValueError as e:
            raise EmbedRequestError(f"embed: malformed /tokenize response: {type(e).__name__}: {e}") from e
        count = parsed.get("count") if isinstance(parsed, dict) else None
        if count is None:
            tokens = parsed.get("tokens") if isinstance(parsed, dict) else None
            if not isinstance(tokens, list):
                raise EmbedRequestError(f"embed: /tokenize response carries neither 'count' nor 'tokens': {parsed!r}")
            count = len(tokens)
        try:
            return int(count)
        except (TypeError, ValueError) as e:
            # A 200 carrying a non-numeric `count` is a bad response like any other, but this
            # conversion sat outside the wrapper above, so the bare `ValueError` escaped design
            # 12.6's taxonomy entirely: a route whose whole contract is that it *shares* that
            # taxonomy failed the Invocation on one bad body instead of retrying the request.
            raise EmbedRequestError(f"embed: /tokenize returned a non-numeric count {count!r}: {type(e).__name__}: {e}") from e

    async def embed(self, texts: list[str]):
        """``POST /v1/embeddings`` -> ``(len(texts), native_dim)`` float32, in input order.

        Native width, not ``stored_dim``: the MRL slice is client-side, so every
        response arrives full-width (design 6.1) and that width doubles as the
        wrong-model assertion.
        """
        import numpy as np

        if not texts:
            # Never send an empty ``input`` -- vLLM rejects it, and a zero-Chunk
            # Document is a recorded outcome rather than a failure (design 5.5).
            return np.zeros((0, self.native_dim or 0), dtype=np.float32)
        encoded = await self._with_backoff("POST /v1/embeddings", lambda: self._send_embeddings(texts))
        return self._decode(encoded)

    async def _send_embeddings(self, texts: list[str]) -> list:
        """One ``/v1/embeddings`` call -> the base64 payloads, ordered by ``index``.

        **The body is exactly design 12.5's, and neither extra parameter may be left
        to its default.** ``endianness`` defaults to ``"native"``, which is the
        *serving host's* byte order -- decided on the server, stated nowhere in the
        response, while the client's ``frombuffer`` would use its own. A byte-reversed
        IEEE-754 word is usually still an ordinary finite number, arrives in the right
        count, and normalizes to unit length, so nothing downstream detects it and
        both Sinks fill with plausible garbage. ``"little"`` makes the server byteswap
        only if it must, and is a no-op on x86-64 and aarch64. ``embed_dtype`` defaults
        to ``float32`` today, so pinning it is belt-and-braces -- but the other four
        values are lossy and ``float16`` returns *half* the bytes, which ``"<f4"``
        would decode as a vector of half the width rather than raising.

        vLLM's ``OpenAIBaseModel`` sets ``extra="allow"`` and merely debug-logs keys it
        does not know, so sending these fields is always safe but acceptance proves
        nothing. That is why :meth:`_decode`'s width check is not optional.

        ``truncate_prompt_tokens`` and ``dimensions`` never appear. vLLM's default on
        oversized input is to *error*, which is the safe case, so the enforcement is
        that the parameter is absent rather than that some flag is set: silent
        truncation would leave stored offsets describing text that was never embedded.
        """
        body = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": _ENCODING_FORMAT,
            "embed_dtype": _EMBED_DTYPE,
            "endianness": _ENDIANNESS,
        }
        limit, _ = self._limits()
        # The semaphore wraps only the send, as pplx and the OCR path both do, so
        # parsing never holds a slot the server could be using. ``outstanding`` moves
        # with it rather than with the call, so it means "the server has this" and not
        # "someone is queued behind the semaphore".
        async with limit:
            self.outstanding += 1
            try:
                status, raw = await self._post(f"{self.url}/v1/embeddings", json_data=body, api_key=self.api_key)
            finally:
                self.outstanding -= 1
        snippet = (raw or b"")[:300]
        # Design 12.6 scopes terminal-without-retry, for the Document, to exactly two
        # responses: "a `400` context overflow, and `413`". A 400 that says nothing
        # about context length is a request the server would not parse, which is a bad
        # response like any other -- it belongs on the bad-response axis below, rather
        # than in a taxonomy wider than the design draws. Failing the whole status
        # would deny such a request the eight attempts one of which may well succeed.
        if status == 400 and _CONTEXT_OVERFLOW.search(raw or b""):
            raise TerminalDocumentError(f"embed: /v1/embeddings returned 400 context overflow: {snippet!r}", oversize=True)
        if status == 413:
            raise TerminalDocumentError(f"embed: /v1/embeddings returned 413: {snippet!r}")
        if status != 200:
            raise EmbedRequestError(f"embed: /v1/embeddings returned {status}: {snippet!r}")
        try:
            data = json.loads(raw)["data"]
            # vLLM fans one ``input`` array into N independent engine requests and
            # merges them as they finish, so arrival order is not input order. The
            # ``index`` field is the only thing that ties a payload back to its Chunk.
            # No positional fallback. `.get("index", i)` defaulted to the arrival order the
            # line above calls unreliable, and a *partial* omission is worse than a total one:
            # it sorts real indices against positions on a single scale and interleaves them.
            # A missing key raises `KeyError` into the handler below, which is where a response
            # this malformed belongs.
            order = sorted(range(len(data)), key=lambda i: data[i]["index"])
            encoded = [data[i]["embedding"] for i in order]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise EmbedRequestError(f"embed: malformed /v1/embeddings response: {type(e).__name__}: {e}") from e
        if len(encoded) != len(texts):
            raise EmbedRequestError(f"embed: sent {len(texts)} inputs and got {len(encoded)} embeddings back")
        return encoded

    def _decode(self, encoded: list):
        """base64 -> ``(n, native_dim)`` float32, with the width backstop from design 16.2.

        A wrong width fails the **Document** without a retry: the width is a property
        of the model the server loaded, so the next eight attempts return the same
        bytes, and a short vector is exactly the silent-wrongness class this design
        refuses -- it slices, normalizes and stores like any other.
        """
        import numpy as np

        vectors = []
        for i, item in enumerate(encoded):
            if not isinstance(item, str):
                # A list of floats means ``encoding_format`` was ignored -- and a build
                # that ignores that one ignored ``embed_dtype`` and ``endianness`` too,
                # so the bytes carry no guarantee at all. Refuse rather than accept.
                raise TerminalDocumentError(f"embed: embedding {i} came back as {type(item).__name__}, not a base64 string; the server ignored encoding_format")
            raw = base64.b64decode(item)
            expected = self.native_dim
            if expected is None:
                if len(raw) % 4:
                    raise TerminalDocumentError(f"embed: embedding {i} is {len(raw)} bytes, not a whole number of float32 words")
                expected = self.native_dim = len(raw) // 4
            if len(raw) != expected * 4:
                raise TerminalDocumentError(
                    f"embed: embedding {i} is {len(raw)} bytes, expected {expected * 4} ({expected} x float32); "
                    "the server is serving a different model or a different embed_dtype"
                )
            vectors.append(np.frombuffer(raw, dtype=_DECODE_DTYPE))
        # ``astype`` after the stack, never in place of the explicit ``"<f4"``: the
        # decode stays byte-order-explicit, and this only puts the result in the
        # client's own order for the arithmetic in vectors.py. A no-op on any
        # little-endian host, which is every host this design targets.
        return np.stack(vectors).astype(np.float32, copy=False)
