"""Tests for the embedding server client.

Everything here runs against a fake transport injected as ``post`` -- no server, no
sockets. ``asyncio.sleep`` is patched throughout the retry tests so the *shape* of
each backoff is asserted rather than waited out; a faithful connection-axis run
would otherwise take about four minutes.

The one exception is :class:`LiveBase64RoundTripTest`, which discharges design 16.2's
live obligation and skips unless ``PAPERSCALE_LIVE_EMBED_URL`` names a running vLLM.
The default suite never needs a server.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import os
import unittest
import urllib.error
import urllib.request
from unittest import mock

from paperscale.embed import client as client_mod
from paperscale.embed.client import EmbedClient, EmbedRequestError, ServerGoneError, TerminalDocumentError

_EMBEDDINGS = "http://vllm/v1/embeddings"


def _b64(values, dtype: str = "<f4") -> str:
    import numpy as np

    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode()


def _ok(vectors, dtype: str = "<f4", indices=None):
    """A 200 shaped like vLLM's: ``{"data": [{"index": i, "embedding": "<base64>"}]}``."""
    if indices is None:
        indices = range(len(vectors))
    data = [{"index": i, "embedding": _b64(v, dtype)} for i, v in zip(indices, vectors)]
    return 200, json.dumps({"data": data}).encode()


def _err(status: int, message: str = "boom"):
    return status, json.dumps({"error": {"message": message}}).encode()


class _Transport:
    """Stand-in for :func:`paperscale.pipeline.apost`.

    Replays a script of outcomes -- an exception instance is raised, a
    ``(status, body)`` tuple is returned -- and repeats the last entry forever, so a
    test that wants "fails every time" supplies exactly one. Every body is recorded,
    which is how the wire-format tests see what actually went out.
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict]] = []
        self.in_flight: list[int] = []
        self.client: EmbedClient | None = None

    @property
    def bodies(self) -> list[dict]:
        return [body for _, body in self.calls]

    async def __call__(self, url, json_data, api_key=None):
        self.calls.append((url, json_data))
        if self.client is not None:
            self.in_flight.append(self.client.outstanding)
        outcome = self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _NoSleep:
    """Swap ``asyncio.sleep`` for a recorder; the delays become the assertion."""

    def __init__(self):
        self.delays: list[float] = []

    async def _sleep(self, delay, result=None):
        self.delays.append(delay)
        return result

    def __enter__(self):
        self._patch = mock.patch("asyncio.sleep", new=self._sleep)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class _Uniform:
    """Records every ``random.uniform(a, b)`` and returns a fixed draw.

    A recorder rather than a seeded PRNG: what matters is that the connection axis
    asks for a *range* (and which range), and that the response axis never asks at
    all. Asserting on drawn values would test the stdlib.
    """

    def __init__(self, value: float = 0.0):
        self.calls: list[tuple[float, float]] = []
        self.value = value

    def __call__(self, a, b):
        self.calls.append((a, b))
        return self.value


def _client(transport, **kwargs) -> EmbedClient:
    kwargs.setdefault("model", "served-id")
    c = EmbedClient("http://vllm", post=transport, **kwargs)
    if isinstance(transport, _Transport):
        transport.client = c
    return c


class WireFormatTest(unittest.TestCase):
    def test_every_embedding_body_carries_all_three_fields_none_defaulted(self):
        transport = _Transport(_ok([[1.0, 2.0]]))
        c = _client(transport)
        asyncio.run(c.embed(["hello"]))

        url, body = transport.calls[0]
        self.assertEqual(url, _EMBEDDINGS)
        self.assertEqual(body["encoding_format"], "base64")
        self.assertEqual(body["embed_dtype"], "float32")
        self.assertEqual(body["endianness"], "little")
        self.assertEqual(body["input"], ["hello"])
        self.assertEqual(body["model"], "served-id")
        # Exactly these five keys: an extra field is a fact the server may or may not
        # honour, and design 12.5 pins the body rather than a minimum.
        self.assertEqual(set(body), {"model", "input", "encoding_format", "embed_dtype", "endianness"})

    def test_truncate_prompt_tokens_and_dimensions_never_appear(self):
        # Exercise a retried embed and a tokenize, so every body the client can build
        # is inspected -- not just the happy-path one.
        transport = _Transport(_err(500), _ok([[1.0, 2.0]]), (200, json.dumps({"count": 4}).encode()))
        c = _client(transport)
        with _NoSleep():
            asyncio.run(c.embed(["a"]))
            asyncio.run(c.tokenize("a"))
        self.assertEqual(len(transport.bodies), 3)
        for body in transport.bodies:
            self.assertNotIn("truncate_prompt_tokens", body)
            self.assertNotIn("dimensions", body)

    def test_decoder_reads_little_endian_and_never_the_client_default(self):
        import numpy as np

        # Hand the client big-endian bytes. A decoder that trusted the platform would
        # read them as [1.0, 2.0] on x86-64; the pinned "<f4" must reproduce exactly
        # what frombuffer("<f4") sees, garbage values and all. That is the whole point
        # of sending endianness="little": nothing in the response says which it was.
        raw = np.asarray([1.0, 2.0], dtype=">f4").tobytes()
        transport = _Transport((200, json.dumps({"data": [{"index": 0, "embedding": base64.b64encode(raw).decode()}]}).encode()))
        c = _client(transport)
        got = asyncio.run(c.embed(["x"]))

        expected = np.frombuffer(raw, dtype="<f4")
        self.assertTrue(np.array_equal(got[0], expected))
        self.assertFalse(np.array_equal(got[0], np.asarray([1.0, 2.0], dtype=np.float32)))

    def test_round_trip_of_a_little_endian_payload_is_exact(self):
        import numpy as np

        values = [[0.5, -1.25, 3.0], [1.0, 0.0, -0.75]]
        transport = _Transport(_ok(values))
        c = _client(transport)
        got = asyncio.run(c.embed(["a", "b"]))
        self.assertEqual(got.shape, (2, 3))
        self.assertEqual(got.dtype, np.float32)
        self.assertTrue(np.array_equal(got, np.asarray(values, dtype=np.float32)))

    def test_payloads_are_ordered_by_index_not_arrival(self):
        import numpy as np

        # vLLM fans one input array into N engine requests and merges them as they
        # finish, so the second Chunk can come back first.
        transport = _Transport(_ok([[2.0], [1.0]], indices=[1, 0]))
        c = _client(transport)
        got = asyncio.run(c.embed(["first", "second"]))
        self.assertTrue(np.array_equal(got, np.asarray([[1.0], [2.0]], dtype=np.float32)))

    def test_wrong_width_fails_the_document_without_retrying(self):
        transport = _Transport(_ok([[1.0, 2.0]]))
        c = _client(transport)
        c.native_dim = 4
        with _NoSleep() as sleeper, self.assertRaises(TerminalDocumentError) as caught:
            asyncio.run(c.embed(["x"]))
        self.assertIn("8 bytes", str(caught.exception))
        self.assertIn("16", str(caught.exception))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(sleeper.delays, [])

    def test_native_dim_latches_from_the_first_response_and_then_bites(self):
        transport = _Transport(_ok([[1.0, 2.0, 3.0]]), _ok([[1.0, 2.0]]))
        c = _client(transport)
        asyncio.run(c.embed(["x"]))
        self.assertEqual(c.native_dim, 3)
        with _NoSleep(), self.assertRaises(TerminalDocumentError):
            asyncio.run(c.embed(["y"]))

    def test_float_list_response_fails_the_document(self):
        # encoding_format ignored means embed_dtype and endianness were ignored too,
        # so the bytes carry no guarantee at all.
        body = json.dumps({"data": [{"index": 0, "embedding": [1.0, 2.0]}]}).encode()
        transport = _Transport((200, body))
        c = _client(transport)
        with _NoSleep(), self.assertRaises(TerminalDocumentError) as caught:
            asyncio.run(c.embed(["x"]))
        self.assertIn("encoding_format", str(caught.exception))

    def test_empty_input_sends_no_request(self):
        transport = _Transport(_ok([[1.0]]))
        c = _client(transport)
        c.native_dim = 4
        got = asyncio.run(c.embed([]))
        self.assertEqual(got.shape, (0, 4))
        self.assertEqual(transport.calls, [])


class RetryAxesTest(unittest.TestCase):
    def test_fd_exhaustion_is_unbounded_and_consumes_no_attempt(self):
        # max_request_retries=2 -- if fd exhaustion spent the response budget, this
        # would raise on the second call instead of succeeding on the thirteenth.
        emfile = OSError(errno.EMFILE, "Too many open files")
        transport = _Transport(*([emfile] * 12), _ok([[1.0]]))
        c = _client(transport, max_request_retries=2)
        with _NoSleep() as sleeper:
            asyncio.run(c.embed(["x"]))
        self.assertEqual(len(transport.calls), 13)
        self.assertEqual(sleeper.delays, [1, 2, 4, 8, 16, 30, 30, 30, 30, 30, 30, 30])

    def test_retrying_rises_inside_a_backoff_and_falls_back_after_it(self):
        # Design 13.2 wants "requests currently in backoff", so the number has to come
        # down again. A tally would satisfy an "it went up" assertion and still be the
        # wrong number: the panel row would climb all Invocation and never say whether
        # anything is stalled *now*. Both axes that sleep are counted here.
        transport = _Transport(ConnectionRefusedError("no route"), OSError(errno.EMFILE, "Too many open files"), _ok([[1.0]]))
        c = _client(transport)
        during: list[int] = []

        async def sleep(delay, result=None):
            during.append(c.retrying)
            return result

        self.assertEqual(c.retrying, 0)
        with mock.patch("asyncio.sleep", new=sleep):
            asyncio.run(c.embed(["x"]))
        self.assertEqual(during, [1, 1])
        self.assertEqual(c.retrying, 0)

    def test_retrying_falls_back_to_zero_when_an_axis_gives_up(self):
        # The decrement has to survive the raise, or a server that goes away leaves the
        # row pinned at its high-water mark for the rest of the Invocation.
        transport = _Transport(ConnectionRefusedError("no route"))
        c = _client(transport)
        with _NoSleep(), self.assertRaises(ServerGoneError):
            asyncio.run(c.embed(["x"]))
        self.assertEqual(c.retrying, 0)

    def test_connection_axis_exhausts_into_server_gone(self):
        transport = _Transport(ConnectionRefusedError("no route"))
        c = _client(transport)
        with _NoSleep() as sleeper, self.assertRaises(ServerGoneError):
            asyncio.run(c.embed(["x"]))
        # Design 12.6 budgets the connection axis at 6. Six sends, five backoffs --
        # the sixth failure is the one that gives up, which is what the ServerGoneError
        # message has always claimed and what the response axis does with its own 8.
        self.assertEqual(len(transport.calls), 6)
        self.assertEqual(len(sleeper.delays), 5)

    def test_connection_axis_uses_full_jitter_over_the_capped_window(self):
        transport = _Transport(ConnectionRefusedError("no route"))
        c = _client(transport)
        uniform = _Uniform()
        with _NoSleep(), mock.patch("random.uniform", new=uniform), self.assertRaises(ServerGoneError):
            asyncio.run(c.embed(["x"]))
        # uniform(0, min(10 * 2**(n-1), 120)) -- full jitter, floor stays at zero so a
        # restarted server is probed early by someone rather than by all 64 at once.
        self.assertEqual(uniform.calls, [(0, 10), (0, 20), (0, 40), (0, 80), (0, 120)])

    def test_response_axis_spends_max_request_retries_then_fails_the_document(self):
        transport = _Transport(_err(500))
        c = _client(transport)
        with _NoSleep() as sleeper, self.assertRaises(EmbedRequestError):
            asyncio.run(c.embed(["x"]))
        self.assertEqual(len(transport.calls), 8)
        self.assertEqual(sleeper.delays, [2, 4, 8, 16, 30, 30, 30])

    def test_response_axis_has_no_jitter(self):
        transport = _Transport(_err(500))
        c = _client(transport, max_request_retries=3)
        uniform = _Uniform()
        with _NoSleep() as sleeper, mock.patch("random.uniform", new=uniform), self.assertRaises(EmbedRequestError):
            asyncio.run(c.embed(["x"]))
        self.assertEqual(uniform.calls, [])
        self.assertEqual(sleeper.delays, [2, 4])

    def test_a_timeout_lands_on_the_response_axis(self):
        # Since 3.11 asyncio.TimeoutError *is* builtins.TimeoutError, which subclasses
        # OSError. Written in pplx's clause order, every timeout would be read as a
        # dead server and end the Invocation after six tries.
        transport = _Transport(asyncio.TimeoutError())
        c = _client(transport, max_request_retries=3)
        with _NoSleep(), self.assertRaises(EmbedRequestError):
            asyncio.run(c.embed(["x"]))
        self.assertEqual(len(transport.calls), 3)

    def test_a_malformed_body_is_retried_on_the_response_axis(self):
        transport = _Transport((200, b"not json"), _ok([[1.0]]))
        c = _client(transport)
        with _NoSleep() as sleeper:
            asyncio.run(c.embed(["x"]))
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(sleeper.delays, [2])

    def test_short_response_is_retried(self):
        transport = _Transport(_ok([[1.0]]), _ok([[1.0], [2.0]]))
        c = _client(transport)
        with _NoSleep():
            got = asyncio.run(c.embed(["a", "b"]))
        self.assertEqual(got.shape, (2, 1))
        self.assertEqual(len(transport.calls), 2)

    def test_context_overflow_400_is_terminal_and_flagged_oversize(self):
        transport = _Transport(_err(400, "This model's maximum context length is 32768 tokens, however you requested 40000"))
        c = _client(transport)
        with _NoSleep() as sleeper, self.assertRaises(TerminalDocumentError) as caught:
            asyncio.run(c.embed(["x"]))
        self.assertTrue(caught.exception.oversize)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(sleeper.delays, [])

    def test_413_is_terminal_but_not_oversize(self):
        transport = _Transport((413, b"Request Entity Too Large"))
        c = _client(transport)
        with _NoSleep(), self.assertRaises(TerminalDocumentError) as caught:
            asyncio.run(c.embed(["x"]))
        self.assertFalse(caught.exception.oversize)
        self.assertEqual(len(transport.calls), 1)

    def test_a_400_that_is_not_a_context_overflow_stays_on_the_response_axis(self):
        # Design 12.6 scopes the terminal outcome to "a `400` context overflow, and
        # `413`" -- not to the status. A malformed-request 400 naming no context length
        # is a bad response, so it is retried, and the failure that finally lands is an
        # EmbedRequestError rather than a TerminalDocumentError.
        transport = _Transport(_err(400, "model `wrong` does not exist"))
        c = _client(transport, max_request_retries=3)
        with _NoSleep() as sleeper, self.assertRaises(EmbedRequestError) as caught:
            asyncio.run(c.embed(["x"]))
        self.assertNotIsInstance(caught.exception, TerminalDocumentError)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual(sleeper.delays, [2, 4])

    def test_a_non_overflow_400_can_be_retried_into_a_success(self):
        # The point of the axis: the very next attempt is allowed to work. Under the
        # old status-wide rule this Document died on the first response.
        transport = _Transport(_err(400, "invalid request body"), _ok([[1.0, 2.0]]))
        c = _client(transport)
        with _NoSleep():
            got = asyncio.run(c.embed(["x"]))
        self.assertEqual(got.shape, (1, 2))
        self.assertEqual(len(transport.calls), 2)

    def test_tokenize_shares_the_taxonomy(self):
        transport = _Transport(ConnectionRefusedError("no route"))
        c = _client(transport)
        with _NoSleep(), self.assertRaises(ServerGoneError):
            asyncio.run(c.tokenize("x"))
        self.assertEqual(len(transport.calls), 6)


class RoutesTest(unittest.TestCase):
    def test_tokenize_reads_the_exact_count(self):
        transport = _Transport((200, json.dumps({"count": 1234, "max_model_len": 32768}).encode()))
        c = _client(transport)
        self.assertEqual(asyncio.run(c.tokenize("some text")), 1234)
        url, body = transport.calls[0]
        self.assertEqual(url, "http://vllm/tokenize")
        self.assertEqual(body, {"model": "served-id", "prompt": "some text"})

    def test_tokenize_falls_back_to_the_token_list(self):
        transport = _Transport((200, json.dumps({"tokens": [1, 2, 3]}).encode()))
        self.assertEqual(asyncio.run(_client(transport).tokenize("x")), 3)

    def test_models_returns_the_served_id_and_max_model_len(self):
        body = json.dumps({"data": [{"id": "Qwen/Qwen3-Embedding-0.6B", "max_model_len": 32768}]}).encode()

        async def fake_get(url, api_key=None):
            self.assertEqual(url, "http://vllm/v1/models")
            return 200, body

        with mock.patch.object(client_mod, "_aget", new=fake_get):
            served, max_model_len = asyncio.run(EmbedClient("http://vllm/", model="m").models())
        self.assertEqual(served, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(max_model_len, 32768)

    def test_models_latches_the_model_only_when_unset(self):
        body = json.dumps({"data": [{"id": "served-by-vllm", "max_model_len": 8192}]}).encode()

        async def fake_get(url, api_key=None):
            return 200, body

        with mock.patch.object(client_mod, "_aget", new=fake_get):
            blank = EmbedClient("http://vllm")
            asyncio.run(blank.models())
            self.assertEqual(blank.model, "served-by-vllm")

            explicit = EmbedClient("http://vllm", model="operator-choice")
            asyncio.run(explicit.models())
            self.assertEqual(explicit.model, "operator-choice")

    def test_models_error_status_is_not_read_as_a_dead_server(self):
        async def fake_get(url, api_key=None):
            return 404, b"not found"

        with mock.patch.object(client_mod, "_aget", new=fake_get), _NoSleep(), self.assertRaises(EmbedRequestError):
            asyncio.run(EmbedClient("http://vllm", max_request_retries=2).models())


class ConcurrencyTest(unittest.TestCase):
    def test_tokenize_bound_is_three_halves_of_concurrency(self):
        for concurrency, expected in ((64, 96), (65, 97), (2, 3), (1, 1)):
            c = EmbedClient("http://vllm", concurrency=concurrency)
            self.assertEqual(c.concurrency, concurrency)
            self.assertEqual(c.tokenize_concurrency, expected)

    def test_tokenize_does_not_wait_on_embedding_slots(self):
        """Both embedding slots are held open; tokenize must still complete."""

        async def scenario():
            gate = asyncio.Event()
            saturated = asyncio.Event()
            sent = {"embed": 0, "tokenize": 0}

            async def post(url, json_data, api_key=None):
                if url.endswith("/v1/embeddings"):
                    sent["embed"] += 1
                    if sent["embed"] >= 2:
                        saturated.set()
                    await gate.wait()
                    return _ok([[1.0]])
                sent["tokenize"] += 1
                return 200, json.dumps({"count": 7}).encode()

            c = EmbedClient("http://vllm", model="m", concurrency=2, post=post)
            embeds = [asyncio.create_task(c.embed(["a"])) for _ in range(2)]
            await saturated.wait()
            counts = await asyncio.gather(*(c.tokenize("x") for _ in range(3)))
            gate.set()
            await asyncio.gather(*embeds)
            return counts, sent, c.outstanding

        counts, sent, outstanding = asyncio.run(scenario())
        self.assertEqual(counts, [7, 7, 7])
        self.assertEqual(sent["tokenize"], 3)
        self.assertEqual(outstanding, 0)

    def test_outstanding_rises_with_in_flight_embeddings_and_falls_back(self):
        async def scenario():
            gate = asyncio.Event()
            arrived = asyncio.Event()
            state = {"n": 0, "peak": 0}

            async def post(url, json_data, api_key=None):
                state["n"] += 1
                state["peak"] = max(state["peak"], c.outstanding)
                if state["n"] >= 4:
                    arrived.set()
                await gate.wait()
                return _ok([[1.0]])

            c = EmbedClient("http://vllm", model="m", concurrency=4, post=post)
            self.assertEqual(c.outstanding, 0)
            tasks = [asyncio.create_task(c.embed(["a"])) for _ in range(4)]
            await arrived.wait()
            mid = c.outstanding
            gate.set()
            await asyncio.gather(*tasks)
            return mid, state["peak"], c.outstanding

        mid, peak, after = asyncio.run(scenario())
        self.assertEqual(mid, 4)
        self.assertEqual(peak, 4)
        self.assertEqual(after, 0)

    def test_outstanding_ignores_tokenize(self):
        observed = []

        async def post(url, json_data, api_key=None):
            observed.append(c.outstanding)
            return 200, json.dumps({"count": 3}).encode()

        c = EmbedClient("http://vllm", model="m", post=post)
        asyncio.run(c.tokenize("x"))
        self.assertEqual(observed, [0])
        self.assertEqual(c.outstanding, 0)

    def test_outstanding_falls_back_after_a_failed_send(self):
        transport = _Transport(ConnectionRefusedError("no route"), _ok([[1.0]]))
        c = _client(transport)
        with _NoSleep():
            asyncio.run(c.embed(["x"]))
        # Each attempt raises inside the semaphore; a counter that only decremented on
        # success would climb forever and the panel would report a saturated server.
        self.assertEqual(transport.in_flight, [1, 1])
        self.assertEqual(c.outstanding, 0)

    def test_one_client_survives_a_second_event_loop(self):
        transport = _Transport(_ok([[1.0]]))
        c = _client(transport)
        asyncio.run(c.embed(["a"]))
        asyncio.run(c.embed(["b"]))
        self.assertEqual(len(transport.calls), 2)


# Design 16.2's live obligation. It is opt-in because the default suite must not need
# a GPU: without the variable every test in the class below skips, and the skip reason
# is the whole recipe for running it.
_LIVE_URL = os.environ.get("PAPERSCALE_LIVE_EMBED_URL", "").rstrip("/")
_LIVE_SKIP = (
    "live server test, off by default. To run it: start a pooling-task vLLM, e.g. "
    "`vllm-omni serve Qwen/Qwen3-Embedding-0.6B --port 8000`, then "
    "`PAPERSCALE_LIVE_EMBED_URL=http://localhost:8000 poetry run pytest -q tests/embed/test_client.py`. "
    "Set PAPERSCALE_LIVE_EMBED_MODEL to override the model id, which otherwise comes from GET /v1/models."
)
_LIVE_TEXT = "The quick brown fox jumps over the lazy dog."
_LIVE_TIMEOUT_SECONDS = 180


def _live_post(route: str, body: dict) -> tuple[int, bytes]:
    """One POST to the live server, stdlib only -- no paperscale code in the way.

    Routes are spelled in full at the call sites, because the one that surprises
    people is ``/tokenize``: vLLM mounts tokenize and detokenize at the **top level**
    and only the OpenAI-compatible routes carry ``/v1``.
    """
    request = urllib.request.Request(
        f"{_LIVE_URL}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_LIVE_TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@unittest.skipUnless(_LIVE_URL, _LIVE_SKIP)
class LiveBase64RoundTripTest(unittest.TestCase):
    """Design 16.2's live check: are ``embed_dtype`` and ``endianness`` HONORED?

    16.2 settled the wire format by reading vLLM's source, and left exactly one thing
    open: *"a live check must assert on the response, not on the absence of an error"*.
    That distinction is the whole point. vLLM's ``OpenAIBaseModel`` sets
    ``extra="allow"`` and merely debug-logs keys it does not know, so a build that has
    never heard of ``endianness`` accepts the field, ignores it, falls back to
    ``"native"``, and on a big-endian host hands back byte-reversed floats that arrive
    in the right count and normalize to unit length. Acceptance proves nothing; only
    the bytes do.

    Three assertions carry it, all against the same one input:

    * the base64 payload decodes bit-equal to the ``"float"`` response, at Native
      dimension x 4 bytes;
    * ``embed_dtype`` is honored -- ``float16`` returns **half** the bytes, which the
      pinned ``"<f4"`` would otherwise decode as a vector of half the width;
    * ``endianness`` is honored -- ``"big"`` returns the exact byte-swap of
      ``"little"``, rather than the identical bytes.
    """

    @classmethod
    def setUpClass(cls):
        status, raw = 0, b""
        try:
            request = urllib.request.Request(f"{_LIVE_URL}/v1/models", method="GET")
            with urllib.request.urlopen(request, timeout=_LIVE_TIMEOUT_SECONDS) as response:
                status, raw = response.status, response.read()
        except OSError as e:  # a deliberately opted-in run must fail loudly, never skip
            raise AssertionError(f"PAPERSCALE_LIVE_EMBED_URL={_LIVE_URL!r} is not answering GET /v1/models: {e}") from e
        assert status == 200, f"GET /v1/models returned {status}: {raw[:300]!r}"
        card = json.loads(raw)["data"][0]
        cls.model = os.environ.get("PAPERSCALE_LIVE_EMBED_MODEL") or card["id"]

        status, raw = _live_post("/v1/embeddings", {"model": cls.model, "input": [_LIVE_TEXT], "encoding_format": "float"})
        assert status == 200, f"the float request returned {status}: {raw[:300]!r}"
        cls.floats = json.loads(raw)["data"][0]["embedding"]
        cls.native_dimension = len(cls.floats)

        status, raw = _live_post(
            "/v1/embeddings",
            {"model": cls.model, "input": [_LIVE_TEXT], "encoding_format": "base64", "embed_dtype": "float32", "endianness": "little"},
        )
        assert status == 200, f"the base64 request returned {status}: {raw[:300]!r}"
        cls.payload = json.loads(raw)["data"][0]["embedding"]

    def test_base64_decodes_bit_equal_to_the_float_response(self):
        import numpy as np

        self.assertIsInstance(self.payload, str, "encoding_format was ignored: the response is still a list of floats")
        raw = base64.b64decode(self.payload)
        self.assertEqual(len(raw), self.native_dimension * 4, "the payload is not Native dimension x 4 raw float32 bytes")
        decoded = np.frombuffer(raw, dtype="<f4")
        reference = np.asarray(self.floats, dtype=np.float32)
        # Bit-equal, not almost-equal: base64 of float32 is lossless, so any drift here
        # is a decode that disagrees with the server rather than rounding.
        self.assertEqual(decoded.tobytes(), reference.tobytes())

    def test_embed_dtype_is_honored_and_not_silently_ignored(self):
        status, raw = _live_post(
            "/v1/embeddings",
            {"model": self.model, "input": [_LIVE_TEXT], "encoding_format": "base64", "embed_dtype": "float16", "endianness": "little"},
        )
        if status != 200:
            # Rejecting the value is also proof the field is read. What must never
            # happen is a 200 carrying float32 bytes, which would mean "<f4" is right
            # only by luck.
            return
        half = base64.b64decode(json.loads(raw)["data"][0]["embedding"])
        self.assertEqual(
            len(half),
            self.native_dimension * 2,
            f"embed_dtype is ignored: float16 returned {len(half)} bytes, not {self.native_dimension * 2}; float32 is right only by luck",
        )

    def test_endianness_is_honored_and_not_silently_ignored(self):
        import numpy as np

        status, raw = _live_post(
            "/v1/embeddings",
            {"model": self.model, "input": [_LIVE_TEXT], "encoding_format": "base64", "embed_dtype": "float32", "endianness": "big"},
        )
        if status != 200:
            return  # rejected rather than ignored; see the sibling test
        big = base64.b64decode(json.loads(raw)["data"][0]["embedding"])
        little = base64.b64decode(self.payload)
        self.assertNotEqual(big, little, 'endianness is ignored: "big" and "little" returned identical bytes')
        self.assertEqual(big, np.frombuffer(little, dtype="<f4").astype(">f4").tobytes(), 'the "big" payload is not a clean byte-swap of "little"')

    def test_the_client_decodes_the_live_payload_bit_equal(self):
        import numpy as np

        c = EmbedClient(_LIVE_URL, model=self.model)
        served, max_model_len = asyncio.run(c.models())
        self.assertTrue(served)
        self.assertGreater(max_model_len, 0)

        vectors = asyncio.run(c.embed([_LIVE_TEXT]))
        self.assertEqual(vectors.shape, (1, self.native_dimension))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(vectors[0].tobytes(), np.asarray(self.floats, dtype=np.float32).tobytes())

        # Two inputs in one request must come back as two distinct Chunk vectors in
        # input order -- vLLM merges them as they finish, so only ``index`` ties a
        # payload to its Chunk.
        pair = asyncio.run(c.embed([_LIVE_TEXT, "A completely unrelated sentence about tax law."]))
        self.assertEqual(pair.shape, (2, self.native_dimension))
        self.assertFalse(np.array_equal(pair[0], pair[1]))
        self.assertEqual(pair[0].tobytes(), vectors[0].tobytes())

    def test_tokenize_is_mounted_at_the_top_level(self):
        # The route the client calls, confirmed against a real build: vLLM mounts
        # tokenize outside the OpenAI-compatible prefix, so it is POST /tokenize and
        # never POST /v1/tokenize.
        status, raw = _live_post("/tokenize", {"model": self.model, "prompt": _LIVE_TEXT})
        self.assertEqual(status, 200, f"POST /tokenize returned {status}: {raw[:300]!r}")
        self.assertGreater(asyncio.run(EmbedClient(_LIVE_URL, model=self.model).tokenize(_LIVE_TEXT)), 0)


if __name__ == "__main__":
    unittest.main()
