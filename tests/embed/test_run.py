"""End-to-end tests for the `paperscale embed` orchestrator.

Everything runs against a fake vLLM injected at the client's two transport seams --
`paperscale.embed.client.apost` for the two POST routes and `_aget` for `/v1/models`.
The client, the chunker, the Adapters, both Sinks and the Resume intersection are the
real ones, so what is under test is the wiring: startup order, the split-on-failure
path, the counts by outcome and the exit code.

The fake counts one token per character, which makes the packing arithmetic something
a reader can do in their head: with `max_model_len` 4096 the Chunk budget is 4032
characters and the request budget is the flag's 32000, so the small Documents below
all land in one request together -- which is exactly the case split-on-failure exists
for.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from paperscale.cli import _parse_runs, build_parser
from paperscale.embed import client as client_mod
from paperscale.embed import run as run_mod
from paperscale.embed.npz_sink import FAILURES_NAME, MANIFEST_NAME
from tests.evaluation.fixtures import make_dolma_record, write_run

# The 0.6B Adapter: native width 1024, card context 32768, an empty document-side
# Instruction. Chosen over a stub so the tests exercise a registry key an operator can
# actually type, and so `--embed-dim`'s real floor (32) applies.
_MODEL = "qwen3-embedding-0.6b"
_NATIVE_DIM = 1024

# The marker a Document's text carries to make the fake server refuse the request it
# rides in. Nothing in the pipeline looks at it; only the fake does.
_POISON = "POISON"


class _FakeServer:
    """The three routes `run_embed` touches, with scripted failures.

    Token counts are character counts. Vectors are deterministic in the text so a
    reordered response would be detectable, and never zero-norm (which
    `slice_and_normalize` refuses by design).
    """

    def __init__(self, *, native_dim: int = _NATIVE_DIM, model_id: str = "fake/embedder", max_model_len: int = 4096) -> None:
        self.native_dim = native_dim
        self.model_id = model_id
        self.max_model_len = max_model_len
        self.embed_calls: list[list[str]] = []
        self.tokenize_calls: list[str] = []
        # Whether a request carrying the poison marker is refused, and with which
        # flavour of 400. `refuse` is switched off to prove a failed Document is
        # retried by the next Invocation rather than remembered as failed.
        self.refuse = True
        self.oversize = False

    async def aget(self, url, api_key=None):
        body = {"data": [{"id": self.model_id, "max_model_len": self.max_model_len}]}
        return 200, json.dumps(body).encode()

    async def apost(self, url, json_data, api_key=None):
        if url.endswith("/v1/tokenize"):
            self.tokenize_calls.append(json_data["prompt"])
            return 200, json.dumps({"count": len(json_data["prompt"])}).encode()
        if url.endswith("/v1/embeddings"):
            texts = list(json_data["input"])
            self.embed_calls.append(texts)
            if self.refuse and any(_POISON in text for text in texts):
                if self.oversize:
                    return 400, b'{"error": {"message": "This model\'s maximum context length is 4096 tokens"}}'
                return 400, b'{"error": {"message": "bad request"}}'
            data = [{"index": i, "embedding": _b64(self._vector(text))} for i, text in enumerate(texts)]
            return 200, json.dumps({"data": data}).encode()
        raise AssertionError(f"unexpected route {url}")

    def _vector(self, text: str):
        import numpy as np

        vector = np.arange(1, self.native_dim + 1, dtype="<f4")
        # A per-text scale keeps every row distinct without ever reaching zero norm.
        return vector * float(1 + (len(text) % 7))


def _b64(values) -> str:
    import numpy as np

    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode()


@contextlib.contextmanager
def _serving(server: _FakeServer):
    with mock.patch.object(client_mod, "apost", new=server.apost), mock.patch.object(client_mod, "_aget", new=server.aget):
        yield server


def _args(*argv: str):
    return build_parser().parse_args(["embed", *argv])


def _run(*argv: str) -> tuple[int, str]:
    """Invoke the handler and return `(exit code, stdout)`.

    stdout carries the end-of-run report, which is the only place several of the
    counts by outcome are visible at all.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_mod.run_embed(_args(*argv))
    return code, buffer.getvalue()


def _corpus(root: Path, names: list[str], *, poison: Sequence[str] = (), empty: Sequence[str] = ()) -> Path:
    records = []
    for name in names:
        if name in empty:
            pages = [""]
        elif name in poison:
            pages = [f"page one of {name} {_POISON}", "page two"]
        else:
            pages = [f"page one of {name}", "page two"]
        # A relative Source-File, so the derived Document name is the bare filename
        # and the expected output path is readable in every assertion below.
        records.append(make_dolma_record(name, pages))
    return write_run(root, records)


class RunLabelValidationTest(unittest.TestCase):
    """Design 14.4 / obligation 36 -- the check is embed's, and only embed's."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        _corpus(self.root / "run", ["a.pdf"])

    def test_a_slash_in_a_label_is_rejected_and_names_the_character(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            _run("--run", f"legal/2024={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", _MODEL)
        message = str(caught.exception)
        self.assertIn("'/'", message)
        self.assertIn("legal/2024", message)

    def test_the_message_names_the_first_offending_character_not_the_last(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            _run("--run", f"a b+c={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", _MODEL)
        self.assertIn("' '", str(caught.exception))

    def test_evaluate_still_accepts_the_label_embed_rejects(self) -> None:
        # The whole reason the check is not in `_parse_runs`: that function is shared,
        # and `evaluate` puts the label in a SQLite column where a slash is harmless.
        # If this ever fails, the check has been tightened globally and an existing
        # subcommand's contract has been broken to serve a new one.
        self.assertEqual(_parse_runs(["legal/2024=/tmp/x"]), [("legal/2024", "/tmp/x")])
        parsed = build_parser().parse_args(["evaluate", "--run", "legal/2024=/tmp/x"])
        self.assertEqual(parsed.run, ["legal/2024=/tmp/x"])

    def test_a_plain_label_passes(self) -> None:
        run_mod._validate_run_labels([("legal-2024.v2_a", "/tmp/x")])


def _embed_actions():
    """The `embed` subparser's argparse actions, for asserting on flag metadata."""
    subparsers = next(a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction))
    return subparsers.choices["embed"]._actions


class RequiredModelTest(unittest.TestCase):
    """Obligation 35's CLI half -- omitting `--embed-model` is an error, not a default.

    `test_adapters.py` pins the other half (the registry carries no
    `DEFAULT_EMBED_MODEL`). Both are needed: a `default=` added to the argparse flag
    would satisfy the registry test while silently reintroducing the default, and a
    default here is the failure the design refuses -- vectors from two models are not
    comparable, and nothing downstream could tell which one wrote a Sink.
    """

    def test_omitting_the_model_is_a_parse_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            build_parser().parse_args(["embed", "--run", "r=/tmp/x"])
        # argparse's own "required argument" exit, not a handler-level check.
        self.assertEqual(caught.exception.code, 2)

    def test_the_flag_carries_no_default(self) -> None:
        action = next(a for a in _embed_actions() if "--embed-model" in a.option_strings)
        self.assertTrue(action.required)
        self.assertIsNone(action.default)


class SinkSelectionTest(unittest.TestCase):
    """Obligation 37 -- at least one Sink must be live (design 14.5)."""

    def test_no_npz_without_lancedb_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf"])
            with self.assertRaises(SystemExit) as caught:
                _run("--run", f"r={root / 'run'}", "--out", str(root / "out"), "--embed-model", _MODEL, "--no-npz")
            self.assertIn("--lancedb", str(caught.exception))

    def test_no_npz_with_lancedb_is_accepted(self) -> None:
        # Guards the inverse: the rejection above must not be a blanket ban on --no-npz.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf"])
            with _serving(_FakeServer()):
                code, _out = _run(
                    "--run",
                    f"r={root / 'run'}",
                    "--out",
                    str(root / "out"),
                    "--embed-model",
                    _MODEL,
                    "--no-npz",
                    "--lancedb",
                    str(root / "db"),
                )
            self.assertEqual(code, 0)
            # The manifest is written even with the file Sink off (design 17.2 item 1):
            # it is the only record of which Sinks built this output.
            self.assertTrue((root / "out" / MANIFEST_NAME).exists())
            self.assertEqual(list((root / "out").glob("*.npz")), [])


class EndToEndTest(unittest.TestCase):
    """A clean Invocation: counts by outcome, the outputs, and exit 0."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        _corpus(self.root / "run", ["a.pdf", "b.pdf", "blank.pdf"], empty=["blank.pdf"])

    def _embed(self, *extra: str):
        with _serving(_FakeServer()) as server:
            code, report = _run("--run", f"r={self.root / 'run'}", "--out", str(self.out), "--embed-model", _MODEL, *extra)
        return code, report, server

    def test_counts_by_outcome_and_a_zero_exit(self) -> None:
        code, report, _server = self._embed()
        self.assertEqual(code, 0)
        self.assertIn("2 embedded", report)
        self.assertIn("1 empty", report)
        self.assertIn("0 skipped", report)
        self.assertIn("0 failed", report)

    def test_every_document_gets_a_sidecar_and_an_npz(self) -> None:
        self._embed()
        for name in ("a.pdf", "b.pdf", "blank.pdf"):
            self.assertTrue((self.out / f"{name}.npz").exists(), name)
            self.assertTrue((self.out / f"{name}.json").exists(), name)

    def test_the_stored_width_is_the_flag_not_the_native_width(self) -> None:
        import numpy as np

        self._embed("--embed-dim", "64")
        with np.load(self.out / "a.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(z["chunk_vectors"].shape[1], 64)
            self.assertEqual(z["document_vector"].shape, (64,))

    def test_no_failures_file_is_left_behind_when_nothing_failed(self) -> None:
        self._embed()
        self.assertFalse((self.out / FAILURES_NAME).exists())

    def test_a_second_invocation_skips_everything_and_embeds_nothing(self) -> None:
        self._embed()
        code, report, server = self._embed()
        self.assertEqual(code, 0)
        self.assertIn("3 skipped", report)
        self.assertIn("0 embedded", report)
        # The point of the derived Resume state: not one request is sent for a corpus
        # the Sink already holds. Only the startup width probe reaches the server.
        self.assertEqual(server.embed_calls, [["paperscale"]])

    def test_no_resume_re_embeds_and_deletes_nothing(self) -> None:
        self._embed()
        marker = self.out / "a.pdf.npz"
        before = marker.stat().st_mtime_ns
        code, report, _server = self._embed("--no-resume")
        self.assertEqual(code, 0)
        self.assertIn("0 skipped", report)
        self.assertIn("2 embedded", report)
        # Overwritten in place, never removed and re-created behind a gap in which a
        # Consumer would see the file missing.
        self.assertTrue(marker.exists())
        self.assertGreaterEqual(marker.stat().st_mtime_ns, before)


class SplitOnFailureTest(unittest.TestCase):
    """Obligations 45 and 46 -- one bad Document must not take its request-mates down."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        _corpus(self.root / "run", ["good1.pdf", "bad.pdf", "good2.pdf"], poison=["bad.pdf"])

    def _embed(self, *, oversize: bool = False):
        server = _FakeServer()
        server.oversize = oversize
        with _serving(server):
            code, report = _run(
                "--run",
                f"r={self.root / 'run'}",
                "--out",
                str(self.out),
                "--embed-model",
                _MODEL,
                # One worker at a time keeps the three Documents in one request, which
                # is the configuration the split exists for.
                "--concurrency",
                "1",
            )
        return code, report, server

    def test_only_the_poison_document_is_recorded_failed(self) -> None:
        code, report, _server = self._embed()
        self.assertEqual(code, 1)
        self.assertIn("2 embedded", report)
        self.assertIn("1 failed", report)

    def test_the_request_is_re_issued_one_document_at_a_time(self) -> None:
        _code, _report, server = self._embed()
        # The probe, then the mixed request, then one request per Document in it.
        multi = [call for call in server.embed_calls if len(call) > 1]
        self.assertEqual(len(multi), 1, server.embed_calls)
        singles = server.embed_calls[server.embed_calls.index(multi[0]) + 1 :]
        self.assertTrue(all(len(call) == 1 for call in singles), singles)
        self.assertEqual(len(singles), 3)

    def test_the_survivors_are_written_and_the_failure_is_not(self) -> None:
        self._embed()
        self.assertTrue((self.out / "good1.pdf.npz").exists())
        self.assertTrue((self.out / "good2.pdf.npz").exists())
        self.assertFalse((self.out / "bad.pdf.npz").exists())
        # Nothing partial either: the sidecar is written first, so a half-written
        # Document would show up as a `.json` with no `.npz`.
        self.assertFalse((self.out / "bad.pdf.json").exists())

    def test_the_failures_file_names_the_failed_document_and_only_it(self) -> None:
        self._embed()
        body = (self.out / FAILURES_NAME).read_text(encoding="utf-8")
        self.assertEqual(body.split(), ["bad.pdf"])

    def test_a_failed_document_is_retried_by_the_next_invocation(self) -> None:
        # The failures file is a convenience, not state: Resume derives from the
        # outputs, and a failed Document simply has no output.
        self._embed()
        server = _FakeServer()
        server.refuse = False
        with _serving(server):
            code, report = _run("--run", f"r={self.root / 'run'}", "--out", str(self.out), "--embed-model", _MODEL)
        self.assertEqual(code, 0)
        self.assertIn("2 skipped", report)
        self.assertIn("1 embedded", report)
        self.assertFalse((self.out / FAILURES_NAME).exists())

    def test_a_context_overflow_is_counted_apart_and_named_a_bug(self) -> None:
        code, report, _server = self._embed(oversize=True)
        self.assertEqual(code, 1)
        self.assertIn("1 failed", report)
        self.assertIn("context overflow", report)
        self.assertIn("chunker or context-length bug", report)


class ProbeTest(unittest.TestCase):
    """Obligation 5 -- a server of the wrong width stops the Invocation, naming both."""

    def test_a_narrower_server_stops_before_anything_is_written(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf"])
            with _serving(_FakeServer(native_dim=512)):
                with self.assertRaises(SystemExit) as caught:
                    _run("--run", f"r={root / 'run'}", "--out", str(root / "out"), "--embed-model", _MODEL)
            message = str(caught.exception)
            self.assertIn("512", message)
            self.assertIn(str(_NATIVE_DIM), message)
            # Step 8 runs before step 9, so not even the manifest exists yet.
            self.assertFalse((root / "out" / MANIFEST_NAME).exists())


class ServerGoneTest(unittest.TestCase):
    """Design 17.1 -- a dead server ends the Invocation instead of failing the corpus."""

    def test_the_invocation_stops_rather_than_marking_every_document_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", [f"doc{i}.pdf" for i in range(5)])
            server = _FakeServer()
            errors = io.StringIO()
            with _serving(server), _NoSleep(), contextlib.redirect_stderr(errors):
                # Armed after the probe, so startup succeeds and the failure lands
                # where the design cares about it: in the middle of the corpus.
                original = server.apost

                async def die(url, json_data, api_key=None):
                    if url.endswith("/v1/embeddings") and server.embed_calls:
                        raise ConnectionRefusedError("server went away")
                    return await original(url, json_data, api_key)

                with mock.patch.object(client_mod, "apost", new=die):
                    code, report = _run("--run", f"r={root / 'run'}", "--out", str(root / "out"), "--embed-model", _MODEL, "--concurrency", "1")
            self.assertEqual(code, 1)
            self.assertIn("cannot reach", errors.getvalue())
            # Every Document is left un-embedded rather than recorded failed, so the
            # next Invocation retries them all.
            self.assertIn("0 embedded", report)
            self.assertIn("0 failed", report)
            self.assertFalse((root / "out" / FAILURES_NAME).exists())


class TokenizeFailureTest(unittest.TestCase):
    """Design 12.6 -- a `/v1/tokenize` failure fails the Document, not the Invocation."""

    def test_one_document_fails_and_the_rest_are_embedded(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["good.pdf", "bad.pdf"], poison=["bad.pdf"])
            server = _FakeServer()
            original = server.apost

            async def refuse_poison_tokenize(url, json_data, api_key=None):
                if url.endswith("/v1/tokenize") and _POISON in json_data["prompt"]:
                    return 500, b"tokenizer exploded"
                return await original(url, json_data, api_key)

            with _serving(server), _NoSleep(), mock.patch.object(client_mod, "apost", new=refuse_poison_tokenize):
                code, report = _run("--run", f"r={root / 'run'}", "--out", str(root / "out"), "--embed-model", _MODEL)
            self.assertEqual(code, 1)
            self.assertIn("1 embedded", report)
            self.assertIn("1 failed", report)
            self.assertEqual((root / "out" / FAILURES_NAME).read_text(encoding="utf-8").split(), ["bad.pdf"])


class StartupOrderTest(unittest.TestCase):
    """The three stops that must happen before any GPU work (design 12.1)."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_name_collision_is_fatal_before_the_server_is_touched(self) -> None:
        # Two Source-Files that differ only by a leading slash derive one name.
        records = [make_dolma_record("/a/c.pdf", ["one"]), make_dolma_record("a/c.pdf", ["two"])]
        write_run(self.root / "run", records)
        server = _FakeServer()
        with _serving(server):
            with self.assertRaises(SystemExit) as caught:
                _run("--run", f"r={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", _MODEL)
        self.assertIn("a/c.pdf", str(caught.exception))
        self.assertEqual(server.embed_calls, [])

    def test_an_unknown_embed_model_is_a_message_and_not_a_traceback(self) -> None:
        _corpus(self.root / "run", ["a.pdf"])
        with self.assertRaises(SystemExit) as caught:
            _run("--run", f"r={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", "qwen4")
        self.assertIn("qwen4", str(caught.exception))

    def test_an_embed_dim_above_the_native_width_is_rejected_not_clamped(self) -> None:
        _corpus(self.root / "run", ["a.pdf"])
        with self.assertRaises(SystemExit) as caught:
            _run("--run", f"r={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", _MODEL, "--embed-dim", "2048")
        self.assertIn("2048", str(caught.exception))

    def test_the_reporter_title_carries_the_served_id_not_the_flag(self) -> None:
        # Design 13.3: the header is the only place the model id fits, and it must be
        # what the server answered -- the width probe cannot tell two models apart.
        _corpus(self.root / "run", ["a.pdf"])
        titles: list[str] = []
        # `run.py` imports `make_reporter` inside the handler, so the attribute is
        # looked up on `paperscale.tui` at call time and patching there is enough.
        with _serving(_FakeServer(model_id="Qwen/Qwen3-Embedding-0.6B")):
            with mock.patch("paperscale.tui.make_reporter", side_effect=_recording_reporter(titles)):
                _run("--run", f"r={self.root / 'run'}", "--out", str(self.root / "out"), "--embed-model", _MODEL)
        self.assertEqual(len(titles), 1)
        self.assertIn("Qwen/Qwen3-Embedding-0.6B", titles[0])
        self.assertNotIn(_MODEL, titles[0])


class ManifestGuardTest(unittest.TestCase):
    """Step 9: an Invocation whose settings differ stops before writing (obligation 21)."""

    def test_a_changed_embed_dim_stops_the_second_invocation(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf"])
            argv = ("--run", f"r={root / 'run'}", "--out", str(root / "out"), "--embed-model", _MODEL)
            with _serving(_FakeServer()):
                self.assertEqual(_run(*argv)[0], 0)
            with _serving(_FakeServer()):
                with self.assertRaises(SystemExit) as caught:
                    _run(*argv, "--embed-dim", "512")
            message = str(caught.exception)
            self.assertIn("stored_dim", message)
            self.assertIn("768", message)
            self.assertIn("512", message)

    def test_a_layout_change_reports_both_values(self) -> None:
        # Obligation 33. One --run writes `bare`; adding a second would write
        # `labelled`, under which every existing Document matches nothing.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf"])
            _corpus(root / "run2", ["b.pdf"])
            out = root / "out"
            with _serving(_FakeServer()):
                self.assertEqual(_run("--run", f"r={root / 'run'}", "--out", str(out), "--embed-model", _MODEL)[0], 0)
            with _serving(_FakeServer()):
                with self.assertRaises(SystemExit) as caught:
                    _run("--run", f"r={root / 'run'}", "--run", f"s={root / 'run2'}", "--out", str(out), "--embed-model", _MODEL)
            message = str(caught.exception)
            self.assertIn("bare", message)
            self.assertIn("labelled", message)


class TwoRunsTest(unittest.TestCase):
    """Two Runs give the `labelled` layout, and one PDF in both is two Documents."""

    def test_the_same_source_file_in_two_runs_is_not_a_collision(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "one", ["shared.pdf"])
            _corpus(root / "two", ["shared.pdf"])
            out = root / "out"
            with _serving(_FakeServer()):
                code, report = _run(
                    "--run",
                    f"a={root / 'one'}",
                    "--run",
                    f"b={root / 'two'}",
                    "--out",
                    str(out),
                    "--embed-model",
                    _MODEL,
                )
            self.assertEqual(code, 0)
            self.assertIn("2 embedded", report)
            self.assertTrue((out / "a" / "shared.pdf.npz").exists())
            self.assertTrue((out / "b" / "shared.pdf.npz").exists())


class LanceSinkWiringTest(unittest.TestCase):
    """Both Sinks live: the intersection, and `is_new` coming from the table itself."""

    def test_both_sinks_are_written_and_the_second_run_skips(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf", "b.pdf"])
            argv = (
                "--run",
                f"r={root / 'run'}",
                "--out",
                str(root / "out"),
                "--embed-model",
                _MODEL,
                "--lancedb",
                str(root / "db"),
            )
            with _serving(_FakeServer()):
                code, report = _run(*argv)
            self.assertEqual(code, 0)
            self.assertIn("2 embedded", report)
            self.assertEqual(_document_rows(root / "db"), 2)

            with _serving(_FakeServer()):
                code, report = _run(*argv)
            self.assertEqual(code, 0)
            self.assertIn("2 skipped", report)
            self.assertEqual(_document_rows(root / "db"), 2)

    def test_no_resume_replaces_rows_instead_of_duplicating_them(self) -> None:
        # The hole `is_new` closes: under --no-resume the Resume set is empty, so a
        # Document taken as "new" would be `add()`ed on top of rows already there.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _corpus(root / "run", ["a.pdf", "b.pdf"])
            argv = (
                "--run",
                f"r={root / 'run'}",
                "--out",
                str(root / "out"),
                "--embed-model",
                _MODEL,
                "--lancedb",
                str(root / "db"),
            )
            with _serving(_FakeServer()):
                _run(*argv)
            with _serving(_FakeServer()):
                code, _report = _run(*argv, "--no-resume")
            self.assertEqual(code, 0)
            self.assertEqual(_document_rows(root / "db"), 2)


class MultiChunkTest(unittest.TestCase):
    """A Document too big for one Chunk, and too big for one request.

    The Chunk budget here is 4032 "tokens" (characters), so each 3,000-character page
    is its own Chunk; twenty-seven of them is 81,000 tokens against a 32,000-token
    request budget, which forces the Document across several requests. It must still
    arrive as one output with every Chunk in it, in order.
    """

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        pages = [f"{i:04d} " + ("x" * 2995) for i in range(27)]
        write_run(self.root / "run", [make_dolma_record("big.pdf", pages)])
        self.text = "\n".join(pages)

    def test_the_document_lands_once_with_every_chunk_and_the_chunks_tile_it(self) -> None:
        import numpy as np

        with _serving(_FakeServer()) as server:
            code, report = _run("--run", f"r={self.root / 'run'}", "--out", str(self.out), "--embed-model", _MODEL)
        self.assertEqual(code, 0)
        self.assertIn("1 embedded", report)
        with np.load(self.out / "big.pdf.npz", allow_pickle=False) as z:
            n = z["chunk_vectors"].shape[0]
            self.assertEqual(n, 27)
            self.assertEqual(int(z["start_char"][0]), 0)
            self.assertEqual(int(z["end_char"][-1]), len(self.text))
            for i in range(n - 1):
                self.assertEqual(int(z["end_char"][i]), int(z["start_char"][i + 1]))
        # The probe plus the Chunks, and no Chunk sent twice: the token budget is what
        # decided the split, so the count of texts on the wire is the count of Chunks.
        self.assertEqual(sum(len(call) for call in server.embed_calls), 1 + 27)
        self.assertGreater(len([call for call in server.embed_calls if len(call) > 1]), 1)


class _Phase:
    def __init__(self, recorder) -> None:
        self.recorder = recorder

    def advance(self, n: int = 1) -> None:
        self.recorder.advanced += n

    def done(self) -> None:
        self.recorder.finished = True


class _Recorder:
    """A reporter that records what the orchestrator pushes at it.

    The panel's rendering is `tests/embed/test_panel.py`'s; what belongs here is the
    other half -- what `run.py` actually decides to push, above all the bar's total.
    """

    def __init__(self) -> None:
        self.phases: list[tuple[str, int | None]] = []
        self.stats: dict[tuple[str, str], object] = {}
        self.logs: list[str] = []
        self.advanced = 0
        self.finished = False

    def __enter__(self) -> "_Recorder":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def phase(self, name: str, total: int | None = None) -> _Phase:
        self.phases.append((name, total))
        return _Phase(self)

    def log(self, message: str) -> None:
        self.logs.append(message)

    def set_stat(self, name: str, value, *, group: str = "run") -> None:
        self.stats[(group, name)] = value


class PanelWiringTest(unittest.TestCase):
    """Design 13.5 -- the bar counts only Documents that will actually be embedded."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        _corpus(self.root / "run", ["a.pdf", "b.pdf", "blank.pdf"], empty=["blank.pdf"])

    def _embed(self) -> _Recorder:
        recorder = _Recorder()
        with _serving(_FakeServer()), mock.patch("paperscale.tui.make_reporter", return_value=recorder):
            _run("--run", f"r={self.root / 'run'}", "--out", str(self.out), "--embed-model", _MODEL)
        return recorder

    def test_the_total_is_corpus_minus_skipped(self) -> None:
        recorder = self._embed()
        self.assertEqual(recorder.phases, [("embedding", 3)])
        self.assertEqual(recorder.stats[("run", "skipped")], 0)

    def test_a_fully_resumed_invocation_reads_zero_of_zero(self) -> None:
        self._embed()
        recorder = self._embed()
        # The symptom standing decision 7 leaves undetected, surfaced with no
        # threshold: a stale output directory visibly has nothing to do.
        self.assertEqual(recorder.phases, [("embedding", 0)])
        self.assertEqual(recorder.stats[("run", "skipped")], 3)
        self.assertEqual(recorder.advanced, 0)

    def test_the_split_is_stated_once_at_startup(self) -> None:
        self._embed()
        recorder = self._embed()
        self.assertEqual([line for line in recorder.logs if "already held" in line], ["embedding 0 of 3 Document(s); 3 already held by every enabled Sink."])

    def test_the_default_log_path_sits_beside_the_output_tree_not_inside_it(self) -> None:
        # A live reporter owns the screen, so --disk-logging gets a default. `<out>` is
        # the deliverable a Consumer opens; a `logs/` directory appearing inside it is
        # one more thing every reader has to learn to ignore.
        self._embed()
        self.assertTrue(list((self.root / "logs").glob("embed-*.log")))
        self.assertFalse((self.out / "logs").exists())

    def test_the_issue_rows_exist_from_the_first_frame(self) -> None:
        # A column that grows a new row mid-run reads as something breaking, so both
        # are seeded at zero before the first Document.
        recorder = self._embed()
        self.assertEqual(recorder.stats[("issues", "failed")], 0)
        self.assertEqual(recorder.stats[("issues", "oversize")], 0)
        self.assertEqual(recorder.stats[("run", "empty")], 1)
        self.assertEqual(recorder.stats[("run", "documents")], 2)
        self.assertEqual(recorder.advanced, 3)


def _document_rows(db_path: Path) -> int:
    import lancedb

    return lancedb.connect(str(db_path)).open_table("documents").count_rows()


def _recording_reporter(titles: list[str]):
    from paperscale.tui import make_reporter as real_make_reporter

    def record(tui, *, title, stream=None):
        titles.append(title)
        return real_make_reporter(tui, title=title, stream=stream)

    return record


class _NoSleep:
    """Swap `asyncio.sleep` for a recorder so a retry axis is not waited out.

    A faithful connection-axis run takes about four minutes of bounded backoff, which
    is the point of the axis and not something a test should pay for.
    """

    def __init__(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
