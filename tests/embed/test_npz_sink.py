"""Tests for the `.npz` Sink: the manifest, the sidecar, the eight arrays, the ordering.

Everything here runs against a real temporary directory rather than a filesystem fake.
Three of the properties under test -- atomic rename, "no `.npz` after a crash", and
`np.load(allow_pickle=False)` -- are properties of the filesystem and of numpy's own
reader, and a fake would only assert that this file's beliefs about them are internally
consistent.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from paperscale.embed.chunking import Chunk
from paperscale.embed.names import NameCollisionError
from paperscale.embed.invariants import Invariants, SinkInvariantError, compare_invariant_facts
from paperscale.embed.npz_sink import CHUNKER, FAILURES_NAME, MANIFEST_NAME, POOLING, NpzSink
from paperscale.embed.vectors import EmbeddedDocument

LOGGER = "paperscale.embed.npz_sink"

_STORED_DIM = 768
_NATIVE_DIM = 1024

# The eight arrays of design 8.4 -- and nothing else. `chunk_index` and `n_chunks` are
# absent on purpose: they are `arange(len(token_count))` and `len(token_count)`.
_EXPECTED_KEYS = sorted(["chunk_vectors", "document_vector", "start_char", "end_char", "first_page", "last_page", "token_count", "is_partial_page"])

_CREATED = datetime.datetime(2026, 8, 18, 9, 4, 37, tzinfo=datetime.timezone.utc)


def _invariants(**overrides) -> Invariants:
    base = dict(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        stored_dim=_STORED_DIM,
        native_dim=_NATIVE_DIM,
        document_instruction="",
        query_instruction="Instruct: {task_description}\nQuery:{query}",
        pooling=POOLING,
        chunker=CHUNKER,
        chunk_budget_tokens=32640,
        layout="bare",
    )
    base.update(overrides)
    return Invariants(**base)


def _unit_rows(n: int, dim: int = _STORED_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    return (raw / np.linalg.norm(raw, axis=1)[:, None]).astype(np.float32)


def _document(
    *,
    document_name: str = "law/case.pdf",
    run_label: str = "qwen",
    source_file: str = "/law/case.pdf",
    n_chunks: int = 3,
    stored_dim: int = _STORED_DIM,
) -> EmbeddedDocument:
    """A Document shaped the way `run.py` hands one over -- unit rows, parallel Chunks."""
    chunks = [
        Chunk(start_char=100 * i, end_char=100 * (i + 1), first_page=i, last_page=i + 1, token_count=40 + i, is_partial_page=bool(i % 2))
        for i in range(n_chunks)
    ]
    chunk_vectors = _unit_rows(n_chunks, stored_dim) if n_chunks else np.zeros((0, stored_dim), dtype=np.float32)
    document_vector = chunk_vectors[0].copy() if n_chunks else np.zeros((0,), dtype=np.float32)
    return EmbeddedDocument(
        document_name=document_name,
        run_label=run_label,
        source_file=source_file,
        source_digest="9f2b1c4d8e0a3f57",
        created=_CREATED,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        document_vector=document_vector,
    )


class _SinkTestCase(unittest.TestCase):
    """One temporary tree per test, with the Sink already opened."""

    layout = "bare"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # A path one level below the temporary root, so `open()` has to create it -- the
        # `<out>` an operator names does not exist on a first Invocation.
        self.out = Path(tmp.name) / "vectors"
        self.sink = NpzSink(self.out, _invariants(layout=self.layout), ["npz"])
        self.sink.open()

    def temp_files(self) -> list[Path]:
        return sorted(p for p in self.out.rglob("*") if p.name.endswith(".tmp"))


class RoundTripTest(_SinkTestCase):
    def test_eight_arrays_and_nothing_else(self):
        self.sink.write(_document())
        with np.load(self.out / "law" / "case.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(sorted(z.files), _EXPECTED_KEYS)

    def test_shapes_dtypes_and_values_survive_allow_pickle_false(self):
        doc = _document()
        self.sink.write(doc)
        # `allow_pickle=False` is the whole reason provenance lives outside the file: an
        # object array would make this call raise, and allow_pickle=True asks a Consumer to
        # execute whatever the file contains.
        with np.load(self.out / "law" / "case.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(z["chunk_vectors"].shape, (3, _STORED_DIM))
            self.assertEqual(z["chunk_vectors"].dtype, np.float32)
            self.assertEqual(z["document_vector"].shape, (_STORED_DIM,))
            self.assertEqual(z["document_vector"].dtype, np.float32)
            np.testing.assert_array_equal(z["chunk_vectors"], doc.chunk_vectors)
            np.testing.assert_array_equal(z["document_vector"], doc.document_vector)
            for name in ("start_char", "end_char", "first_page", "last_page", "token_count"):
                self.assertEqual(z[name].dtype, np.int32, msg=name)
                self.assertEqual(z[name].shape, (3,), msg=name)
            self.assertEqual(z["is_partial_page"].dtype, np.bool_)
            np.testing.assert_array_equal(z["start_char"], [0, 100, 200])
            np.testing.assert_array_equal(z["end_char"], [100, 200, 300])
            np.testing.assert_array_equal(z["first_page"], [0, 1, 2])
            np.testing.assert_array_equal(z["last_page"], [1, 2, 3])
            np.testing.assert_array_equal(z["token_count"], [40, 41, 42])
            np.testing.assert_array_equal(z["is_partial_page"], [False, True, False])

    def test_the_archive_is_stored_not_deflated(self):
        # savez, never savez_compressed: ~10% on near-random normalized floats, paid for
        # with CPU on every read of a format the Consumer reads many times.
        self.sink.write(_document())
        with zipfile.ZipFile(self.out / "law" / "case.pdf.npz") as archive:
            for info in archive.infolist():
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED, msg=info.filename)

    def test_the_source_extension_is_appended_not_replaced(self):
        # Issue #32: replacing the extension collapses `case.pdf` and `case.tiff` onto one
        # output and the second silently overwrites the first.
        self.sink.write(_document(document_name="law/case.pdf", source_file="/law/case.pdf"))
        self.sink.write(_document(document_name="law/case.tiff", source_file="/law/case.tiff"))
        self.assertTrue((self.out / "law" / "case.pdf.npz").exists())
        self.assertTrue((self.out / "law" / "case.tiff.npz").exists())

    def test_the_sidecar_carries_the_four_per_document_facts(self):
        self.sink.write(_document())
        payload = json.loads((self.out / "law" / "case.pdf.json").read_text())
        self.assertEqual(
            payload,
            {
                "source_file": "/law/case.pdf",
                "source_digest": "9f2b1c4d8e0a3f57",
                "run_label": "qwen",
                "created": "2026-08-18T09:04:37Z",
            },
        )

    def test_the_sidecar_records_the_raw_source_file_not_the_derived_name(self):
        # The name is lossy by construction (a leading slash and a tarball member both
        # collapse), so the raw string is the only route back to the PDF.
        self.sink.write(_document(document_name="law/case.pdf", source_file="/law/./case.pdf"))
        payload = json.loads((self.out / "law" / "case.pdf.json").read_text())
        self.assertEqual(payload["source_file"], "/law/./case.pdf")

    def test_a_naive_created_is_refused(self):
        doc = _document()
        naive = dataclasses.replace(doc, created=datetime.datetime(2026, 8, 18, 9, 4, 37))
        with self.assertRaises(ValueError) as caught:
            self.sink.write(naive)
        self.assertIn("timezone-aware", str(caught.exception))

    def test_rewriting_a_document_replaces_both_files(self):
        # `--no-resume` re-embeds in place, so the second write must land on the same two
        # names rather than accumulate a second pair.
        self.sink.write(_document())
        self.sink.write(_document(n_chunks=1))
        with np.load(self.out / "law" / "case.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(z["token_count"].shape, (1,))
        self.assertEqual(sorted(p.name for p in (self.out / "law").iterdir()), ["case.pdf.json", "case.pdf.npz"])

    def test_no_temporary_files_survive_a_normal_write(self):
        self.sink.write(_document())
        self.assertEqual(self.temp_files(), [])


class LabelledLayoutTest(_SinkTestCase):
    layout = "labelled"

    def test_the_run_label_is_a_directory(self):
        self.sink.write(_document(run_label="nemotron"))
        self.assertTrue((self.out / "nemotron" / "law" / "case.pdf.npz").exists())
        self.assertTrue((self.out / "nemotron" / "law" / "case.pdf.json").exists())

    def test_two_runs_holding_one_pdf_do_not_collide(self):
        self.sink.write(_document(run_label="qwen"))
        self.sink.write(_document(run_label="nemotron"))
        self.assertEqual(self.sink.known(), {("qwen", "law/case.pdf"), ("nemotron", "law/case.pdf")})

    def test_the_label_comes_from_the_path_and_no_sidecar_is_opened(self):
        # Design 11.1 budgets one walk over `<out>` and nothing else. Here the run label is
        # the first path component, so the walk answers the whole key on its own -- a read
        # per Document would cost more than the name manifest that section rejects.
        self.sink.write(_document(run_label="qwen"))
        self.sink.write(_document(document_name="other.pdf", run_label="nemotron"))
        with mock.patch.object(NpzSink, "_sidecar_run_label", side_effect=AssertionError("a sidecar was opened")) as opened:
            self.assertEqual(self.sink.known(), {("qwen", "law/case.pdf"), ("nemotron", "other.pdf")})
        opened.assert_not_called()

    def test_a_document_whose_sidecar_was_removed_is_re_embedded(self):
        # The pair is still checked -- against the names the walk already returned, which
        # costs no syscall -- so a broken pair is re-embedded rather than counted as done.
        self.sink.write(_document(run_label="qwen"))
        (self.out / "qwen" / "law" / "case.pdf.json").unlink()
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self.sink.known(), set())

    def test_a_stray_npz_at_the_root_names_no_document(self):
        # Every Document in this layout sits under its Run's directory, so a file at the
        # root has no label and names nothing.
        (self.out / "loose.npz").write_bytes(b"")
        (self.out / "loose.json").write_text("{}")
        self.assertEqual(self.sink.known(), set())


class EmptyDocumentTest(_SinkTestCase):
    """Design 11.4: a Document with no usable text is a recorded outcome, not a failure."""

    def test_every_array_is_length_zero_and_the_widths_survive(self):
        self.sink.write(_document(n_chunks=0))
        with np.load(self.out / "law" / "case.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(sorted(z.files), _EXPECTED_KEYS)
            self.assertEqual(z["chunk_vectors"].shape, (0, _STORED_DIM))
            self.assertEqual(z["chunk_vectors"].dtype, np.float32)
            self.assertEqual(z["document_vector"].shape, (0,))
            self.assertEqual(z["document_vector"].dtype, np.float32)
            for name in ("start_char", "end_char", "first_page", "last_page", "token_count"):
                self.assertEqual(z[name].shape, (0,), msg=name)
                self.assertEqual(z[name].dtype, np.int32, msg=name)
            self.assertEqual(z["is_partial_page"].shape, (0,))
            self.assertEqual(z["is_partial_page"].dtype, np.bool_)

    def test_one_line_distinguishes_it(self):
        self.sink.write(_document(document_name="empty.pdf", n_chunks=0))
        self.sink.write(_document(document_name="full.pdf", n_chunks=2))
        with np.load(self.out / "empty.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(z["document_vector"].size, 0)
        with np.load(self.out / "full.pdf.npz", allow_pickle=False) as z:
            self.assertNotEqual(z["document_vector"].size, 0)

    def test_no_zero_vector_is_written(self):
        # A zero vector was rejected: it is not a unit vector, nothing else in the store is
        # anything but a unit vector, and it would sit in a search index looking like data.
        self.sink.write(_document(n_chunks=0))
        with np.load(self.out / "law" / "case.pdf.npz", allow_pickle=False) as z:
            self.assertEqual(z["document_vector"].shape, (0,))

    def test_an_empty_document_is_known_so_it_is_not_retried_forever(self):
        self.sink.write(_document(n_chunks=0))
        self.assertEqual(self.sink.known(), {("qwen", "law/case.pdf")})

    def test_a_contradictory_document_is_refused(self):
        doc = _document(n_chunks=0)
        broken = dataclasses.replace(doc, document_vector=np.zeros((_STORED_DIM,), dtype=np.float32))
        with self.assertRaises(ValueError):
            self.sink.write(broken)

    def test_a_width_that_contradicts_the_manifest_is_refused(self):
        # `stored_dim` is a manifest fact about the whole tree, so a file of another width
        # breaks a promise made to every Consumer that reads the manifest.
        with self.assertRaises(ValueError) as caught:
            self.sink.write(_document(stored_dim=512))
        self.assertIn("512", str(caught.exception))
        self.assertIn(str(_STORED_DIM), str(caught.exception))


class _Crash(RuntimeError):
    """Stands in for the process dying between the sidecar and the `.npz`."""


class WriteOrderTest(_SinkTestCase):
    """Design 8.5: if the `.npz` exists, the sidecar exists.

    Resume uses the `.npz` alone as its completion marker, so the reverse order would leave
    Documents that Resume counts as done and that have no identity, forever --
    `unstructured-ingest` and ColBERT both ship exactly that bug.
    """

    def test_a_crash_before_the_npz_leaves_no_npz(self):
        with mock.patch("numpy.savez", side_effect=_Crash("died between the two writes")):
            with self.assertRaises(_Crash):
                self.sink.write(_document())
        self.assertTrue((self.out / "law" / "case.pdf.json").exists())
        self.assertFalse((self.out / "law" / "case.pdf.npz").exists())
        self.assertEqual(self.temp_files(), [])

    def test_a_crash_partway_through_the_npz_leaves_no_npz(self):
        def half_written(handle, **arrays):
            handle.write(b"PK\x03\x04 truncated")
            raise _Crash("died mid-write")

        with mock.patch("numpy.savez", side_effect=half_written):
            with self.assertRaises(_Crash):
                self.sink.write(_document())
        self.assertFalse((self.out / "law" / "case.pdf.npz").exists())
        self.assertEqual(self.temp_files(), [])

    def test_a_document_left_by_a_crash_is_not_known(self):
        with mock.patch("numpy.savez", side_effect=_Crash("died between the two writes")):
            with self.assertRaises(_Crash):
                self.sink.write(_document())
        self.assertEqual(self.sink.known(), set())

    def test_a_keyboard_interrupt_leaves_no_debris_either(self):
        # The most likely way an operator ends a twelve-hour Invocation.
        with mock.patch("numpy.savez", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.sink.write(_document())
        self.assertFalse((self.out / "law" / "case.pdf.npz").exists())
        self.assertEqual(self.temp_files(), [])


class ManifestTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "vectors"

    def manifest(self) -> dict:
        return json.loads((self.out / MANIFEST_NAME).read_text())

    def test_a_first_invocation_writes_the_nine_facts_and_one_entry(self):
        NpzSink(self.out, _invariants(), ["npz"]).open()
        manifest = self.manifest()
        self.assertEqual(
            [k for k in manifest if k not in ("sinks", "invocations")],
            ["model_id", "stored_dim", "native_dim", "document_instruction", "query_instruction", "pooling", "chunker", "chunk_budget_tokens", "layout"],
        )
        self.assertEqual(manifest["sinks"], ["npz"])
        self.assertEqual(len(manifest["invocations"]), 1)
        self.assertEqual(sorted(manifest["invocations"][0]), ["created", "paperscale_version"])

    def test_the_manifest_is_written_even_with_the_npz_sink_disabled(self):
        # Design 17.2 item 1: `<out>` exists under `--no-npz` anyway, because the failures
        # file lives there, and the invariant comparison is useful regardless.
        NpzSink(self.out, _invariants(), ["lancedb"]).open()
        self.assertEqual(self.manifest()["sinks"], ["lancedb"])

    def test_a_matching_second_invocation_appends_exactly_one_entry(self):
        NpzSink(self.out, _invariants(), ["npz"]).open()
        first = self.manifest()
        NpzSink(self.out, _invariants(), ["npz"]).open()
        second = self.manifest()
        self.assertEqual(len(second["invocations"]), 2)
        self.assertEqual(second["invocations"][0], first["invocations"][0])
        # Nothing else changes: overwriting the manifest was rejected outright, because it
        # would leave the file describing vectors it did not describe.
        self.assertEqual({k: v for k, v in second.items() if k != "invocations"}, {k: v for k, v in first.items() if k != "invocations"})

    def test_a_changed_invariant_stops_and_reports_both_values(self):
        NpzSink(self.out, _invariants(), ["npz"]).open()
        before = self.manifest()
        with self.assertRaises(SinkInvariantError) as caught:
            NpzSink(self.out, _invariants(model_id="nvidia/Nemotron-3-Embed-8B-BF16"), ["npz"]).open()
        message = str(caught.exception)
        self.assertIn("Qwen/Qwen3-Embedding-0.6B", message)
        self.assertIn("nvidia/Nemotron-3-Embed-8B-BF16", message)
        # It stops *before* anything is written, including its own log entry.
        self.assertEqual(self.manifest(), before)

    def test_a_same_width_different_model_is_caught_here_and_nowhere_else(self):
        # The check `native_dim` cannot make: both 8B models are 4096 wide, so the wrong
        # -model assertion agrees and Resume, with no content detection, skips every name.
        NpzSink(self.out, _invariants(model_id="Qwen/Qwen3-Embedding-8B", native_dim=4096), ["npz"]).open()
        with self.assertRaises(SinkInvariantError):
            NpzSink(self.out, _invariants(model_id="nvidia/Nemotron-3-Embed-8B-BF16", native_dim=4096), ["npz"]).open()

    def test_every_disagreeing_fact_is_listed(self):
        # One operator decision usually moves several facts at once; reporting the first
        # would turn it into several failed runs.
        NpzSink(self.out, _invariants(), ["npz"]).open()
        with self.assertRaises(SinkInvariantError) as caught:
            NpzSink(self.out, _invariants(stored_dim=512, chunk_budget_tokens=8000, document_instruction="passage: "), ["npz"]).open()
        message = str(caught.exception)
        for fragment in ("stored_dim", "chunk_budget_tokens", "document_instruction", "512", "8000", "passage: "):
            self.assertIn(fragment, message)

    def test_a_layout_change_stops_and_explains_the_run_set(self):
        NpzSink(self.out, _invariants(layout="bare"), ["npz"]).open()
        with self.assertRaises(SinkInvariantError) as caught:
            NpzSink(self.out, _invariants(layout="labelled"), ["npz"]).open()
        message = str(caught.exception)
        self.assertIn("'bare'", message)
        self.assertIn("'labelled'", message)
        self.assertIn("run set", message)

    def test_an_added_sink_is_recorded_for_the_next_invocation(self):
        # The *warning* about an added Sink belongs to `resume.sink_set_warning`, which
        # says it once from run.py with the corpus size in hand; test_resume.py owns those
        # assertions. What is this Sink's own is the record: `previous_sinks` for the
        # caller to compare, and the new set written back to the manifest.
        NpzSink(self.out, _invariants(), ["npz"]).open()
        sink = NpzSink(self.out, _invariants(), ["npz", "lancedb"])
        sink.open()
        self.assertEqual(sink.previous_sinks, ["npz"])
        self.assertEqual(self.manifest()["sinks"], ["npz", "lancedb"])

    def test_a_dropped_sink_is_recorded_the_same_way(self):
        # Recorded identically whichever direction the set moved. Only the caller judges
        # which direction is expensive, and it cannot judge without both lists.
        NpzSink(self.out, _invariants(), ["npz", "lancedb"]).open()
        sink = NpzSink(self.out, _invariants(), ["npz"])
        sink.open()
        self.assertEqual(sink.previous_sinks, ["npz", "lancedb"])
        self.assertEqual(self.manifest()["sinks"], ["npz"])

    def test_opening_a_tree_says_nothing_about_the_sink_set(self):
        # This Sink no longer speaks about the set at all, in any direction: one fact
        # reported twice in two wordings reads as two facts.
        NpzSink(self.out, _invariants(), ["npz"]).open()
        with self.assertNoLogs(LOGGER, level="INFO"):
            NpzSink(self.out, _invariants(), ["npz", "lancedb"]).open()

    def test_an_unreadable_manifest_stops_rather_than_being_replaced(self):
        self.out.mkdir(parents=True)
        (self.out / MANIFEST_NAME).write_text("{ this is not json")
        with self.assertRaises(SinkInvariantError) as caught:
            NpzSink(self.out, _invariants(), ["npz"]).open()
        self.assertIn(MANIFEST_NAME, str(caught.exception))
        self.assertEqual((self.out / MANIFEST_NAME).read_text(), "{ this is not json")

    def test_a_manifest_missing_a_fact_is_a_disagreement(self):
        NpzSink(self.out, _invariants(), ["npz"]).open()
        manifest = self.manifest()
        del manifest["query_instruction"]
        (self.out / MANIFEST_NAME).write_text(json.dumps(manifest))
        with self.assertRaises(SinkInvariantError) as caught:
            NpzSink(self.out, _invariants(), ["npz"]).open()
        self.assertIn("query_instruction", str(caught.exception))

    def test_no_document_is_written_before_the_comparison(self):
        NpzSink(self.out, _invariants(), ["npz"]).open()
        mismatched = NpzSink(self.out, _invariants(stored_dim=512), ["npz"])
        with self.assertRaises(SinkInvariantError):
            mismatched.open()
        self.assertEqual([p.name for p in self.out.rglob("*.npz")], [])


class ReservedNameTest(_SinkTestCase):
    """Design 7.5: a derived output landing on the reserved prefix is fatal, not an overwrite."""

    def test_a_reserved_document_name_is_refused(self):
        with self.assertRaises(NameCollisionError) as caught:
            self.sink.write(_document(document_name="paperscale-embed.pdf", source_file="/paperscale-embed.pdf"))
        self.assertIn("paperscale-embed", str(caught.exception))

    def test_the_manifest_survives_the_attempt(self):
        before = (self.out / MANIFEST_NAME).read_text()
        with self.assertRaises(NameCollisionError):
            self.sink.write(_document(document_name="paperscale-embed.pdf"))
        self.assertEqual((self.out / MANIFEST_NAME).read_text(), before)
        self.assertEqual(self.temp_files(), [])

    def test_the_failures_file_name_is_covered_by_the_same_prefix(self):
        with self.assertRaises(NameCollisionError):
            self.sink.write(_document(document_name="paperscale-embed-failures.pdf"))

    def test_only_the_first_component_is_reserved(self):
        # The manifest and the failures file sit at `<out>/` in both layouts, so nothing
        # below the root can reach them.
        self.sink.write(_document(document_name="law/paperscale-embed.pdf"))
        self.assertTrue((self.out / "law" / "paperscale-embed.pdf.npz").exists())


class SharedComparisonTest(unittest.TestCase):
    """`compare_invariant_facts` -- the one comparison both Sinks stop through.

    The `.npz` manifest and the LanceDB table metadata disagree in the same shape and report
    it in the same shape; only the artifact they name and the fix they offer differ.
    """

    def _compare(self, recorded, current, **overrides):
        options = {
            "subject": "/out/paperscale-embed.json",
            "holder": "this tree",
            "nothing_yet": "nothing has been embedded.",
            "remedy": "Re-run with the settings that built this tree.",
        }
        options.update(overrides)
        compare_invariant_facts(recorded, current, **options)

    def test_a_matching_output_is_silent(self):
        self.assertIsNone(self._compare({"model_id": "a", "stored_dim": 768}, {"model_id": "a", "stored_dim": 768}))

    def test_every_disagreement_is_listed_with_both_values(self):
        with self.assertRaises(SinkInvariantError) as caught:
            self._compare({"model_id": "a", "stored_dim": 768}, {"model_id": "b", "stored_dim": 512})
        message = str(caught.exception)
        self.assertIn("2 invariant fact(s) disagree", message)
        self.assertIn("nothing has been embedded.", message)
        for fragment in ("model_id", "'a'", "'b'", "stored_dim", "768", "512", "this tree has"):
            self.assertIn(fragment, message)

    def test_a_fact_the_output_does_not_record_is_a_disagreement(self):
        # Otherwise an output with no facts at all would read as agreeing with everything.
        with self.assertRaises(SinkInvariantError) as caught:
            self._compare({}, {"model_id": "b"})
        self.assertIn("<absent>", str(caught.exception))

    def test_a_note_is_added_only_for_the_fact_that_disagreed(self):
        notes = {"layout": "      pass the same run set", "model_id": "      unreachable note"}
        with self.assertRaises(SinkInvariantError) as caught:
            self._compare({"model_id": "a", "layout": "bare"}, {"model_id": "a", "layout": "labelled"}, notes=notes)
        message = str(caught.exception)
        self.assertIn("pass the same run set", message)
        self.assertNotIn("unreachable note", message)


class KnownTest(_SinkTestCase):
    def test_it_reports_the_run_label_and_the_document_name(self):
        self.sink.write(_document(document_name="law/case.pdf", run_label="qwen"))
        self.sink.write(_document(document_name="deep/er/other.pdf", run_label="qwen"))
        self.assertEqual(self.sink.known(), {("qwen", "law/case.pdf"), ("qwen", "deep/er/other.pdf")})

    def test_a_bare_tree_can_hold_two_labels(self):
        # A `bare` tree accumulates single-Run Invocations, and the layout carries no
        # label, which is why the label is read from the sidecar rather than the path --
        # the one `open()` per Document design 11.1 does not budget for, kept because
        # assuming the current label would reduce the key to the Document name alone.
        self.sink.write(_document(document_name="a.pdf", run_label="qwen"))
        self.sink.write(_document(document_name="b.pdf", run_label="nemotron"))
        self.assertEqual(self.sink.known(), {("qwen", "a.pdf"), ("nemotron", "b.pdf")})

    def test_a_missing_output_directory_is_not_an_error(self):
        sink = NpzSink(self.out / "not-created-yet", _invariants(), ["npz"])
        self.assertEqual(sink.known(), set())

    def test_a_document_whose_sidecar_was_removed_is_re_embedded(self):
        self.sink.write(_document())
        (self.out / "law" / "case.pdf.json").unlink()
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self.sink.known(), set())

    def test_the_manifest_and_the_failures_file_are_not_documents(self):
        self.sink.write_failures(["law/case.pdf"])
        self.assertEqual(self.sink.known(), set())

    def test_a_half_written_document_is_not_counted(self):
        # The temporary must not end in `.npz`, or the walk would count a partial file as a
        # finished Document -- the exact bug the temp-plus-rename dance exists to prevent.
        self.sink.write(_document())
        (self.out / "law" / "case.pdf.npz.abc123.tmp").write_bytes(b"partial")
        self.assertEqual(self.sink.known(), {("qwen", "law/case.pdf")})


class FailuresFileTest(_SinkTestCase):
    def test_one_document_name_per_line(self):
        self.sink.write_failures(["law/case.pdf", "law/other.pdf"])
        self.assertEqual((self.out / FAILURES_NAME).read_text(), "law/case.pdf\nlaw/other.pdf\n")

    def test_it_is_rewritten_not_appended(self):
        self.sink.write_failures(["first.pdf"])
        self.sink.write_failures(["second.pdf"])
        self.assertEqual((self.out / FAILURES_NAME).read_text(), "second.pdf\n")

    def test_an_invocation_with_no_failures_removes_a_stale_file(self):
        # A file left from the previous Invocation would name Documents this one embedded
        # successfully, and an operator reading it has no way to tell.
        self.sink.write_failures(["first.pdf"])
        self.sink.write_failures([])
        self.assertFalse((self.out / FAILURES_NAME).exists())

    def test_no_failures_and_no_file_is_not_an_error(self):
        self.sink.write_failures([])
        self.assertFalse((self.out / FAILURES_NAME).exists())


if __name__ == "__main__":
    unittest.main()
