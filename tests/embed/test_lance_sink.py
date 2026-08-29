"""Tests for the LanceDB Sink.

These run against the real lancedb (0.37.1) rather than a fake, because every claim the Sink
rests on is a claim about that library: that a `fixed_size_list` refuses a wrong-width vector,
that schema metadata survives reopen and `add()`, that a NULL vector is skipped by search, that
a plain upsert leaves a phantom Chunk behind, and that an unescaped `'` in a predicate is a
tokenizer error. A fake would only re-assert what the author believed.

Every database is a fresh temporary directory, so nothing here depends on execution order.
"""

from __future__ import annotations

import datetime
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa

from paperscale.embed.chunking import Chunk
from paperscale.embed.lance_sink import CHUNKS_TABLE, DOCUMENTS_TABLE, METADATA_FACTS, LanceSink, _sql_literal, table_metadata
from paperscale.embed.invariants import Invariants, SinkInvariantError
from paperscale.embed.vectors import EmbeddedDocument

_STORED_DIM = 8
_NATIVE_DIM = 1024
_CREATED = datetime.datetime(2026, 8, 20, 9, 30, tzinfo=datetime.timezone.utc)


def _invariants(**overrides) -> Invariants:
    facts = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "stored_dim": _STORED_DIM,
        "native_dim": _NATIVE_DIM,
        "document_instruction": "",
        "query_instruction": "Instruct: {task_description}\nQuery:{query}",
        "pooling": "token_weighted_mean",
        "chunker": "greedy_page_pack",
        "chunk_budget_tokens": 32704,
        "layout": "bare",
    }
    facts.update(overrides)
    return Invariants(**facts)


def _unit(seed: int, dim: int = _STORED_DIM) -> np.ndarray:
    """One deterministic unit-length float32 row -- the shape `vectors.py` hands the Sink."""
    raw = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


def _document(name: str, *, run_label: str = "legal", n_chunks: int = 2, dim: int = _STORED_DIM, seed: int = 0) -> EmbeddedDocument:
    chunks = [
        Chunk(start_char=i * 100, end_char=(i + 1) * 100, first_page=i + 1, last_page=i + 1, token_count=25 + i, is_partial_page=bool(i % 2))
        for i in range(n_chunks)
    ]
    vectors = np.array([_unit(seed + i, dim) for i in range(n_chunks)], dtype=np.float32).reshape(n_chunks, dim)
    pooled = _unit(seed + 900, dim) if n_chunks else np.zeros(0, dtype=np.float32)
    return EmbeddedDocument(
        document_name=name,
        run_label=run_label,
        source_file=f"/corpus/{name}",
        source_digest=f"{seed:016x}",
        created=_CREATED,
        chunks=chunks,
        chunk_vectors=vectors,
        document_vector=pooled,
    )


def _connect(path: Path):
    import lancedb

    return lancedb.connect(str(path))


def _rows(path: Path, table: str) -> list[dict]:
    return _connect(path).open_table(table).to_arrow().to_pylist()


def _keys(path: Path, table: str) -> list[tuple]:
    """`(document_name, chunk_index)` for `chunks`, `(document_name, n_chunks)` for `documents`."""
    second = "chunk_index" if table == CHUNKS_TABLE else "n_chunks"
    return sorted((row["document_name"], row[second]) for row in _rows(path, table))


class _RecordingTable:
    """Records which write API the Sink reached for, and delegates everything else untouched."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[str] = []

    def add(self, *args, **kwargs):
        self.calls.append("add")
        return self.inner.add(*args, **kwargs)

    def merge_insert(self, *args, **kwargs):
        self.calls.append("merge_insert")
        return self.inner.merge_insert(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _SinkTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "vectors.lancedb"

    def tearDown(self):
        self._tmp.cleanup()

    def sink(self, *, batch_size: int = 64, **overrides) -> LanceSink:
        sink = LanceSink(self.path, _invariants(**overrides), batch_size=batch_size)
        sink.open()
        return sink

    def write_all(self, sink: LanceSink, docs: list[EmbeddedDocument], *, is_new: bool = True) -> None:
        for doc in docs:
            sink.write(doc, is_new=is_new)
        sink.close()

    def record(self, sink: LanceSink) -> tuple[_RecordingTable, _RecordingTable]:
        """Swap both table handles for recorders, so a test can see which write API the Sink reached for.

        Which of `add` and `merge_insert` runs is not visible in the resulting rows -- both end with the
        Document stored -- so the only way to hold the design's amended write path (#40) is to watch the call.
        """
        documents, chunks = _RecordingTable(sink._documents), _RecordingTable(sink._chunks)
        sink._documents, sink._chunks = documents, chunks  # type: ignore[bad-assignment]
        return documents, chunks


class TableLayoutTest(_SinkTestCase):
    def test_two_tables_are_created_and_both_name_the_vector_column_vector(self):
        # One table with an is_doc discriminator would return a mixture of Chunk and Document
        # vectors on every search; the column name is what lets .search() take no argument.
        self.sink().close()
        db = _connect(self.path)
        self.assertEqual(sorted(db.list_tables().tables), [CHUNKS_TABLE, DOCUMENTS_TABLE])
        for name in (DOCUMENTS_TABLE, CHUNKS_TABLE):
            field = db.open_table(name).schema.field("vector")
            self.assertEqual(field.type, pa.list_(pa.float32(), _STORED_DIM), msg=name)

    def test_chunks_does_not_repeat_the_per_document_provenance(self):
        self.sink().close()
        columns = set(_connect(self.path).open_table(CHUNKS_TABLE).schema.names)
        self.assertNotIn("source_file", columns)
        self.assertNotIn("source_digest", columns)
        self.assertNotIn("created", columns)
        # token_count does live here, so a Consumer reading `chunks` alone can rebuild the
        # Document vector without joining `documents`.
        self.assertIn("token_count", columns)

    def test_metadata_carries_the_eight_invariant_facts_and_never_layout(self):
        self.sink().close()
        for name in (DOCUMENTS_TABLE, CHUNKS_TABLE):
            recorded = {key.decode(): value.decode() for key, value in _connect(self.path).open_table(name).schema.metadata.items()}
            self.assertEqual(sorted(recorded), sorted(METADATA_FACTS), msg=name)
            self.assertNotIn("layout", recorded, msg=name)
            self.assertEqual(recorded["stored_dim"], str(_STORED_DIM), msg=name)
            # Qwen3's document Instruction is the empty string, never None. It has to survive as
            # a recorded fact, or a later Invocation cannot tell "no Instruction" from "absent".
            self.assertEqual(recorded["document_instruction"], "", msg=name)

    def test_metadata_survives_a_reopen_and_an_add(self):
        sink = self.sink()
        self.write_all(sink, [_document("a.pdf")])
        recorded = _connect(self.path).open_table(DOCUMENTS_TABLE).schema.metadata
        self.assertEqual(recorded[b"model_id"], b"Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(recorded[b"chunk_budget_tokens"], b"32704")

    def test_table_metadata_is_write_once(self):
        # The immutability is the point: it makes the block an assertion about the table rather
        # than a comment on it. Only field-level replacement exists; the dataset-level
        # replace_schema_metadata is not on Table at all.
        table = _connect(self.path)
        self.sink().close()
        handle = table.open_table(DOCUMENTS_TABLE)
        self.assertFalse(hasattr(handle, "replace_schema_metadata"))
        self.assertTrue(hasattr(handle, "update_field_metadata"))

    def test_a_btree_index_on_document_name_exists_in_both_tables_and_no_vector_index(self):
        self.sink().close()
        for name in (DOCUMENTS_TABLE, CHUNKS_TABLE):
            indices = _connect(self.path).open_table(name).list_indices()
            self.assertEqual([(index.index_type, index.columns) for index in indices], [("BTree", ["document_name"])], msg=name)

    def test_reopening_does_not_rebuild_the_index(self):
        self.sink().close()
        first = list(_connect(self.path).open_table(DOCUMENTS_TABLE).list_indices())[0].index_uuid
        self.sink().close()
        second = list(_connect(self.path).open_table(DOCUMENTS_TABLE).list_indices())[0].index_uuid
        self.assertEqual(first, second)

    def test_metadata_values_are_all_strings(self):
        metadata = table_metadata(_invariants())
        self.assertEqual(len(metadata), 8)
        for key, value in metadata.items():
            self.assertIsInstance(value, str, msg=key)


class MetadataGuardTest(_SinkTestCase):
    def test_a_changed_fact_stops_before_any_row_is_written(self):
        self.write_all(self.sink(), [_document("a.pdf"), _document("b.pdf", seed=10)])
        before = _rows(self.path, DOCUMENTS_TABLE)

        second = LanceSink(self.path, _invariants(model_id="nvidia/Nemotron-3-Embed-8B-BF16"), batch_size=64)
        with self.assertRaises(SinkInvariantError) as caught:
            second.open()
        message = str(caught.exception)
        self.assertIn("Qwen/Qwen3-Embedding-0.6B", message)
        self.assertIn("nvidia/Nemotron-3-Embed-8B-BF16", message)
        self.assertEqual(_rows(self.path, DOCUMENTS_TABLE), before)

    def test_every_disagreeing_fact_is_reported_not_only_the_first(self):
        # A table built with a different model differs in several facts at once; reporting one at
        # a time turns one operator decision into four failed runs.
        self.sink().close()
        second = LanceSink(self.path, _invariants(model_id="other", native_dim=4096, query_instruction="query: "), batch_size=64)
        with self.assertRaises(SinkInvariantError) as caught:
            second.open()
        message = str(caught.exception)
        self.assertIn("3 invariant fact(s) disagree", message)
        for fact in ("model_id", "native_dim", "query_instruction"):
            self.assertIn(fact, message)

    def test_the_message_names_this_table_and_the_fix_that_is_not_the_manifests(self):
        # Both Sinks stop through one comparison (`npz_sink.compare_invariant_facts`). What
        # differs is what the message names: a `.lance` table rather than the manifest, and a
        # fix that cannot be "repair it", because table metadata is write-once.
        self.sink().close()
        with self.assertRaises(SinkInvariantError) as caught:
            LanceSink(self.path, _invariants(model_id="other")).open()
        message = str(caught.exception)
        self.assertIn(f"{self.path / DOCUMENTS_TABLE}.lance", message)
        self.assertIn("this table has", message)
        self.assertIn("nothing has been written.", message)
        self.assertIn("write-once", message)

    def test_a_matching_second_invocation_opens_and_appends(self):
        self.write_all(self.sink(), [_document("a.pdf")])
        self.write_all(self.sink(), [_document("b.pdf", seed=10)])
        self.assertEqual([name for name, _ in _keys(self.path, DOCUMENTS_TABLE)], ["a.pdf", "b.pdf"])

    def test_layout_is_not_compared_because_lancedb_has_no_filesystem_layout(self):
        self.write_all(self.sink(layout="bare"), [_document("a.pdf")])
        self.write_all(self.sink(layout="labelled"), [_document("b.pdf", run_label="other", seed=10)])
        self.assertEqual(len(_rows(self.path, DOCUMENTS_TABLE)), 2)

    def test_a_table_with_no_metadata_at_all_is_a_disagreement(self):
        # Not an embed database: every fact reads as absent rather than as a match.
        schema = pa.schema([pa.field("document_name", pa.string()), pa.field("vector", pa.list_(pa.float32(), _STORED_DIM))])
        _connect(self.path).create_table(DOCUMENTS_TABLE, schema=schema)
        with self.assertRaises(SinkInvariantError) as caught:
            LanceSink(self.path, _invariants()).open()
        self.assertIn("<absent>", str(caught.exception))


class WritePathTest(_SinkTestCase):
    def test_a_new_document_is_written_with_add_and_never_merge_insert(self):
        sink = self.sink()
        documents, chunks = self.record(sink)
        self.write_all(sink, [_document("a.pdf")])
        self.assertEqual(documents.calls, ["add"])
        self.assertEqual(chunks.calls, ["add"])

    def test_replacing_a_document_goes_through_merge_insert(self):
        self.write_all(self.sink(), [_document("a.pdf")])
        sink = self.sink()
        documents, chunks = self.record(sink)
        self.write_all(sink, [_document("a.pdf", n_chunks=1, seed=5)], is_new=False)
        self.assertEqual(documents.calls, ["merge_insert"])
        self.assertEqual(chunks.calls, ["merge_insert"])

    def test_a_document_the_table_already_holds_is_replaced_even_when_marked_new(self):
        # Design 11.2 leans on "LanceDB upserts harmlessly" to heal a crash between the two Sinks.
        # add() does not look for a matching row, so without this guard the heal would append a
        # second copy of every row instead.
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=3)])
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=3, seed=7)], is_new=True)
        self.assertEqual(len(_rows(self.path, DOCUMENTS_TABLE)), 1)
        self.assertEqual(len(_rows(self.path, CHUNKS_TABLE)), 3)

    def test_nothing_is_committed_until_the_batch_is_full(self):
        sink = self.sink(batch_size=4)
        for index in range(3):
            sink.write(_document(f"d{index}.pdf", seed=index), is_new=True)
        self.assertEqual(_connect(self.path).open_table(DOCUMENTS_TABLE).count_rows(), 0)
        sink.write(_document("d3.pdf", seed=3), is_new=True)
        self.assertEqual(_connect(self.path).open_table(DOCUMENTS_TABLE).count_rows(), 4)
        sink.close()

    def test_close_flushes_the_partial_batch(self):
        sink = self.sink(batch_size=64)
        sink.write(_document("a.pdf"), is_new=True)
        self.assertEqual(_connect(self.path).open_table(DOCUMENTS_TABLE).count_rows(), 0)
        sink.close()
        self.assertEqual(_connect(self.path).open_table(DOCUMENTS_TABLE).count_rows(), 1)

    def test_the_default_batch_is_sixty_four_and_one_batch_is_one_data_file(self):
        # One add() produces one data file, which is what makes batch 64 a fragmentation trade
        # (~1,562 fragments per 100k Documents) rather than a free choice.
        sink = LanceSink(self.path, _invariants())
        self.assertEqual(sink.batch_size, 64)
        sink.open()
        self.write_all(sink, [_document(f"d{index}.pdf", seed=index) for index in range(128)])
        for name in (DOCUMENTS_TABLE, CHUNKS_TABLE):
            data = os.listdir(self.path / f"{name}.lance" / "data")
            self.assertEqual(len(data), 2, msg=f"{name}: {data}")

    def test_writing_after_close_fails_rather_than_vanishing(self):
        sink = self.sink()
        sink.close()
        with self.assertRaises(RuntimeError):
            sink.write(_document("a.pdf"), is_new=True)


class ScopedDeleteTest(_SinkTestCase):
    def test_re_embedding_to_fewer_chunks_leaves_no_phantom_and_spares_the_neighbour(self):
        # Measured: a plain upsert writes chunk_index 0 and 1 and leaves 2 behind, giving the
        # Document a phantom third Chunk whose vector describes text that no longer exists.
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=3), _document("n.pdf", n_chunks=3, seed=20)])
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=2, seed=40)], is_new=False)

        self.assertEqual(_keys(self.path, CHUNKS_TABLE), [("a.pdf", 0), ("a.pdf", 1), ("n.pdf", 0), ("n.pdf", 1), ("n.pdf", 2)])
        self.assertEqual(_keys(self.path, DOCUMENTS_TABLE), [("a.pdf", 2), ("n.pdf", 3)])

    def test_the_replacement_vectors_are_the_new_ones(self):
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=2, seed=1)])
        replacement = _document("a.pdf", n_chunks=2, seed=50)
        self.write_all(self.sink(), [replacement], is_new=False)
        stored = {row["chunk_index"]: row["vector"] for row in _rows(self.path, CHUNKS_TABLE)}
        for index in (0, 1):
            np.testing.assert_allclose(stored[index], replacement.chunk_vectors[index], rtol=0, atol=0)

    def test_re_embedding_to_zero_chunks_removes_every_chunk_row(self):
        # The re-OCR that yields no text at all: the merge still runs, with an empty source, and
        # the scoped delete is the only thing that clears the old rows.
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=3), _document("n.pdf", n_chunks=2, seed=20)])
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=0)], is_new=False)
        self.assertEqual(_keys(self.path, CHUNKS_TABLE), [("n.pdf", 0), ("n.pdf", 1)])
        self.assertEqual(_keys(self.path, DOCUMENTS_TABLE), [("a.pdf", 0), ("n.pdf", 2)])

    def test_the_same_document_name_in_another_run_is_untouched(self):
        # run_label is part of the key, so two Runs holding the same PDF are two Documents.
        self.write_all(self.sink(), [_document("a.pdf", run_label="legal", n_chunks=3), _document("a.pdf", run_label="med", n_chunks=3, seed=30)])
        self.write_all(self.sink(), [_document("a.pdf", run_label="legal", n_chunks=1, seed=60)], is_new=False)
        by_run = sorted((row["run_label"], row["chunk_index"]) for row in _rows(self.path, CHUNKS_TABLE))
        self.assertEqual(by_run, [("legal", 0), ("med", 0), ("med", 1), ("med", 2)])


class PredicateQuotingTest(_SinkTestCase):
    def test_sql_literal_doubles_every_quote_and_keeps_its_own(self):
        self.assertEqual(_sql_literal("plain.pdf"), "'plain.pdf'")
        self.assertEqual(_sql_literal("law/O'Brien v. State.pdf"), "'law/O''Brien v. State.pdf'")
        self.assertEqual(_sql_literal("x' OR document_name != '"), "'x'' OR document_name != '''")

    def test_the_naive_predicate_is_what_breaks(self):
        # The reason the escape exists, kept as a regression guard on the premise: if lancedb ever
        # accepted this, the quoting decision would be worth re-examining rather than inherited.
        self.write_all(self.sink(), [_document("law/O'Brien v. State.pdf", n_chunks=2)])
        table = _connect(self.path).open_table(CHUNKS_TABLE)
        with self.assertRaises(ValueError) as caught:
            (
                table.merge_insert(["run_label", "document_name", "chunk_index"])
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete("run_label = 'legal' AND document_name = 'law/O'Brien v. State.pdf'")
                .execute(pa.Table.from_pylist([], schema=table.schema))
            )
        self.assertIn("Error tokenizing statement", str(caught.exception))

    def test_a_name_with_an_apostrophe_round_trips_through_the_delete_predicate(self):
        name = "law/O'Brien v. State.pdf"
        self.write_all(self.sink(), [_document(name, n_chunks=3), _document("plain.pdf", n_chunks=2, seed=20)])
        self.write_all(self.sink(), [_document(name, n_chunks=2, seed=40)], is_new=False)
        self.assertEqual(_keys(self.path, CHUNKS_TABLE), [("law/O'Brien v. State.pdf", 0), ("law/O'Brien v. State.pdf", 1), ("plain.pdf", 0), ("plain.pdf", 1)])

    def test_an_injection_shaped_name_deletes_no_other_document(self):
        # when_not_matched_by_source_delete deletes. An unescaped name shaped like this one would
        # widen the predicate to every other Document instead of failing loudly.
        hostile = "x' OR document_name != '"
        self.write_all(
            self.sink(),
            [_document(hostile, n_chunks=3), _document("a.pdf", n_chunks=2, seed=20), _document("b.pdf", n_chunks=2, seed=30)],
        )
        self.write_all(self.sink(), [_document(hostile, n_chunks=1, seed=40)], is_new=False)
        self.assertEqual(_keys(self.path, CHUNKS_TABLE), [("a.pdf", 0), ("a.pdf", 1), ("b.pdf", 0), ("b.pdf", 1), (hostile, 0)])

    def test_a_run_label_with_an_apostrophe_is_escaped_too(self):
        self.write_all(self.sink(), [_document("a.pdf", run_label="o'brien", n_chunks=3)])
        self.write_all(self.sink(), [_document("a.pdf", run_label="o'brien", n_chunks=1, seed=40)], is_new=False)
        self.assertEqual(_keys(self.path, CHUNKS_TABLE), [("a.pdf", 0)])


class EmptyDocumentTest(_SinkTestCase):
    def test_an_empty_document_is_one_null_row_and_no_chunk_rows(self):
        # It has to be recorded, or it is retried on every Invocation forever; with derived Resume
        # state the record has to be an output, because there is nowhere else for one to live.
        self.write_all(self.sink(), [_document("empty.pdf", n_chunks=0), _document("a.pdf", n_chunks=2, seed=20)])
        rows = {row["document_name"]: row for row in _rows(self.path, DOCUMENTS_TABLE)}
        self.assertEqual(rows["empty.pdf"]["n_chunks"], 0)
        self.assertIsNone(rows["empty.pdf"]["vector"])
        self.assertEqual([row["document_name"] for row in _rows(self.path, CHUNKS_TABLE)], ["a.pdf", "a.pdf"])

    def test_vector_search_skips_the_empty_document(self):
        # A zero vector was rejected precisely because it would sit in the index looking like data.
        real = _document("a.pdf", n_chunks=1, seed=3)
        self.write_all(self.sink(), [_document("empty.pdf", n_chunks=0), real])
        hits = _connect(self.path).open_table(DOCUMENTS_TABLE).search(real.document_vector).limit(10).to_list()
        self.assertEqual([hit["document_name"] for hit in hits], ["a.pdf"])

    def test_an_empty_document_still_counts_as_known(self):
        self.write_all(self.sink(), [_document("empty.pdf", n_chunks=0)])
        self.assertEqual(self.sink().known(), {("legal", "empty.pdf")})


class ColumnCoverageTest(_SinkTestCase):
    def test_every_column_is_populated_on_every_write(self):
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=2)])
        for table in (DOCUMENTS_TABLE, CHUNKS_TABLE):
            for row in _rows(self.path, table):
                for column, value in row.items():
                    self.assertIsNotNone(value, msg=f"{table}.{column}")

    def test_an_omitted_column_lands_as_none_with_no_complaint(self):
        # The hazard the explicit row dicts exist for: the schema catches typos, not omissions.
        self.sink().close()
        table = _connect(self.path).open_table(CHUNKS_TABLE)
        table.add([{"document_name": "a.pdf", "run_label": "legal", "chunk_index": 0, "vector": [0.0] * _STORED_DIM}])
        self.assertIsNone(_rows(self.path, CHUNKS_TABLE)[0]["token_count"])

        with self.assertRaises(ValueError) as caught:
            table.add([{"document_name": "a.pdf", "run_label": "legal", "chunk_index": 1, "vector": [0.0] * _STORED_DIM, "surprise": 1}])
        self.assertIn("does not exist in table schema", str(caught.exception))

    def test_the_stored_row_carries_the_chunk_offsets_and_flags(self):
        doc = _document("a.pdf", n_chunks=2)
        self.write_all(self.sink(), [doc])
        rows = sorted(_rows(self.path, CHUNKS_TABLE), key=lambda row: row["chunk_index"])
        for index, (row, chunk) in enumerate(zip(rows, doc.chunks)):
            self.assertEqual(row["chunk_index"], index)
            self.assertEqual(row["start_char"], chunk.start_char)
            self.assertEqual(row["end_char"], chunk.end_char)
            self.assertEqual(row["first_page"], chunk.first_page)
            self.assertEqual(row["last_page"], chunk.last_page)
            self.assertEqual(row["token_count"], chunk.token_count)
            self.assertEqual(row["is_partial_page"], chunk.is_partial_page)

    def test_created_round_trips_as_the_same_instant(self):
        self.write_all(self.sink(), [_document("a.pdf")])
        self.assertEqual(_rows(self.path, DOCUMENTS_TABLE)[0]["created"], _CREATED)


class VectorWidthTest(_SinkTestCase):
    def test_a_wider_vector_than_stored_dim_is_refused(self):
        # The wrong-model check the column type makes for free. Building the batch against the
        # table's own schema moves the refusal one layer earlier than lancedb's `Cast error:
        # Cannot cast to FixedSizeList(...)`, and pyarrow names both widths too -- what matters is
        # that no partial batch reaches the table.
        sink = self.sink()
        with self.assertRaises(ValueError) as caught:
            sink.write(_document("a.pdf", n_chunks=1, dim=_NATIVE_DIM), is_new=True)
            sink.flush()
        self.assertIn(str(_STORED_DIM), str(caught.exception))
        self.assertIn(str(_NATIVE_DIM), str(caught.exception))
        self.assertEqual(_connect(self.path).open_table(CHUNKS_TABLE).count_rows(), 0)


class KnownTest(_SinkTestCase):
    def test_a_fresh_database_knows_nothing(self):
        self.assertEqual(self.sink().known(), set())

    def test_known_is_run_label_then_document_name(self):
        self.write_all(self.sink(), [_document("a.pdf", run_label="legal"), _document("a.pdf", run_label="med", seed=20)])
        self.assertEqual(self.sink().known(), {("legal", "a.pdf"), ("med", "a.pdf")})

    def test_known_is_a_copy_the_caller_may_consume(self):
        self.write_all(self.sink(), [_document("a.pdf")])
        sink = self.sink()
        first = sink.known()
        first.clear()
        self.assertEqual(sink.known(), {("legal", "a.pdf")})
        sink.close()

    def test_known_covers_more_documents_than_one_page_would(self):
        # limit(None) is explicit for this reason: a truncated set re-embeds Documents the table
        # already holds, and a default page of 10 would truncate every real corpus.
        docs = [_document(f"d{index}.pdf", seed=index) for index in range(25)]
        self.write_all(self.sink(batch_size=4), docs)
        self.assertEqual(len(self.sink().known()), 25)


class _FailingDocuments:
    """Passes `delete` through and fails `merge_insert`, which is precisely the crash window."""

    def __init__(self, inner):
        self.inner = inner

    def merge_insert(self, *args, **kwargs):
        raise RuntimeError("crash between the two commits")

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ReplaceCrashTest(_SinkTestCase):
    """`_replace` is two commits with no transaction across them, so the order decides the damage."""

    def test_a_crash_before_the_documents_write_leaves_the_marker_missing_not_stale(self):
        # `known()` reads `documents` alone, which makes that row the commit marker. With the
        # delete last, this crash left the marker in place describing three Chunks that the
        # merge below had already replaced with two -- and Resume, seeing the name, would skip
        # that Document for good. Deleting first means a crash anywhere leaves no marker, and
        # the next Invocation repairs it by re-embedding.
        self.write_all(self.sink(), [_document("a.pdf", n_chunks=3)])

        sink = self.sink()
        sink._documents = _FailingDocuments(sink._documents)  # type: ignore[bad-assignment]
        sink.write(_document("a.pdf", n_chunks=2, seed=5), is_new=False)
        with self.assertRaises(RuntimeError):
            sink.flush()

        # The `chunks` write landing is what puts this mid-window rather than before it: the
        # tables really are inconsistent here, and the marker's absence is what makes that
        # temporary instead of permanent.
        self.assertEqual(len(_rows(self.path, CHUNKS_TABLE)), 2)
        self.assertEqual(self.sink().known(), set())


if __name__ == "__main__":
    unittest.main()
