"""The LanceDB Sink: two tables, write-once table metadata, `add()` when new and `merge_insert` to replace.

Everything asserted here was **measured against the installed lancedb 0.37.1 / pyarrow 25.0.1 /
numpy 2.5.2**, never read from documentation -- several comparable projects' docstrings contradict
their own code (LanceDB's own retry default says 10 in the docstring and 7 in the signature).

**Two tables, not one table with an `is_doc` column.** Vector search reads a whole table, so a single
table would return a mixture of Chunk vectors and Document vectors on every query, and the Consumer
would have to remember a filter forever -- forgetting it yields wrong neighbours rather than an
error. Two tables make the wrong query impossible instead of merely discouraged. Both name the vector
column `vector`, so `.search()` works without naming a column.

**`run_label` is a column, not a table name.** Table-per-label was rejected on a concrete obstacle:
table names accept only `[A-Za-z0-9._-]`, while `_parse_runs` (`src/paperscale/cli.py`) only strips
whitespace and checks non-empty and unique, so table-per-label needs a label sanitizer -- a second
name-mangling rule with its own collision question, next to the one `names.py` already answers.
Namespaces exist in 0.37.1 and cannot be used: `db.create_namespace(["legal"])` succeeds and
`list_namespaces()` lists it, but `create_table` has no `namespace` parameter in the Python sync API.
To keep two embedding models side by side, use a second database directory; the metadata comparison
in `open()` makes the alternative fail loudly.

**Fragmentation is a read-time cost the Consumer can pay off in one call.** One `add()` produces one
data file, so batch 64 leaves roughly 1,562 fragments per 100k Documents. It costs no file
descriptors -- a full scan holds a flat +16 over baseline whether the table carries 50 fragments or
600, because Lance reads through a bounded pool rather than one handle per fragment -- and one
`Table.optimize(cleanup_older_than=timedelta(0))` collapsed forty fragments to one with every row
intact. That call is the Consumer's to make after a large Invocation, not `embed`'s.

**Single writer.** `merge_insert` is materially worse under concurrency than `add`, so the shape is
*workers embed, one writer commits*. Nothing here is thread-safe and nothing here needs to be.

Design authority: `docs/design/embed.md` section 9 (all of it), 11.4 for the empty Document, and 16.1
for the compaction and file-descriptor measurements.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paperscale.embed.invariants import Invariants, compare_invariant_facts

if TYPE_CHECKING:
    import pyarrow as pa
    from lancedb.db import DBConnection
    from lancedb.table import Table

    from paperscale.embed.vectors import EmbeddedDocument

logger = logging.getLogger(__name__)

__all__ = ["CHUNKS_TABLE", "DOCUMENTS_TABLE", "METADATA_FACTS", "LanceSink", "table_metadata"]

DOCUMENTS_TABLE = "documents"
CHUNKS_TABLE = "chunks"

# Eight facts, not the manifest's nine: `layout` is a filesystem fact and LanceDB has no filesystem
# layout, so recording it here would be a promise about a directory tree these tables do not have.
METADATA_FACTS = (
    "model_id",
    "stored_dim",
    "native_dim",
    "document_instruction",
    "query_instruction",
    "pooling",
    "chunker",
    "chunk_budget_tokens",
)


def table_metadata(invariants: Invariants) -> dict[str, str]:
    """The eight invariant facts, stringified, as `pa.schema(..., metadata=...)` wants them.

    Arrow schema metadata is a bytes-to-bytes map, so every value goes on the wire as a string and
    comes back as bytes; `stored_dim` is `"768"` in the table and `768` in the `Invariants`, which is
    why `_compare_metadata` compares the stringified forms on both sides rather than the values.

    Repeating these eight as *columns* was rejected: there is exactly one model per table by
    construction, so `WHERE model_id = ...` answers a question nobody has.
    """
    return {fact: str(getattr(invariants, fact)) for fact in METADATA_FACTS}


def _sql_literal(value: str) -> str:
    """Quote a string as a SQL literal, doubling every `'` inside it.

    **This is the only place a Document name may enter a predicate.** The delete predicate below is a
    SQL string built from a filesystem path, and `law/O'Brien v. State.pdf` -- an unremarkable name in
    a legal corpus -- makes the naive f-string raise `Error tokenizing statement`. It failed loudly
    that time. A name shaped like `x' OR document_name != '` would not, and
    `when_not_matched_by_source_delete` *deletes*. Every identifier interpolated here comes from
    `Source-File`, which `names.py` knowingly leaves unnormalized and which nobody in this pipeline
    controls.

    Returns the literal **with** its surrounding quotes, so a caller cannot forget them.
    """
    return "'" + value.replace("'", "''") + "'"


def _document_scope(run_label: str, document_name: str) -> str:
    """The predicate that bounds a delete to one Document and no other."""
    return f"run_label = {_sql_literal(run_label)} AND document_name = {_sql_literal(document_name)}"


def _documents_schema(stored_dim: int, metadata: dict[str, str]) -> pa.Schema:
    """One row per Document.

    `vector` is `fixed_size_list<float32, stored_dim>`, which enforces the wrong-model check for
    free: a 4096-wide vector into a 768-wide column is `Cast error: Cannot cast to
    FixedSizeList(768): value at index 0 has length 4096`. Unlike the `.npz` Sink's manifest, this
    check needs no separate file and cannot be skipped.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("document_name", pa.string()),
            pa.field("run_label", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("source_digest", pa.string()),
            pa.field("created", pa.timestamp("us", tz="UTC")),
            pa.field("n_chunks", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), stored_dim)),
        ],
        metadata=metadata,
    )


def _chunks_schema(stored_dim: int, metadata: dict[str, str]) -> pa.Schema:
    """One row per Chunk.

    It deliberately does not repeat `source_file`, `source_digest` or `created`: `document_name` is
    the identity and joins the two tables. `token_count` does sit here, next to the Chunk vectors, so
    a Consumer reading `chunks` alone can rebuild the Document vector without a join.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("document_name", pa.string()),
            pa.field("run_label", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), stored_dim)),
            pa.field("start_char", pa.int32()),
            pa.field("end_char", pa.int32()),
            pa.field("first_page", pa.int32()),
            pa.field("last_page", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("is_partial_page", pa.bool_()),
        ],
        metadata=metadata,
    )


def _document_row(doc: EmbeddedDocument) -> dict[str, Any]:
    """Every column of `documents`, named, on every write.

    Omitting a column is **not** an error: an unknown column is rejected (`field 'surprise' does not
    exist in table schema`) but a row missing `n_chunks` lands with `None` and no complaint, and
    `pa.Table.from_pylist` fills an absent key with null just as happily. The schema catches typos,
    not omissions, so the row is one literal dict rather than something assembled from whatever the
    Document happens to carry.

    `tolist()` widens float32 to Python floats and pyarrow casts them straight back, which is exact
    -- every float32 has an exact float64 image.
    """
    return {
        "document_name": doc.document_name,
        "run_label": doc.run_label,
        "source_file": doc.source_file,
        "source_digest": doc.source_digest,
        "created": doc.created,
        "n_chunks": len(doc.chunks),
        # NULL, not a zero vector. A zero vector is not a unit vector, nothing else in the store is
        # anything but a unit vector, and it would sit in a search index looking like data. A NULL
        # fixed_size_list is accepted, reads back as None, and vector search skips the row
        # (measured) -- so the empty output says what actually happened instead of pretending.
        "vector": None if not doc.chunks else doc.document_vector.tolist(),
    }


def _chunk_rows(doc: EmbeddedDocument) -> list[dict[str, Any]]:
    """Every column of `chunks`, named, for each Chunk of one Document.

    `chunk_index` is derived here rather than carried on the `Chunk`: it is exactly the row's
    position in `doc.chunks`, and `doc.chunk_vectors[i]` is that Chunk's vector, so a stored copy
    would be a second source of truth for something the parallel arrays already state.
    """
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(doc.chunks):
        rows.append(
            {
                "document_name": doc.document_name,
                "run_label": doc.run_label,
                "chunk_index": index,
                "vector": doc.chunk_vectors[index].tolist(),
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "first_page": chunk.first_page,
                "last_page": chunk.last_page,
                "token_count": chunk.token_count,
                "is_partial_page": chunk.is_partial_page,
            }
        )
    return rows


class LanceSink:
    """Writes one Invocation's Documents into one LanceDB database of two tables.

    Buffers whole Documents and flushes every `batch_size` of them. **64 is not a flag**: it anchors
    to `--concurrency 64`, so one batch is about one full sweep of in-flight requests rather than an
    unrelated constant; it bounds a crash to 64 Documents of GPU time; and because a Document is done
    only when *every* enabled Sink holds it, a smaller batch means LanceDB lags the `.npz` tree by
    less. Both costs are structural and neither is observable mid-run, so a flag would invite tuning
    against a metric that does not exist.
    """

    def __init__(self, path: Path, invariants: Invariants, *, batch_size: int = 64) -> None:
        self.path = Path(path)
        self.invariants = invariants
        self.batch_size = batch_size
        self._db: DBConnection | None = None
        self._documents: Table | None = None
        self._chunks: Table | None = None
        self._doc_schema: pa.Schema | None = None
        self._chunk_schema: pa.Schema | None = None
        self._pending: list[tuple[EmbeddedDocument, bool]] = []
        # The startup snapshot of what the `documents` table already holds. It is both Resume's
        # answer and the guard in `write` that keeps `add()` from duplicating an existing Document.
        self._known: set[tuple[str, str]] = set()
        self._corrected = 0

    def open(self) -> None:
        """Connect, adopt or create both tables, and stop before the first row if the model differs.

        The metadata comparison is the reason this Sink has an `open()` at all. #36 found the
        counter-example that earns it: **ColBERT's resume carries a literal `# TODO: Verify config
        matches`**, so resuming with a different checkpoint silently corrupts the index. Table
        metadata is write-once here -- `Table` exposes only the field-level `replace_field_metadata`
        and `update_field_metadata`, and the dataset-level `replace_schema_metadata` raises
        `ImportError` without a separate `pylance` install -- and that immutability is the point: it
        makes the block an assertion about the table rather than a comment on it.
        """
        import lancedb

        metadata = table_metadata(self.invariants)
        self._doc_schema = _documents_schema(self.invariants.stored_dim, metadata)
        self._chunk_schema = _chunks_schema(self.invariants.stored_dim, metadata)

        db = lancedb.connect(str(self.path))
        # `list_tables()` rather than the deprecated `table_names()`; `limit=None` returns every name
        # in one page, and this database holds exactly the two tables below by construction.
        existing = set(db.list_tables().tables)
        self._db = db
        self._documents = self._open_or_create(db, DOCUMENTS_TABLE, self._doc_schema, existing)
        self._chunks = self._open_or_create(db, CHUNKS_TABLE, self._chunk_schema, existing)
        self._known = self._scan_known()

    def known(self) -> set[tuple[str, str]]:
        """`{(run_label, document_name)}` this Sink already holds, as of `open()`.

        Read once at startup and never inside the Document loop -- there is no per-Document lookup
        anywhere in this Sink. A copy, because Resume combines these sets and this one is also the
        guard `write` consults.
        """
        self._require_open()
        return set(self._known)

    def write(self, doc: EmbeddedDocument, *, is_new: bool) -> None:
        """Buffer one finished Document; flush once `batch_size` of them have arrived.

        `is_new` must be asked of **this Sink** (`known()`), not of the Resume intersection across
        Sinks. Design 11.2 leans on LanceDB tolerating a double write -- a crash between the two
        Sinks leaves a Document in one and not the other, the intersection says "not done", and the
        next Invocation writes both, where "one write is a no-op and the other completes". That is
        only true if the Document goes down the `merge_insert` path: `add()` does not look for a
        matching row, so an `is_new=True` for a Document the table already holds appends a **second**
        copy of every one of its rows. The guard below turns that corruption into the replace it
        meant to be, and the count is reported at `close()`.
        """
        self._require_open()
        key = (doc.run_label, doc.document_name)
        if is_new and key in self._known:
            self._corrected += 1
            logger.debug("lancedb: %r in run %r is already in the table; replacing rather than appending", doc.document_name, doc.run_label)
            is_new = False
        self._pending.append((doc, is_new))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """Commit the buffered Documents: one `add()` for the new ones, one `merge_insert` per replace."""
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        new = [doc for doc, is_new in pending if is_new]
        if new:
            self._add(new)
        for doc, is_new in pending:
            if not is_new:
                self._replace(doc)
        self._known.update((doc.run_label, doc.document_name) for doc, _ in pending)

    def close(self) -> None:
        """Flush what is left and drop the handles, so a later `write` fails instead of vanishing."""
        self.flush()
        if self._corrected:
            logger.info(
                "lancedb: %d Document(s) arrived marked new but were already in the table, and were replaced instead of appended. "
                "That is the expected repair after a crash between Sinks; if it is every Document, `is_new` is being taken from the "
                "Resume intersection rather than from this Sink's known().",
                self._corrected,
            )
        self._documents = None
        self._chunks = None
        self._db = None

    # -- internals ---------------------------------------------------------------------------

    def _require_open(self) -> tuple[Table, Table]:
        """Both table handles, or the error that says `open()` was skipped or `close()` already ran.

        It returns the pair rather than only raising so the write paths read the handles here, once,
        instead of each dereferencing an attribute the type says may be `None`.
        """
        if self._documents is None or self._chunks is None:
            raise RuntimeError("LanceSink.open() must be called before writing or reading, and close() ends it")
        return self._documents, self._chunks

    def _open_or_create(self, db: DBConnection, name: str, schema: pa.Schema, existing: set[str]) -> Table:
        """Adopt an existing table after checking its metadata, or create it with ours.

        `create_table(..., exist_ok=True)` looks like it does this in one call and must not be used:
        on any difference at all it raises `Schema Error: Provided schema does not match existing
        table schema`, which names neither the fact that disagrees nor either value, and it treats a
        changed `stored_dim` and a changed `model_id` as the same opaque sentence.
        """
        if name in existing:
            table = db.open_table(name)
            self._compare_metadata(name, table.schema.metadata)
        else:
            table = db.create_table(name, schema=schema)
        self._ensure_index(table)
        return table

    def _compare_metadata(self, name: str, recorded_raw: dict[Any, Any] | None) -> None:
        """The eight metadata facts, through the comparison both Sinks report with.

        The recorded side is decoded first: Arrow schema metadata is a bytes-to-bytes map, so the
        stringified forms are what both sides compare -- `stored_dim` is `"768"` in the table and
        `768` in the `Invariants`.
        """
        compare_invariant_facts(
            {_decode(key): _decode(value) for key, value in (recorded_raw or {}).items()},
            table_metadata(self.invariants),
            subject=f"{self.path / name}.lance",
            holder="this table",
            nothing_yet="nothing has been written.",
            remedy=(
                "Re-run with the settings that built these tables, or point the LanceDB Sink at a fresh directory -- the table metadata is "
                "write-once and cannot be corrected in place."
            ),
        )

    def _ensure_index(self, table: Table) -> None:
        """A `BTree` scalar index on `document_name`, in both tables.

        `create_scalar_index` is deprecated as of 0.25.0 and raises `DeprecatedWarning`; the
        `config=BTree()` form raises nothing. The index is lossless, so it costs only build time, and
        it serves the two lookups this design performs: the `merge_insert` key match and Resume's
        question. Building it on the empty table at creation registers it, and rows added afterwards
        are answered by a scan until a Consumer's `optimize()` folds them in -- slower, never wrong.

        **No vector index.** `create_index` builds IVF_PQ, which is lossy: it trades recall for speed,
        and the right trade depends on a corpus size and a query pattern that live in the Consumer.
        Search works with no index at all -- brute force, exact, returns `_distance` -- so the default
        is correct rather than merely absent, and there is no index-build phase to wait on.
        """
        from lancedb.index import BTree

        if any("document_name" in config.columns for config in table.list_indices()):
            return
        table.create_index("document_name", config=BTree())

    def _scan_known(self) -> set[tuple[str, str]]:
        """`SELECT document_name, run_label FROM documents`, served by the `BTree`.

        `limit(None)` is explicit because the alternative is not an error: a default limit would
        truncate the set silently, and a short Resume set re-embeds Documents the table already holds
        -- which, without `write`'s guard, is a duplicate row for every one of them.
        """
        documents, _ = self._require_open()
        rows = documents.search().select(["document_name", "run_label"]).limit(None).to_list()
        return {(row["run_label"], row["document_name"]) for row in rows}

    def _add(self, docs: list[EmbeddedDocument]) -> None:
        """Append Documents no row in either table matches.

        `add()` rather than `merge_insert` for the new case is the amended write path (#40). A
        `merge_insert` is a read-modify-write against the **whole table**, so each call costs
        O(table) rather than O(batch) and N Documents at batch B cost O(N^2/B) -- which put batch size
        in a fight with the crash-loss window with no comfortable value in between. For a new
        Document it buys nothing: there is no matching row to read, update or delete.

        **`documents` is written last, and that ordering is load-bearing** for the same reason the
        `.npz` Sink writes its sidecar before the `.npz`: Resume reads `documents`, so the marker must
        not appear before the rows it claims. A crash between the two calls leaves chunk rows whose
        Document is unknown, and the re-embed appends a second copy of them -- bounded by one batch,
        and the one gap in this Sink the Resume intersection cannot heal on its own.
        """
        import pyarrow as pa

        documents, chunks = self._require_open()
        chunk_rows = [row for doc in docs for row in _chunk_rows(doc)]
        if chunk_rows:
            chunks.add(pa.Table.from_pylist(chunk_rows, schema=self._chunk_schema))
        documents.add(pa.Table.from_pylist([_document_row(doc) for doc in docs], schema=self._doc_schema))

    def _replace(self, doc: EmbeddedDocument) -> None:
        """Overwrite one Document that the tables already hold.

        **The scoped delete on `chunks` is not optional, and it was measured.** Seed a Document with
        `chunk_index` 0, 1, 2 and re-embed after a re-OCR that yields two Chunks: a plain upsert
        writes 0 and 1 and *leaves `chunk_index = 2` behind*. The Document then carries a phantom
        third Chunk whose vector describes text that no longer exists, and a Document vector
        recomputed from that table is wrong. Scoping the predicate to this Document leaves every other
        Document's rows untouched.

        A Document that re-OCR'd to nothing still runs the `chunks` merge with an empty source, which
        is exactly how its old Chunk rows are removed (measured: an empty source with the scoped
        delete removed all three seeded rows).
        """
        import pyarrow as pa

        documents, chunks = self._require_open()
        scope = _document_scope(doc.run_label, doc.document_name)
        (
            chunks.merge_insert(["run_label", "document_name", "chunk_index"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete(scope)
            .execute(pa.Table.from_pylist(_chunk_rows(doc), schema=self._chunk_schema))
        )
        # `merge_insert` counts a NULL vector as a bad vector and refuses the batch, where `add()`
        # takes the same row without comment -- lancedb's `_handle_bad_vector_column` folds null into
        # "wrong dimension" on purpose (`or_kleene(is_null, ...)`, with a comment saying so). So the
        # empty Document, whose row is a deliberate NULL, needs `on_bad_vectors="null"`, which
        # replaces a bad vector with the null it already is. It is asked for **only** in that case:
        # the same flag on a Document that has Chunks would silently null a NaN vector instead of
        # raising, and a NaN reaching a Sink is a bug worth stopping for. A wrong *width* never gets
        # this far -- `from_pylist` against the table's own schema refuses it first.
        rows = pa.Table.from_pylist([_document_row(doc)], schema=self._doc_schema)
        on_bad_vectors = "null" if not doc.chunks else "error"
        (
            documents.merge_insert(["run_label", "document_name"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows, on_bad_vectors=on_bad_vectors)
        )


def _decode(value: Any) -> Any:
    """Arrow schema metadata comes back as bytes on both sides of the map."""
    return value.decode("utf-8") if isinstance(value, bytes) else value
