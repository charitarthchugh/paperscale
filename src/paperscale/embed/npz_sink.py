"""The `.npz` Sink: one Invocation manifest, one sidecar per Document, eight arrays.

**Provenance splits by scope, not by convenience**, and the reason is a hard limit
rather than a preference. `.npz` has no metadata header, and object arrays are
*impossible*, not merely unwise: `np.savez` will happily accept a Python dict, but
reading it back raises `ValueError: Object arrays cannot be loaded when
allow_pickle=False`, and asking a Consumer for `allow_pickle=True` is asking it to
execute whatever the file contains. Anything stored inside an `.npz` therefore has
to be a real dtype.

So the facts that never vary within an Invocation live in one manifest at the
output root, the four facts that vary per Document live in a JSON sidecar beside
the vectors, and the `.npz` holds arrays only (design 8.1).

    <out>/paperscale-embed.json                      <- the Invocation manifest
    <out>/paperscale-embed-failures.txt              <- rewritten each Invocation
    <out>/run/media/cc/law/doc9419897.pdf.npz        <- 8 arrays, nothing else
    <out>/run/media/cc/law/doc9419897.pdf.json       <- the Document sidecar

Three decisions below look like details and are not:

**Write order.** Sidecar first, then the `.npz`, each to a temporary name and
renamed. Rename is atomic within a filesystem, which buys the invariant *if the
`.npz` exists, the sidecar exists* -- and Resume uses the `.npz` alone as its
completion marker, so the ordering is load-bearing rather than tidy (design 8.5).
Two shipping projects have exactly the bug it prevents: `unstructured-ingest`'s
`write_data` opens the destination and dumps into it with no temp swap (while an
atomic writer sits unused in the same module), and ColBERT does the same with
`.residuals.pt`. In both, a crash leaves a truncated file that satisfies the
resume existence check and stays wrong forever.

**Eight arrays, not ten.** `chunk_index` and `n_chunks` are dropped: they are
exactly `np.arange(len(token_count))` and `len(token_count)`, and a stored copy of
a derived value is a second source of truth waiting to disagree with the first.
There is no `normalized` field and no `engine` field either -- normalization is
unconditional and vLLM is the only supported engine, so both would be constant
true and record nothing. That deliberately overrides #26's own ticket text, which
listed `normalized` as mandatory.

**`savez`, not `savez_compressed`.** Measured on the final layout: 8,225 -> 7,400 B
for one Chunk, 42,248 -> 38,908 B for twelve. About 10%, paid for with CPU on every
write and decompression on every read of a format whose Consumer is a classifier
build that reads it many times. Normalized float vectors are close to random bytes,
so deflate has nothing to remove. (An earlier 0.32 ratio that made compression look
worthwhile was an artifact of UTF-32 padding in a layout that no longer exists.)

Chunk text is not stored, for the same family of reasons: the Record already holds
the text and `start_char`/`end_char` are exact, numpy stores unicode as fixed-width
UTF-32 (46,473 B against 11,421 B on a real 8,700-character Document), and storing
UTF-8 bytes instead would put **two incompatible coordinate systems in one file**,
since the offsets count characters and a byte blob is indexed by bytes. That defect
surfaces on the first non-ASCII character, which in a legal corpus is not
hypothetical.

numpy is imported inside the functions that need it: it ships in the optional
`embed` extra and `import paperscale.embed` must keep working without it.

Design: `docs/design/embed.md` sections 8, 11.4 and 7.5.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Callable, TYPE_CHECKING

from paperscale.embed.names import RESERVED_PREFIX, NameCollisionError
from paperscale.version import VERSION

if TYPE_CHECKING:
    from paperscale.embed.vectors import EmbeddedDocument

logger = logging.getLogger(__name__)

# Derived from the reserved prefix rather than spelled out twice: `names.check_collisions`
# refuses any Document whose name lands on that prefix, and the two constants drifting
# apart would leave the guard defending a filename nothing writes.
MANIFEST_NAME = f"{RESERVED_PREFIX}.json"
FAILURES_NAME = f"{RESERVED_PREFIX}-failures.txt"

# The two invariant facts that are properties of this implementation rather than of the
# Invocation's flags. They are constants so run.py names them once instead of retyping two
# string literals that the manifest comparison would then reject on a typo.
POOLING = "token_weighted_mean"
CHUNKER = "greedy_page_pack"


@dataclasses.dataclass(frozen=True)
class Invariants:
    """The nine manifest facts a second Invocation must agree with, or stop.

    This is the check the Adapter's `native_dim` assertion cannot make. That assertion
    catches a server serving a model of the wrong *width*; it says nothing about an
    Invocation appended to a tree an earlier Invocation built with a **different model of
    the same width** -- `qwen3-embedding-8b` over `nemotron-3-embed-8b`, both 4096. With
    content detection removed (standing decision 7) Resume will not catch that either: the
    names match, so every Document is skipped and the tree ends up half one model's
    vectors and half another's, in one search index, silently.

    `layout` is in the block for a related reason (design 11.3). The run-label directory
    appears only when one Invocation embeds more than one Run, so `bare` and `labelled`
    are two layouts over one output directory. The failure a change causes is not silent
    *mixing* but silent **duplication**: the new paths match nothing, every Document is
    re-embedded into a parallel subtree, and the old tree is orphaned.

    `sinks` is deliberately **not** here -- the enabled-Sink set is allowed to change, and
    a change warns rather than stops (design 8.2).
    """

    model_id: str
    stored_dim: int
    native_dim: int
    document_instruction: str
    query_instruction: str
    pooling: str
    chunker: str
    chunk_budget_tokens: int
    layout: str


class SinkInvariantError(RuntimeError):
    """This output tree was built by an Invocation whose settings differ from this one's."""


def _utc_iso(when: datetime.datetime) -> str:
    """ISO-8601 with a literal `Z`, seconds resolution -- the form design 8.2/8.3 shows.

    Naive datetimes are refused rather than assumed to be UTC. A naive `created` silently
    means "whatever the writing machine's clock was set to", and both Sinks serialize this
    field, so the two would disagree about the same Document by the machine's UTC offset.
    The Invocation sets `created` once per Document from one clock, so this fires on the
    first Document or on none.
    """
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise ValueError(f"created must be timezone-aware; got the naive datetime {when!r}")
    return when.astimezone(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, body: Callable[[BinaryIO], Any]) -> None:
    """Write through a temporary sibling and `os.replace` onto the destination.

    The temporary lives in the destination's own directory because rename is atomic only
    *within* a filesystem; a temp under `/tmp` would degrade to a copy across a mount
    boundary and lose the atomicity this exists for. `mkstemp` rather than a fixed
    `.tmp` name so two Documents that somehow derive one path cannot shred each other's
    partial writes, and so a leftover from a killed Invocation is never reused.

    The temporary is named `<final>.<random>.tmp`, which matters to `known()`: it must not
    end in `.npz` or the walk would count a half-written file as a finished Document.

    Cleanup catches `BaseException` so `KeyboardInterrupt` -- the most likely way an
    operator ends a twelve-hour Invocation -- leaves no debris either. There is no
    `fsync`: this defends against process death and cancellation, which is the recorded
    failure (both of #36's shipping examples are process-level). Surviving a power cut
    would need an fsync of each file *and* of the parent directory, at a per-Document cost
    the design did not weigh -- and the losing direction there is benign, since an `.npz`
    that never lands is simply a Document Resume re-embeds.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            body(handle)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Pretty-printed UTF-8 JSON, atomically.

    `indent=2` is not decoration. Design 8.6 pays a 4,096-byte filesystem block per
    sidecar for an ~89-byte file, and the one thing that buys is that a person -- or any
    tool, without numpy -- can read a Document's identity. A single-line dump would spend
    the same block and give that up. `ensure_ascii=False` for the same reason: an escaped
    Instruction is unreadable in the file whose whole job is being readable.
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(path, lambda handle: handle.write(text.encode("utf-8")))


class NpzSink:
    """Manifest, sidecars and arrays under one output directory.

    Lifecycle: `open()` exactly once in the **parent process** before any worker starts,
    then `write()` per Document, then `write_failures()` at the end. `open()` is not a
    worker's to call -- concurrent appends to the manifest would race and corrupt the one
    record of how a tree built over several Invocations came to be (design 8.2).

    Call `open()` even under `--no-npz`. `<out>` exists in that case anyway, because the
    failures file lives there, and the invariant comparison is worth having whether or not
    this Sink writes vectors (design 17.2 item 1). Under `--no-npz` the Sink is simply left
    out of the Resume intersection and `write()` is never called.
    """

    def __init__(self, out: Path, invariants: Invariants, sinks: list[str]) -> None:
        self.out = Path(out)
        self.invariants = invariants
        self.sinks = list(sinks)
        # What the manifest said before this Invocation appended to it. None until `open()`
        # runs, and still None when this Invocation created the tree. resume.py renders the
        # count-bearing user-facing line from it ("--lancedb is new; this will re-embed N
        # Documents"); this Sink cannot, because it does not know the corpus size.
        self.previous_sinks: list[str] | None = None

    # -- manifest ---------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.out / MANIFEST_NAME

    def open(self) -> None:
        """Compare the nine invariants against the manifest, then append one entry.

        On disagreement this raises **before any Document is written**, which is the whole
        point: a mismatched Invocation that got as far as writing has already put two
        models' vectors in one tree.

        On a match, a `{created, paperscale_version}` entry is appended -- about 60 bytes,
        and the only record of how a multi-Invocation tree came to be. Overwriting the
        manifest was rejected outright: it would leave the file describing vectors it did
        not describe.
        """
        self.out.mkdir(parents=True, exist_ok=True)
        manifest = self._read_manifest()
        if manifest is None:
            manifest = {}
        else:
            self._compare_invariants(manifest)
            self.previous_sinks = [s for s in manifest.get("sinks") or [] if isinstance(s, str)]
            self._warn_if_sinks_changed(self.previous_sinks)

        invocations = [e for e in manifest.get("invocations") or [] if isinstance(e, dict)]
        # Rebuilt rather than mutated in place so the key order on disk always matches
        # design 8.2's listing -- nine invariants, then `sinks`, then the log -- however
        # an earlier version happened to order them.
        written: dict[str, Any] = dict(dataclasses.asdict(self.invariants))
        written["sinks"] = self.sinks
        written["invocations"] = invocations + [{"created": _utc_iso(datetime.datetime.now(datetime.timezone.utc)), "paperscale_version": VERSION}]
        _write_json(self.manifest_path, written)

    def _read_manifest(self) -> dict[str, Any] | None:
        """The manifest as a dict, or None when this Invocation is creating the tree.

        An unreadable manifest **stops** rather than being replaced. Replacing it would
        discard the only description of the vectors already in the tree, and the Consumer
        has no other way to learn which model produced them.
        """
        path = self.manifest_path
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SinkInvariantError(
                f"{path} exists but cannot be read ({exc}); nothing has been embedded.\n"
                "  It describes the vectors already in this tree, so it is not safe to replace. Repair it, or embed into a fresh --out directory."
            ) from None
        if not isinstance(manifest, dict):
            raise SinkInvariantError(f"{path} is not a JSON object; nothing has been embedded. Repair it, or embed into a fresh --out directory.")
        return manifest

    def _compare_invariants(self, manifest: dict[str, Any]) -> None:
        """Report **every** disagreeing fact with **both** values, then stop.

        Every fact, not the first: a tree built with a different model usually differs in
        several at once (`model_id`, both Instructions, often `native_dim`), and reporting
        one at a time turns one operator decision into four failed runs.
        """
        current = dataclasses.asdict(self.invariants)
        diffs: list[tuple[str, Any, Any]] = []
        for field, value in current.items():
            if field not in manifest:
                diffs.append((field, "<absent>", value))
            elif manifest[field] != value:
                diffs.append((field, manifest[field], value))
        if not diffs:
            return

        lines = [f"  {field}: this tree has {was!r}, this Invocation has {now!r}" for field, was, now in diffs]
        if any(field == "layout" for field, _was, _now in diffs):
            # The layout guard's fix is not "change a flag back" but "pass the same set of
            # Runs", which is not obvious from the two values alone.
            lines.append("      layout is 'bare' for a single --run and 'labelled' for two or more; pass the same run set, or use a fresh --out directory.")
        raise SinkInvariantError(
            f"{self.manifest_path} was written by a different Invocation: {len(diffs)} invariant fact(s) disagree; nothing has been embedded.\n"
            + "\n".join(lines)
            + "\n  Re-run with the settings that built this tree, or embed into a fresh --out directory."
        )

    def _warn_if_sinks_changed(self, previous: list[str]) -> None:
        """Say the expensive thing out loud before it happens (design 8.2).

        Only an *added* Sink costs GPU time: a Document is done when every enabled Sink
        holds it, so a Sink that holds nothing empties the Resume intersection and the
        whole corpus is re-embedded. Dropping a Sink only makes more Documents count as
        done, so it is recorded and not warned about.
        """
        added = [s for s in self.sinks if s not in previous]
        removed = [s for s in previous if s not in self.sinks]
        if added:
            logger.warning(
                "enabled Sinks changed: this tree was built with %s, this Invocation enables %s. "
                "A newly enabled Sink holds no Documents, so Resume's intersection is empty and every Document will be re-embedded.",
                previous or ["none recorded"],
                self.sinks,
            )
        if removed and not added:
            logger.info("enabled Sinks changed: this tree was built with %s, this Invocation enables %s.", previous, self.sinks)

    # -- Documents --------------------------------------------------------------

    def _document_path(self, run_label: str, document_name: str) -> Path:
        """`<out>/<name>` in the `bare` layout, `<out>/<label>/<name>` in `labelled`.

        The `.npz` and `.json` suffixes are appended to this, never substituted into it:
        `case.pdf` and `case.tiff` must land on different files (issue #32).
        """
        base = self.out / run_label if self.invariants.layout == "labelled" else self.out
        return base.joinpath(*document_name.split("/"))

    def write(self, doc: EmbeddedDocument) -> None:
        """Sidecar, then the `.npz`. Both atomic, in that order (design 8.5)."""
        import numpy as np

        if doc.document_name.split("/", 1)[0].startswith(RESERVED_PREFIX):
            # `names.check_collisions` already refuses this at startup, so reaching here
            # means the derivation and the Sink disagree about what is reserved. Refusing
            # again costs one string comparison and is the difference between a loud stop
            # and an Invocation that overwrites its own manifest halfway through.
            raise NameCollisionError(
                f"Document {doc.document_name!r} would write over the reserved {RESERVED_PREFIX} prefix at the output root "
                f"({MANIFEST_NAME} is the manifest and {FAILURES_NAME} is the failures list); nothing was written for it."
            )

        arrays = self._arrays(doc)
        path = self._document_path(doc.run_label, doc.document_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        _write_json(
            path.with_name(path.name + ".json"),
            {
                "source_file": doc.source_file,
                "source_digest": doc.source_digest,
                "run_label": doc.run_label,
                "created": _utc_iso(doc.created),
            },
        )
        # `np.savez` appends `.npz` to a *string* destination that does not already end in
        # it, which would turn every temporary into `<name>.npz.<rand>.tmp.npz` and leave
        # the real filename empty forever. Handing it an open file object is the documented
        # way to keep the name the caller chose.
        _atomic_write(path.with_name(path.name + ".npz"), lambda handle: np.savez(handle, **arrays))

    def _arrays(self, doc: EmbeddedDocument) -> dict[str, Any]:
        """The eight arrays of design 8.4, built and checked before anything is opened.

        Building them first is deliberate: a shape contradiction here would otherwise be
        discovered *between* the two writes, which is exactly the state the write ordering
        exists to make impossible.

        `int32` throughout. It caps at 2.1 billion characters against a 218k-character
        largest Document in the smoke sample, and it *says* "small count" where `int64`
        says nothing. `np.fromiter` raises `OverflowError` on a value that does not fit
        rather than wrapping it into a negative offset -- checked, not assumed.
        """
        import numpy as np

        chunks = doc.chunks
        n = len(chunks)
        stored_dim = self.invariants.stored_dim
        chunk_vectors = np.asarray(doc.chunk_vectors)
        document_vector = np.asarray(doc.document_vector)

        if chunk_vectors.ndim != 2 or chunk_vectors.shape[0] != n:
            raise ValueError(f"{doc.document_name!r}: {n} Chunks but chunk_vectors has shape {chunk_vectors.shape}")

        if n == 0:
            # The empty-Document layout (design 11.4). Every array is length zero and
            # `chunk_vectors` keeps its width, so a reader distinguishes an empty Document
            # in one line -- `z["document_vector"].size == 0` -- without a special case for
            # the shape. A zero *vector* was rejected: it is not a unit vector, nothing
            # else in the store is anything but a unit vector, and it would sit in a search
            # index looking like data.
            if document_vector.size:
                raise ValueError(f"{doc.document_name!r}: no Chunks but document_vector has shape {document_vector.shape}")
            chunk_vectors = np.zeros((0, stored_dim), dtype=np.float32)
            document_vector = np.zeros((0,), dtype=np.float32)
        else:
            # `stored_dim` is a manifest fact about the *whole tree*, so a file of another
            # width is not a bad Document but a broken promise to every Consumer reading
            # the manifest.
            if chunk_vectors.shape[1] != stored_dim:
                raise ValueError(f"{doc.document_name!r}: chunk_vectors is {chunk_vectors.shape[1]} wide, manifest stored_dim is {stored_dim}")
            if document_vector.shape != (stored_dim,):
                raise ValueError(f"{doc.document_name!r}: document_vector has shape {document_vector.shape}, expected ({stored_dim},)")
            chunk_vectors = chunk_vectors.astype(np.float32, copy=False)
            document_vector = document_vector.astype(np.float32, copy=False)

        return {
            "chunk_vectors": chunk_vectors,
            "document_vector": document_vector,
            "start_char": np.fromiter((c.start_char for c in chunks), dtype=np.int32, count=n),
            "end_char": np.fromiter((c.end_char for c in chunks), dtype=np.int32, count=n),
            "first_page": np.fromiter((c.first_page for c in chunks), dtype=np.int32, count=n),
            "last_page": np.fromiter((c.last_page for c in chunks), dtype=np.int32, count=n),
            "token_count": np.fromiter((c.token_count for c in chunks), dtype=np.int32, count=n),
            "is_partial_page": np.fromiter((c.is_partial_page for c in chunks), dtype=np.bool_, count=n),
        }

    # -- Resume -----------------------------------------------------------------

    def known(self) -> set[tuple[str, str]]:
        """`{(run_label, document_name)}` for every Document this tree already holds.

        Derived from the outputs, never from a recorded list (design 11.1): a manifest of
        names is a second source of truth that a crash can desynchronise from the first,
        and the measurement that was supposed to justify it -- `os.walk` over 20,000
        Documents in 29 ms against 3 ms to read a 1.3 MB JSON of the same names -- buys 26
        milliseconds once per Invocation.

        The `.npz` alone is the completion marker, which the write ordering makes sound.
        The `run_label` half of the key comes from the sidecar rather than the path,
        because the `bare` layout does not carry it: one tree can accumulate Documents from
        several single-Run Invocations under different labels, so there is no single label
        to assume. That costs one `open()` per Document on this path -- the same cost
        design 8.6 already accepts on the Consumer's read path, and the price of the
        sidecar being the readable record of identity.

        A missing or unreadable sidecar means the pair was broken by something other than
        this Sink. The Document is reported as *not* known, so it is re-embedded and both
        files are rewritten -- the same self-healing the two-Sink intersection relies on.
        """
        found: set[tuple[str, str]] = set()
        if not self.out.is_dir():
            return found
        labelled = self.invariants.layout == "labelled"
        for dirpath, _dirnames, filenames in os.walk(self.out):
            for filename in filenames:
                if not filename.endswith(".npz"):
                    continue
                npz = Path(dirpath) / filename
                parts = npz.relative_to(self.out).parts
                if labelled:
                    if len(parts) < 2:
                        # A stray `.npz` at the root of a labelled tree names no Document:
                        # every Document there sits under its Run's directory.
                        continue
                    parts = parts[1:]
                run_label = self._sidecar_run_label(npz)
                if run_label is None:
                    continue
                found.add((run_label, "/".join(parts)[: -len(".npz")]))
        return found

    def _sidecar_run_label(self, npz: Path) -> str | None:
        sidecar = npz.with_suffix(".json")
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            run_label = payload["run_label"]
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("%s has no readable sidecar; treating its Document as not embedded and re-embedding it", npz)
            return None
        if not isinstance(run_label, str) or not run_label:
            logger.warning("%s records no run_label; treating its Document as not embedded and re-embedding it", sidecar)
            return None
        return run_label

    # -- end of Invocation ------------------------------------------------------

    def write_failures(self, names: list[str]) -> None:
        """One Document name per line at `<out>/paperscale-embed-failures.txt`.

        A convenience, not state: Resume derives from the outputs, so a failed Document
        simply has no output and is retried next Invocation whether or not this file
        exists. It shares the reserved prefix, so the collision guard already covers it.

        An empty list **removes** a stale file rather than leaving it. The file describes
        one Invocation, and one left behind by the previous run would name Documents this
        run embedded successfully -- an operator reading it has no way to tell.
        """
        path = self.out / FAILURES_NAME
        if not names:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return
        self.out.mkdir(parents=True, exist_ok=True)
        body = "".join(f"{name}\n" for name in names)
        _atomic_write(path, lambda handle: handle.write(body.encode("utf-8")))


__all__ = ["CHUNKER", "FAILURES_NAME", "MANIFEST_NAME", "POOLING", "Invariants", "NpzSink", "SinkInvariantError"]
