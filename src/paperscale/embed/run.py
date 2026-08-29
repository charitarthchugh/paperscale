"""The `paperscale embed` orchestrator: startup order, the two stages, the report.

Everything else in this package is a piece that can be tested alone. This module is
the only place that knows what order the pieces run in, and the order is the design's
(12.1), not an implementation convenience:

* Three steps can stop the Invocation **before any GPU work** -- the collision check,
  the invariant comparison, and the native-width probe. Each of them guards against a
  failure whose symptom is a Sink that looks perfect and is wrong, so each has to run
  before the first Document is embedded rather than after the first is written.
* `/v1/models` is asked **before the reporter exists**, because the panel header shows
  the id the *server* returned. The width probe cannot cover this: it compares a
  dimension, and two models of equal width are indistinguishable by it (design 3.6).
* Resume state is derived **once**, so the progress bar's total can be `corpus -
  skipped`. That is the mechanism, not a cosmetic: point `embed` at a stale output
  directory and the bar reads `0/0` (design 13.5).

The run itself is two stages over one event loop. A tokenize/chunk stage turns each
Document into Chunks; a packing stage fills `/v1/embeddings` requests from a token
budget, **mixing Documents**, and writes each Document out once all of its Chunk
vectors are back. Both Sinks are written from the event-loop thread and never from a
worker, which is what "single writer" means here -- LanceDB batches at 64 and the
`.npz` pair is two creates and two renames, neither of which is safe to interleave.

The failure taxonomy is design 12.6/17.1 and the dispositions genuinely differ:
a `/tokenize` failure or an exhausted response axis fails **the Document**; a
`ServerGoneError` is terminal for **the Invocation** and propagates, because a dead
server would otherwise burn through the corpus at six connection attempts each,
marking every Document failed -- and a Document recorded failed is one Resume retries,
so the damage would outlive the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["run_embed"]

# Design 14.4. The check lives here and deliberately **not** in `cli._parse_runs`:
# that function is shared with `evaluate`, where a label goes into a SQLite column and
# has never needed a constraint. Tightening it globally would start rejecting inputs
# `evaluate` accepts today, which is a worse trade than one duplicated regex. Here the
# label becomes a directory name in the `labelled` layout, so `--run 'legal/2024=...'`
# silently creates a nested tree that nothing else in the design knows about.
# Sanitizing was rejected: it would invent a second name-mangling rule with its own
# collision question, next to the one `names.py` already owns.
_LABEL_RE = re.compile(r"[A-Za-z0-9._-]+")

# The two Sink names as they appear in the manifest's `sinks` list and in
# `resume.sink_set_warning`'s flag rendering. Spelled once so a typo cannot make the
# next Invocation think a Sink was added.
SINK_NPZ = "npz"
SINK_LANCEDB = "lancedb"

# The width probe's input. Content is irrelevant -- only `.shape[1]` is read -- so it
# is one short ASCII word: the request is paid for on every Invocation and a long one
# would buy nothing.
_PROBE_TEXT = "paperscale"


@dataclasses.dataclass(frozen=True)
class _Document:
    """One Record, reduced to what the two stages need.

    The whole corpus is held in memory between step 3 (which must read every Record to
    derive names and check collisions) and step 11 (which needs the Resume set before
    it can know what to drop). Documents the Resume set covers become garbage as soon
    as `todo` is built, so the peak is one pass over the corpus text; a corpus large
    enough for that to hurt wants a second pass over the JSONL instead, which is a
    change here and nowhere else.
    """

    run_label: str
    document_name: str
    source_file: str
    text: str
    spans: list[list[int]]

    @property
    def key(self) -> tuple[str, str]:
        return (self.run_label, self.document_name)


@dataclasses.dataclass(frozen=True)
class _ChunkRef:
    """One Chunk waiting for a vector, with the text that will be sent for it.

    The packing unit is the Chunk and not the Document, so a Document whose Chunks
    together exceed the request budget is spread over several requests rather than
    sent as one oversized body. `document_index` is what makes split-on-failure
    meaningful: the Documents of a failed request are exactly the distinct
    `document_index` values in it.

    It is a **position** -- the Document's place in `todo`, from `enumerate` -- and not
    a Document name. Nothing outside this module ever sees it: the Sinks are keyed by
    `(run_label, document_name)`, and this index exists only because a dict keyed by an
    integer is what lets a Chunk vector find its Document while the request is in
    flight.
    """

    document_index: int
    #: The Chunk's position within its own Document, which is what orders `vectors`.
    index: int
    text: str
    tokens: int


class _DocState:
    """A Document's Chunk vectors as they arrive, and whether it has already failed.

    Nothing is written until `pending` reaches zero, so a Document that fails halfway
    leaves no partial output -- which is what lets Resume treat the presence of an
    output as proof of completion (design 11.1).
    """

    def __init__(self, doc: _Document, chunks: list) -> None:
        self.doc = doc
        self.chunks = chunks
        self.vectors: list[Any] = [None] * len(chunks)
        self.pending = len(chunks)
        self.failed = False


@dataclasses.dataclass
class _Counts:
    """The end-of-run report's counts by outcome (design 12.7).

    `embedded` and `empty` are disjoint: a zero-Chunk Document is a recorded outcome
    and not a failure (design 5.5/11.4), but it is also not a vector, so folding it
    into `embedded` would make the headline number describe two different things.
    `oversize` is a *subset* of `failed` -- a context overflow did fail its Document --
    and is counted apart because it means the chunker or the context length is wrong,
    which no amount of corpus is responsible for.
    """

    embedded: int = 0
    empty: int = 0
    skipped: int = 0
    failed: int = 0
    oversize: int = 0
    chunks: int = 0


def run_embed(args) -> int:
    """Entry point for `paperscale embed`. Returns the process exit code.

    Steps 1-3 of design 12.1 run synchronously here because none of them touches the
    server and all three can stop the Invocation; everything from `/v1/models` on runs
    under one event loop in `_embed_invocation`.
    """
    from paperscale.cli import _parse_runs
    from paperscale.embed.adapters import build_embed_model, validate_embed_dim

    # -- step 1: flags -------------------------------------------------------
    runs = _parse_runs(args.run)
    _validate_run_labels(runs)
    if args.no_npz and not args.lancedb:
        raise SystemExit("--no-npz needs --lancedb: at least one Sink must be live, or the Invocation would spend the GPU time and write no vectors anywhere.")

    # -- step 2: the Adapter -------------------------------------------------
    try:
        adapter = build_embed_model(args.embed_model)
    except ValueError as exc:
        # `build_embed_model` raises ValueError to mirror `build_ocr_model` verbatim,
        # and `--embed-model` carries no argparse `choices=`. Converting here is what
        # turns a traceback into the one-line message the operator expects from a CLI.
        raise SystemExit(str(exc)) from None
    stored_dim = validate_embed_dim(adapter, args.embed_dim)

    # -- step 3: Records, names, collisions ----------------------------------
    corpus = _load_corpus(runs)

    return asyncio.run(_embed_invocation(args, runs=runs, adapter=adapter, stored_dim=stored_dim, corpus=corpus))


def _validate_run_labels(runs: list[tuple[str, str]]) -> None:
    """Reject a label that cannot be a directory name, naming the character (design 14.4)."""
    for label, _path in runs:
        if _LABEL_RE.fullmatch(label):
            continue
        offender = next(ch for ch in label if not _LABEL_RE.fullmatch(ch))
        raise SystemExit(
            f"--run label {label!r} contains {offender!r}, which embed does not allow: labels are limited to [A-Za-z0-9._-] "
            "because the label becomes a directory name in the output tree. Rename the run."
        )


def _load_corpus(runs: list[tuple[str, str]]) -> list[_Document]:
    """Read every Record, derive every Document name, and refuse a collision (design 12.1 step 3).

    The collision check is scoped to one Run, which is what makes two Runs holding the
    same PDF legal: they are kept apart by the `run_label` half of the key, not by the
    name. Within one Run a collision is fatal here rather than an overwrite later --
    the two Documents would otherwise take turns writing one output and whichever
    finished last would silently be the one a Consumer reads.
    """
    from paperscale.embed.names import NameCollisionError, check_collisions, document_name
    from paperscale.embed.records import DuplicateSourceFileError, iter_records

    corpus: list[_Document] = []
    for label, path in runs:
        pairs: list[tuple[str, str]] = []
        found: list[_Document] = []
        try:
            for record in iter_records(path):
                metadata = record.get("metadata") or {}
                source_file = metadata.get("Source-File") or ""
                name = document_name(source_file)
                pairs.append((name, source_file))
                spans = (record.get("attributes") or {}).get("pdf_page_numbers") or []
                found.append(_Document(label, name, source_file, record.get("text") or "", spans))
        except FileNotFoundError as exc:
            raise SystemExit(f"--run {label}: {exc}") from None
        except DuplicateSourceFileError as exc:
            raise SystemExit(str(exc)) from None
        try:
            check_collisions(label, pairs)
        except NameCollisionError as exc:
            raise SystemExit(str(exc)) from None
        if not found:
            # `resolve_jsonl_paths` answers a directory holding no `.jsonl` with an
            # empty list rather than an error, so a typo'd `--run` path that happens to
            # be a real directory is otherwise indistinguishable from a fully-resumed
            # Invocation: both leave the bar reading 0/0.
            logger.warning("--run %s=%s contributed no Records; check the path", label, path)
        corpus.extend(found)
    return corpus


async def _embed_invocation(args, *, runs: list[tuple[str, str]], adapter, stored_dim: int, corpus: list[_Document]) -> int:
    """Design 12.1 steps 4-12, then the end-of-run report and the exit code."""
    from paperscale.embed.budget import chunk_budget, request_budget, resolve_context_length
    from paperscale.embed.client import EmbedClient, ServerGoneError
    from paperscale.embed.lance_sink import LanceSink
    from paperscale.embed.invariants import Invariants, SinkInvariantError
    from paperscale.embed.npz_sink import CHUNKER, FAILURES_NAME, POOLING, NpzSink
    from paperscale.embed.resume import KnownSink, LayoutChangeError, check_layout, derive_resume_state, invocation_layout, sink_set_warning
    from paperscale.tui import NullReporter, install_tui_logging, make_reporter, open_log_file, restore_console_logging
    from paperscale.vllm_stats import VLLMStats, VLLMStatsPoller, metrics_url

    client = EmbedClient(
        args.embed_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        max_request_retries=args.max_request_retries,
    )

    # -- step 4: what the server actually serves ----------------------------
    served_model_id, server_max_model_len = await client.models()

    # -- steps 5-7: the three token numbers ---------------------------------
    validated_context_length = resolve_context_length(
        card_context_length=adapter.card_context_length,
        server_max_model_len=server_max_model_len,
        override=args.context_length,
    )
    # An exact count, never an estimate: the Instruction is prepended to every Chunk,
    # so an under-count here shows up as a context overflow on the longest Chunk of
    # some Document hours into the run.
    instruction_tokens = await client.tokenize(adapter.document_instruction) if adapter.document_instruction else 0
    chunk_budget_tokens = chunk_budget(validated_context_length, instruction_tokens)
    request_budget_tokens = request_budget(args.request_tokens, chunk_budget_tokens)

    # -- step 8: the width probe --------------------------------------------
    await _probe_native_dim(client, adapter)

    # -- step 9: the Sinks ---------------------------------------------------
    layout = invocation_layout(len(runs))
    invariants = Invariants(
        model_id=served_model_id,
        stored_dim=stored_dim,
        native_dim=adapter.native_dim,
        document_instruction=adapter.document_instruction,
        query_instruction=adapter.query_instruction,
        pooling=POOLING,
        chunker=CHUNKER,
        chunk_budget_tokens=chunk_budget_tokens,
        layout=layout,
    )
    enabled = ([] if args.no_npz else [SINK_NPZ]) + ([SINK_LANCEDB] if args.lancedb else [])
    # Built and opened even under `--no-npz` (design 17.2 item 1): `<out>` exists in
    # that case anyway because the failures file lives there, and the manifest is the
    # only place the enabled-Sink set is recorded for the next Invocation to compare.
    npz = NpzSink(Path(args.out), invariants, enabled)
    sink_warning = None
    try:
        recorded_layout = _recorded_layout(npz.manifest_path)
        if recorded_layout is not None:
            # Ahead of `open()`'s generic nine-fact comparison on purpose. A layout
            # change is the one difference whose cost is not mixing but silent
            # *duplication*, and `check_layout` is the message that says so and names
            # the fix (pass the same run set), which two values alone do not.
            check_layout(out=args.out, recorded=recorded_layout, current=layout)
        npz.open()
        if npz.previous_sinks is not None:
            sink_warning = sink_set_warning(npz.previous_sinks, enabled, corpus_size=len(corpus))
        lance = None
        if args.lancedb:
            lance = LanceSink(Path(args.lancedb), invariants)
            lance.open()
    except (SinkInvariantError, LayoutChangeError) as exc:
        # Both messages are multi-line and already end with the instruction; wrapping
        # them in another sentence would only push the fix further from the eye.
        raise SystemExit(str(exc)) from None

    # -- step 10: the reporter, now that the served id is known --------------
    # The separator is escaped rather than typed so this file stays ASCII; it renders
    # as the middot `pipeline.py` already uses in the OCR reporter's title.
    rep = make_reporter(args.tui, title=f"paperscale embed \u00b7 {served_model_id}")
    live = not isinstance(rep, NullReporter)

    stats = poller = tui_handler = None
    if live:
        # The same dance as `_handle_evaluate`: `install_tui_logging` strips every
        # handler off the root logger, so a file handler attached before the call
        # would be taken straight back off. Handing it the path is what keeps it
        # attached. embed has no workspace, so the default sits beside `--out` --
        # beside and not inside, because `--out` is the deliverable a Consumer reads.
        args.disk_logging = args.disk_logging or _embed_log_path(Path(args.out))
        tui_handler = install_tui_logging(rep, args.disk_logging)
        stats = VLLMStats()
        poller = VLLMStatsPoller(metrics_url(args.embed_url), stats, interval=args.tui_poll_interval)
        poller.start()
    elif args.disk_logging:
        logging.getLogger().addHandler(open_log_file(args.disk_logging))

    counts = _Counts()
    failures: list[str] = []
    server_gone: Exception | None = None
    try:
        with rep:
            if sink_warning:
                rep.log(sink_warning)

            # -- step 11: Resume ---------------------------------------------
            # `--no-npz` leaves the `.npz` Sink out of the intersection: it holds
            # nothing this Invocation writes, so counting it would mark the whole
            # corpus undone on every run.
            resume_sinks: list[KnownSink] = []
            if not args.no_npz:
                resume_sinks.append(npz)
            if lance is not None:
                resume_sinks.append(lance)
            done = derive_resume_state(resume_sinks, no_resume=args.no_resume)
            todo = [doc for doc in corpus if doc.key not in done]
            counts.skipped = len(corpus) - len(todo)
            # `is_new` comes from this Sink's own view and never from the Resume
            # intersection: under `--no-resume` the intersection is empty, so every
            # Document would arrive marked new and `add()` would append a second copy
            # of rows the tables already hold (design 11.5).
            lance_known = lance.known() if lance is not None else set()

            rep.set_stat("documents", 0)
            rep.set_stat("chunks", 0)
            rep.set_stat("skipped", counts.skipped)
            rep.set_stat("empty", 0)
            # Seeded before the first Document so the rows exist from the first frame:
            # a column that grows a new row mid-run reads as something breaking.
            rep.set_stat("failed", 0, group="issues")
            rep.set_stat("oversize", 0, group="issues")
            rep.set_stat("retrying", 0, group="issues")
            rep.log(f"embedding {len(todo)} of {len(corpus)} Document(s); {counts.skipped} already held by every enabled Sink.")

            # -- step 12: run -------------------------------------------------
            # The total is Documents that will actually be embedded, so every unit is
            # work performed and a fully-resumed Invocation reads 0/0 (design 13.5).
            phase = rep.phase("embedding", total=len(todo))
            embedder = _Embedder(
                client=client,
                adapter=adapter,
                todo=todo,
                chunk_budget_tokens=chunk_budget_tokens,
                request_budget_tokens=request_budget_tokens,
                stored_dim=stored_dim,
                npz=None if args.no_npz else npz,
                lance=lance,
                lance_known=lance_known,
                rep=rep,
                phase=phase,
                counts=counts,
                failures=failures,
                stats=stats,
                poller=poller,
                poll_interval=args.tui_poll_interval,
            )
            try:
                await embedder.run()
            except ServerGoneError as exc:
                # Caught here rather than left to propagate out of `run_embed` so the
                # finally below still flushes the LanceDB buffer and writes the
                # failures file: the Documents already embedded are not in doubt.
                server_gone = exc
            phase.done()
    finally:
        if lance is not None:
            lance.close()
        npz.write_failures(sorted(failures))
        if poller is not None:
            poller.stop()
        if tui_handler is not None:
            restore_console_logging(tui_handler)

    _report(counts, out=Path(args.out), failures_name=FAILURES_NAME)
    if server_gone is not None:
        print(str(server_gone), file=sys.stderr)
    # The alternate screen is gone by now, so this lands on the real screen -- without
    # it a --tui run leaves no trace of where its log went.
    if live and args.disk_logging:
        print(f"Full log: {args.disk_logging}", file=sys.stderr)
    return 1 if (counts.failed or server_gone is not None) else 0


async def _probe_native_dim(client, adapter) -> None:
    """One cheap `/v1/embeddings` call, to prove the server serves the model named (design 12.1 step 8).

    This is not a nicety. Point `--embed-model` at a server with a *different size* of
    the same family loaded and, with every vector sliced to `--embed-dim`, the stored
    data carries no trace of the substitution: it slices, re-normalizes and pools like
    any other, both Sinks look perfect, and Resume will not catch it either because it
    only ever asks "do I know this name?".

    The probe runs with the client's `native_dim` still unset so the first response
    latches its own width and this comparison can report **both** numbers. Assigning
    the Adapter's value afterwards is what arms the client's per-payload backstop for
    the rest of the Invocation.
    """
    from paperscale.embed.client import EmbedRequestError, TerminalDocumentError

    try:
        probe = await client.embed([_PROBE_TEXT])
    except (EmbedRequestError, TerminalDocumentError) as exc:
        raise SystemExit(f"embed: the startup dimension probe failed against {client.url}: {exc}") from None
    observed = int(probe.shape[1])
    if observed != adapter.native_dim:
        raise SystemExit(
            f"embed: {client.url} serves vectors {observed} wide, but --embed-model {type(adapter).__name__} is {adapter.native_dim} wide.\n"
            "  That is a different model from the one named. Nothing has been embedded: vectors from two models are not comparable, "
            "and slicing hides the substitution completely."
        )
    client.native_dim = adapter.native_dim


def _recorded_layout(manifest_path: Path) -> str | None:
    """The `layout` the manifest records, or None when there is nothing to compare against.

    Deliberately forgiving: an unreadable or malformed manifest is `NpzSink.open()`'s
    to refuse, with a message about the tree it describes. Raising a second, worse one
    from here would only get in front of it.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    layout = manifest.get("layout")
    return layout if isinstance(layout, str) else None


def _embed_log_path(out: Path) -> str:
    """Where embed's logs go when the dashboard owns the screen.

    Beside `--out` and not inside it. `evaluate` puts its log next to `--db`; the
    analogue here is a sibling of the output tree, because `<out>` is the deliverable
    a Consumer opens and a `logs/` directory appearing inside it is one more thing
    every reader has to learn to ignore.
    """
    return str(out.resolve().parent / "logs" / f"embed-{os.getpid()}.log")


def _report(counts: _Counts, *, out: Path, failures_name: str) -> None:
    """Counts by outcome, on the real screen, after the frame is gone (design 12.7)."""
    print(f"embed: {counts.embedded} embedded, {counts.empty} empty, {counts.skipped} skipped, {counts.failed} failed ({counts.chunks} chunks).")
    if counts.oversize:
        print(
            f"  {counts.oversize} of the failures were context overflows. Chunks are sized from an untruncated token count "
            "precisely so that cannot happen, so this is a chunker or context-length bug, not a corpus problem."
        )
    if counts.failed:
        print(f"  Failed Document names: {out / failures_name}")


class _Embedder:
    """The two stages, the request packer, and the single writer.

    Stage one turns Documents into Chunks (`/tokenize`, bounded by the client's own
    tokenize semaphore). Stage two packs Chunks into `/v1/embeddings` requests by a
    **token budget, never a count of Chunks** -- greedy page packing means one Chunk
    may be a short page and the next forty-five dense ones, so "16 Chunks" is anywhere
    from a few hundred to half a million prefill tokens and an operator cannot reason
    about it. Every Chunk's exact count is already known before packing, so the budget
    is free.

    **Requests mix Documents, and must**: the common case is a single Chunk of a few
    thousand tokens, so refusing to mix would make almost every request tiny. The
    obvious mental model of why that helps is wrong and worth stating -- a request is
    *not* a batch the server processes together. vLLM fans an `input` array of N texts
    into N independent engine requests and merges them as they finish. So request
    batching amortizes HTTP round trips only; concurrency, not batch size, fills the
    engine; and split-on-failure costs more round trips and **identical** engine work,
    which is why it is affordable at all.
    """

    def __init__(
        self,
        *,
        client,
        adapter,
        todo: list[_Document],
        chunk_budget_tokens: int,
        request_budget_tokens: int,
        stored_dim: int,
        npz,
        lance,
        lance_known: set[tuple[str, str]],
        rep,
        phase,
        counts: _Counts,
        failures: list[str],
        stats,
        poller,
        poll_interval: float,
    ) -> None:
        self.client = client
        self.adapter = adapter
        self.todo = todo
        self.chunk_budget_tokens = chunk_budget_tokens
        self.request_budget_tokens = request_budget_tokens
        self.stored_dim = stored_dim
        self.npz = npz
        self.lance = lance
        self.lance_known = lance_known
        self.rep = rep
        self.phase = phase
        self.counts = counts
        self.failures = failures
        self.stats = stats
        self.poller = poller
        self.poll_interval = max(0.5, poll_interval)
        self.states: dict[int, _DocState] = {}
        self._queue_history: list[tuple[float, float | None]] = []
        self._fatal: BaseException | None = None
        self._stop = asyncio.Event()
        self._doc_q: asyncio.Queue = asyncio.Queue()
        for document_index, doc in enumerate(todo):
            self._doc_q.put_nowait((document_index, doc))
        # Bounded, so the chunking stage cannot read the whole corpus into Chunks while
        # the GPU is the thing running behind. The packer is the only consumer, so one
        # slot per request in flight is the natural depth.
        self._prepared_q: asyncio.Queue = asyncio.Queue(maxsize=max(1, client.concurrency))
        self._send_slots = asyncio.Semaphore(client.concurrency)
        self._in_flight: set[asyncio.Task] = set()

    async def run(self) -> None:
        """Drive both stages to completion, then re-raise a fatal server failure.

        The fatal path is a stored exception rather than one that escapes a worker:
        cancelling mid-flight would abandon Documents whose vectors are already back
        and whose writes are one call away, and a `ServerGoneError` says nothing about
        the Sinks.
        """
        n_workers = max(1, min(self.client.tokenize_concurrency, len(self.todo) or 1))
        workers = [asyncio.create_task(self._prepare_worker()) for _ in range(n_workers)]
        packer = asyncio.create_task(self._packer(n_workers))
        # No scraper, no ticker. Without `--tui` there is no `VLLMStats` to read and
        # `set_stat` is a no-op, so a five-second wakeup would only exist to do nothing.
        ticker = asyncio.create_task(self._tick()) if self.poller is not None else None
        try:
            await asyncio.gather(*workers)
            await packer
        finally:
            if ticker is not None:
                ticker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ticker
        if self._fatal is not None:
            raise self._fatal

    # -- stage one: tokenize and chunk ---------------------------------------

    async def _prepare_worker(self) -> None:
        """Turn Documents into Chunks until the queue is empty, then post a sentinel.

        A `/tokenize` failure fails the **Document**, not the Invocation: without a
        token count a Chunk cannot be sized, and the whole design rests on vLLM
        erroring on overflow rather than truncating, so proceeding on a guess is
        unsafe rather than merely imprecise.
        """
        from paperscale.embed.chunking import chunk_document
        from paperscale.embed.client import EmbedRequestError, ServerGoneError, TerminalDocumentError

        try:
            while not self._stop.is_set():
                try:
                    document_index, doc = self._doc_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    chunks = await chunk_document(doc.text, doc.spans, self.chunk_budget_tokens, self.client.tokenize)
                except ServerGoneError as exc:
                    self._halt(exc)
                    break
                except (EmbedRequestError, TerminalDocumentError) as exc:
                    self._record_failure(doc, exc)
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 -- one Document must never end the Invocation
                    logger.exception("embed: unexpected error while chunking %s", doc.document_name)
                    self._record_failure(doc, exc)
                    continue
                await self._prepared_q.put((document_index, doc, chunks))
        finally:
            # One sentinel per worker, on every path out. The packer counts them to
            # know the stage is over, so a worker that returned without posting one
            # would leave it waiting forever -- a hang, not a crash.
            await self._prepared_q.put(None)

    # -- stage two: pack, send, write ----------------------------------------

    async def _packer(self, n_workers: int) -> None:
        """Fill requests from the token budget, mixing Documents, and dispatch them."""
        batch: list[_ChunkRef] = []
        tokens = 0
        finished = 0
        while finished < n_workers:
            item = await self._prepared_q.get()
            if item is None:
                finished += 1
                continue
            if self._stop.is_set():
                # Keep draining rather than break: a worker blocked on `put` would
                # never reach its sentinel and this loop would never end.
                continue
            document_index, doc, chunks = item
            if not chunks:
                # Nothing to send. Recorded as an outcome and written to both Sinks --
                # an empty output is how a Document with no usable text is distinguished
                # from one that was never reached (design 11.4).
                self.states[document_index] = _DocState(doc, chunks)
                self._complete(document_index)
                continue
            self.states[document_index] = _DocState(doc, chunks)
            for index, chunk in enumerate(chunks):
                ref = _ChunkRef(document_index, index, doc.text[chunk.start_char : chunk.end_char], chunk.token_count)
                if batch and tokens + ref.tokens > self.request_budget_tokens:
                    await self._dispatch(batch)
                    batch, tokens = [], 0
                batch.append(ref)
                tokens += ref.tokens
        if batch and not self._stop.is_set():
            await self._dispatch(batch)
        if self._in_flight:
            await asyncio.gather(*list(self._in_flight), return_exceptions=True)

    async def _dispatch(self, refs: list[_ChunkRef]) -> None:
        """Start one request, blocking until a concurrency slot frees.

        Awaiting the slot here rather than queuing tasks behind the client's own
        semaphore is what carries backpressure all the way up: the packer stalls, the
        prepared queue fills, and the chunking stage stops reading ahead.
        """
        await self._send_slots.acquire()
        task = asyncio.create_task(self._send(refs))
        self._in_flight.add(task)
        task.add_done_callback(self._request_finished)

    def _request_finished(self, task: asyncio.Task) -> None:
        self._in_flight.discard(task)
        self._send_slots.release()

    async def _send(self, refs: list[_ChunkRef]) -> None:
        """Run one request as a task, and let no exception escape it silently.

        A task's exception is only seen by whoever awaits it, and the packer's final
        `gather` runs with `return_exceptions=True`. Anything unexpected that got out
        of here would therefore vanish while its Documents sat at `pending > 0`
        forever -- a bar that stops short of its total and a run that reports nothing
        wrong. Failing those Documents is the honest answer.
        """
        from paperscale.embed.client import ServerGoneError

        try:
            await self._embed_refs(refs)
        except ServerGoneError as exc:
            self._halt(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- the disposition is the same for anything unforeseen
            logger.exception("embed: unexpected error while embedding %d chunk(s)", len(refs))
            for document_index in sorted({ref.document_index for ref in refs}):
                self._fail(document_index, exc)

    async def _embed_refs(self, refs: list[_ChunkRef]) -> None:
        """Embed one request's Chunks; on terminal failure, split before failing Documents.

        Because requests mix Documents, a single poison Document would otherwise take
        every Document sharing its request down with it -- one bad PDF failing forty.
        So a multi-Document request that fails terminally is re-issued **one Document
        at a time**, and only those that fail alone are recorded failed. The re-issues
        run sequentially: this is the rare path, and each recursive call takes the
        single-Document branch below, so it cannot split again.

        `ValueError` from `slice_and_normalize` joins the same path deliberately. It
        means a row came back with no usable norm, which belongs to one Chunk of one
        Document, and the split is exactly what isolates which.
        """
        from paperscale.embed.client import EmbedRequestError, TerminalDocumentError
        from paperscale.embed.vectors import slice_and_normalize

        texts = [self.adapter.document_instruction + ref.text for ref in refs]
        try:
            raw = await self.client.embed(texts)
            vectors = slice_and_normalize(raw, self.stored_dim)
        except (EmbedRequestError, TerminalDocumentError, ValueError) as exc:
            document_indexes = sorted({ref.document_index for ref in refs})
            if len(document_indexes) > 1:
                logger.warning(
                    "embed: a request carrying %d Documents failed (%s); re-issuing them one at a time so only the bad one fails",
                    len(document_indexes),
                    exc,
                )
                for document_index in document_indexes:
                    await self._embed_refs([ref for ref in refs if ref.document_index == document_index])
                return
            self._fail(document_indexes[0], exc)
            return
        for ref, row in zip(refs, vectors):
            state = self.states.get(ref.document_index)
            if state is None or state.failed:
                # Its sibling request already failed it. Dropping the vector is right:
                # nothing is written for a failed Document, so Resume retries it whole.
                continue
            state.vectors[ref.index] = row
            state.pending -= 1
            if state.pending == 0:
                self._complete(ref.document_index)

    # -- the single writer ---------------------------------------------------

    def _complete(self, document_index: int) -> None:
        """Pool, write both Sinks in order, and count the outcome.

        Called only from the event-loop thread, which is what makes "single writer"
        true for both Sinks: the `.npz` pair is two creates and two renames whose
        order is load-bearing, and LanceDB's batch of 64 is one buffer.
        """
        import numpy as np

        from paperscale.embed.names import NameCollisionError, source_digest
        from paperscale.embed.vectors import EmbeddedDocument, pool_document_vector

        state = self.states.pop(document_index)
        doc = state.doc
        if state.chunks:
            chunk_vectors = np.stack(state.vectors).astype(np.float32, copy=False)
        else:
            chunk_vectors = np.zeros((0, self.stored_dim), dtype=np.float32)
        document_vector = pool_document_vector(chunk_vectors, [chunk.token_count for chunk in state.chunks])
        embedded = EmbeddedDocument(
            document_name=doc.document_name,
            run_label=doc.run_label,
            source_file=doc.source_file,
            source_digest=source_digest(doc.source_file),
            created=datetime.datetime.now(datetime.timezone.utc),
            chunks=list(state.chunks),
            chunk_vectors=chunk_vectors,
            document_vector=document_vector,
        )
        try:
            if self.npz is not None:
                self.npz.write(embedded)
            if self.lance is not None:
                self.lance.write(embedded, is_new=doc.key not in self.lance_known)
        except (NameCollisionError, ValueError) as exc:
            # A per-Document contradiction (a reserved name that reached the Sink, a
            # width the manifest does not promise). The Invocation-level faults --
            # a disagreeing manifest, a wrong stored_dim for the whole tree -- were
            # already refused at step 9, before any GPU time was spent.
            self._record_failure(doc, exc)
            return

        if state.chunks:
            self.counts.embedded += 1
            self.counts.chunks += len(state.chunks)
            self.rep.set_stat("documents", self.counts.embedded)
            self.rep.set_stat("chunks", self.counts.chunks)
        else:
            self.counts.empty += 1
            self.rep.set_stat("empty", self.counts.empty)
        self.phase.advance()

    # -- failures ------------------------------------------------------------

    def _fail(self, document_index: int, exc: BaseException) -> None:
        state = self.states.get(document_index)
        if state is None or state.failed:
            return
        state.failed = True
        self.states.pop(document_index, None)
        self._record_failure(state.doc, exc)

    def _record_failure(self, doc: _Document, exc: BaseException) -> None:
        """Count one failed Document and advance the bar past it.

        The bar still advances: its total counted this Document, and a bar that never
        reaches its total reads as a hang. `oversize` is read off the exception rather
        than matched here -- only the client can tell a context-overflow 400 from any
        other 400, and it is the one that means the chunker is wrong.
        """
        self.counts.failed += 1
        self.failures.append(doc.document_name)
        self.rep.set_stat("failed", self.counts.failed, group="issues")
        if getattr(exc, "oversize", False):
            self.counts.oversize += 1
            self.rep.set_stat("oversize", self.counts.oversize, group="issues")
        logger.error("embed: %s failed: %s", doc.document_name, exc)
        self.rep.log(f"failed: {doc.document_name}")
        self.phase.advance()

    def _halt(self, exc: BaseException) -> None:
        """Record the first Invocation-terminal failure and let both stages wind down."""
        if self._fatal is None:
            self._fatal = exc
            logger.error("embed: %s", exc)
        self._stop.set()

    # -- the panel -----------------------------------------------------------

    async def _tick(self) -> None:
        """Refresh the `server` column on the poller's own cadence.

        Not on Document completion: a Document can take minutes, and a panel that only
        moves when one lands reads as a hung server rather than a busy one. The queue
        history is sampled here too, one entry per tick, which is what
        `queue_advisory`'s window expects.
        """
        from paperscale.embed.panel import push_embed_stats, queue_advisory

        while True:
            await asyncio.sleep(self.poll_interval)
            if self.stats is not None:
                self._queue_history.append((time.monotonic(), self.stats.rates().waiting))
                advisory = queue_advisory(self._queue_history)
                if advisory:
                    self.rep.log(advisory)
            # Design 13.2's gauge, sampled on the same cadence as the `server` column
            # because it is the same kind of number: true now, not true cumulatively.
            self.rep.set_stat("retrying", self.client.retrying, group="issues")
            push_embed_stats(
                self.rep,
                self.stats,
                self.poller,
                stored_dim=self.stored_dim,
                native_dim=self.adapter.native_dim,
                outstanding=self.client.outstanding,
            )
