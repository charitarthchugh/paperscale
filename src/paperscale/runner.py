"""Durable local-first document-to-Markdown OCR runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
import uuid
import urllib.request

from paperscale.async_pool import AdaptiveLimiter
from paperscale.assembly import MarkdownAssembler
from paperscale.contracts import CURRENT_SCHEMA_VERSION, PageArtifact, ensure_known_schema
from paperscale.profiles.builtin import get_builtin_profile
from paperscale.providers.base import PageOcrProvider
from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    ProviderCapacityProfile,
    ProviderOverloadController,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
)
from paperscale.resources import ResourceGovernor, ResourceKind
from paperscale.quality.verifier import DeterministicQualityVerifier
from paperscale.rendering import PdfPageRenderer, png_dark_fraction
from paperscale.scheduler import CompactIndexError
from paperscale.state.fs_store import FileSystemStateStore

Json = dict[str, Any]
RendererFactory = Callable[[Path, dict[str, Any]], Any]

# Document-claim lease/heartbeat (single-machine multi-process tier). Heartbeat is
# ~lease/3 so a brief stall does not lose the claim, per concurrency-and-queuing.md.
_CLAIM_LEASE_SECONDS = 60.0
_CLAIM_HEARTBEAT_SECONDS = 20.0

# A page whose rendered image has less ink than this is treated as genuinely blank:
# an empty OCR result for it is accepted as a successful empty page rather than a
# failure. Blank scans sit ~0.002-0.005; content pages are an order of magnitude more.
_BLANK_INK_THRESHOLD = 0.01

# Diagnostics that mean "no readable content was produced" — on a near-blank render
# these indicate a genuinely blank page (empty output, or a VLM repetition loop on
# noise), so the page is accepted as blank rather than failed.
_BLANK_ELIGIBLE_DIAGNOSTICS = frozenset({"empty_output", "repeated_ngram", "repeated_character"})


def _render_is_blank(image_bytes: bytes) -> bool:
    try:
        return png_dark_fraction(image_bytes) < _BLANK_INK_THRESHOLD
    except Exception:  # noqa: BLE001 - never let a blank-check error fail a page
        return False


@dataclass(frozen=True, slots=True)
class _AttemptResult:
    """Outcome of one provider attempt within a page's retry/remediation ladder."""

    kind: str  # "succeeded" | "terminal" | "transport" | "content_retryable"
    diagnostic: str
    should_retry: bool
    backoff: float


@dataclass(slots=True)
class _RunContext:
    """Shared state for the asyncio worker pool of one job run."""

    profile: Any
    provider: Any
    renderer: Any
    overload: Any
    limiter: AdaptiveLimiter
    writer: "_IndexWriter"
    render_lock: asyncio.Lock
    manifest: "JobManifest"
    pages: dict[str, Any]
    retry_ambiguous: bool


_WRITER_CLOSE = object()


class _IndexWriter:
    """Single serialized writer for the eventually-consistent compact indexes.

    Workers never write the indexes; they push tiny page-state events onto a queue.
    This one task owns the authoritative in-memory ``pages`` and regenerates all
    three indexes together (so they stay mutually consistent), with a coalesced
    flush cadence of ``max(64 events, 250 ms)`` plus an immediate flush on notable
    events (terminal failure / settle / completion). Index writes are atomic but
    NOT fsync'd — they are a derived, rebuildable rollup of the fsync'd truth.
    """

    _FLUSH_EVENTS = 64
    _FLUSH_INTERVAL = 0.25

    def __init__(self, runner: "DocumentOcrRunner", manifest: "JobManifest", pages: dict[str, Any], *, partial: bool) -> None:
        self._runner = runner
        self._manifest = manifest
        self.pages = pages
        self._partial = partial
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        # Dedicated governor: the writer task must not share the runner's governor
        # stack with worker 0 (concurrent acquire/release would corrupt the stack).
        self._governor = ResourceGovernor()

    async def publish(self, page_number: int, entry: dict[str, Any], *, notable: bool = False) -> None:
        await self._queue.put((str(page_number), entry, notable))

    async def close(self) -> None:
        await self._queue.put(_WRITER_CLOSE)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        pending = 0
        last_flush = loop.time()
        while True:
            timeout: float | None = None
            if pending:
                timeout = max(0.0, self._FLUSH_INTERVAL - (loop.time() - last_flush))
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout) if timeout is not None else await self._queue.get()
            except asyncio.TimeoutError:
                self._flush()
                pending = 0
                last_flush = loop.time()
                continue
            if item is _WRITER_CLOSE:
                if pending:
                    self._flush()
                return
            key, entry, notable = item
            self.pages[key] = entry
            pending += 1
            if notable or pending >= self._FLUSH_EVENTS:
                self._flush()
                pending = 0
                last_flush = loop.time()

    def _flush(self) -> None:
        self._runner._write_indexes(
            self._manifest, self.pages, partial=self._partial, fsync=False, governor=self._governor
        )


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    state_root: Path = Path(".paperscale")
    profile: str = "generic_vlm_markdown"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str | None = None
    capacity: str = "local-vllm-small"
    lease_seconds: float = 300.0
    worker_id: str = "local"
    # --- recovery/retry redesign (not persisted in the manifest; read live) ---
    metered: bool = False
    max_attempts: int = 8
    # --- concurrency/queuing knobs ---
    server_max_num_seqs: int = 128
    # max_in_flight_requests default = 4 x server max_num_seqs (set in __post_init__).
    max_in_flight_requests: int | None = None
    max_open_documents: int = 128
    render_ahead: int | None = None

    def __post_init__(self) -> None:
        if self.max_in_flight_requests is None:
            object.__setattr__(self, "max_in_flight_requests", 4 * self.server_max_num_seqs)
        if self.render_ahead is None:
            object.__setattr__(self, "render_ahead", self.max_in_flight_requests)

    @property
    def in_flight_limit(self) -> int:
        return int(self.max_in_flight_requests or (4 * self.server_max_num_seqs))

    @property
    def render_ahead_limit(self) -> int:
        return int(self.render_ahead or self.in_flight_limit)


@dataclass(frozen=True, slots=True)
class JobManifest:
    job_id: str
    input_path: str
    output_path: str
    document_id: str
    page_count: int
    profile: str
    base_url: str
    model: str
    capacity: str
    created_at: float

    def to_json(self) -> Json:
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "job_manifest",
            "job_id": self.job_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "document_id": self.document_id,
            "page_count": self.page_count,
            "profile": self.profile,
            "base_url": self.base_url,
            "model": self.model,
            "capacity": self.capacity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, payload: Json) -> "JobManifest":
        if payload.get("kind") != "job_manifest":
            raise ValueError("expected job manifest")
        return cls(
            job_id=str(payload["job_id"]),
            input_path=str(payload["input_path"]),
            output_path=str(payload["output_path"]),
            document_id=str(payload["document_id"]),
            page_count=int(payload["page_count"]),
            profile=str(payload["profile"]),
            base_url=str(payload["base_url"]),
            model=str(payload["model"]),
            capacity=str(payload["capacity"]),
            created_at=float(payload["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    pages_total: int
    succeeded: int
    failed_retryable: int
    failed_terminal: int
    ambiguous: int
    pending: int
    in_flight: int
    partial: bool
    output_path: str | None = None

    @property
    def complete(self) -> bool:
        return self.pages_total > 0 and self.succeeded == self.pages_total

    @property
    def settled(self) -> bool:
        """No page can make progress on resume.

        True when no ``pending``/``failed_retryable``/``in_flight``/``reserved``
        pages remain (only ``succeeded`` + terminal). Distinguishes a job that is
        settled-with-failures (resume is an infinite no-op) from one that still has
        retryable work. ``in_flight`` already folds in ``reserved`` in the index.
        """
        return self.pending == 0 and self.failed_retryable == 0 and self.in_flight == 0

    @property
    def has_terminal_failures(self) -> bool:
        return self.failed_terminal > 0

    @classmethod
    def from_index(cls, payload: Json) -> "JobStatus":
        if payload.get("kind") != "status_index":
            raise CompactIndexError("missing or corrupt status index")
        return cls(
            job_id=str(payload["job_id"]),
            pages_total=int(payload["pages_total"]),
            succeeded=int(payload.get("succeeded", 0)),
            failed_retryable=int(payload.get("failed_retryable", 0)),
            failed_terminal=int(payload.get("failed_terminal", 0)),
            ambiguous=int(payload.get("ambiguous", 0)),
            pending=int(payload.get("pending", 0)),
            in_flight=int(payload.get("in_flight", 0)),
            partial=bool(payload.get("partial", False)),
            output_path=payload.get("output_path") if isinstance(payload.get("output_path"), str) else None,
        )

    def to_json_summary(self) -> Json:
        return {
            "job_id": self.job_id,
            "pages_total": self.pages_total,
            "succeeded": self.succeeded,
            "failed_retryable": self.failed_retryable,
            "failed_terminal": self.failed_terminal,
            "ambiguous": self.ambiguous,
            "pending": self.pending,
            "in_flight": self.in_flight,
            "partial": self.partial,
            "output_path": self.output_path,
            "complete": self.complete,
            "settled": self.settled,
        }


class DocumentOcrRunner:
    """Single-process durable OCR runner over local job state."""

    def __init__(
        self,
        config: RunnerConfig | None = None,
        *,
        provider: PageOcrProvider | None = None,
        renderer_factory: RendererFactory | None = None,
        clock: Callable[[], float] | None = None,
        governor: ResourceGovernor | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.state = FileSystemStateStore(Path(self.config.state_root))
        self.provider = provider
        self.renderer_factory = renderer_factory or (lambda path, options: PdfPageRenderer(path, render_options=options))
        self.clock = clock or time.time
        self.verifier = DeterministicQualityVerifier()
        self.governor = governor or ResourceGovernor()
        self._governor_injected = governor is not None
        # Default sleeper is asyncio.sleep (non-blocking); only used inside the async
        # pool. Tests inject a sync recorder (e.g. list.append) which is not awaited.
        self._sleep = sleeper or asyncio.sleep

    def _create_job(self, *, input_path: Path, output_path: Path, job_id: str | None) -> tuple[JobManifest, Json, Any]:
        job_id = job_id or _new_job_id(input_path)
        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            raise FileExistsError(f"job {job_id!r} already exists")
        profile = get_builtin_profile(self.config.profile)
        # Open the renderer and read the page count BEFORE creating the job dir, so a
        # corrupt/unreadable PDF fails cleanly without leaving an orphan job directory.
        renderer = self.renderer_factory(Path(input_path), dict(profile.render_options))
        page_count = int(getattr(renderer, "page_count"))
        job_dir.mkdir(parents=True)
        model = self.config.model or profile.default_model
        manifest = JobManifest(
            job_id=job_id,
            input_path=str(Path(input_path)),
            output_path=str(Path(output_path)),
            document_id=Path(input_path).stem or job_id,
            page_count=page_count,
            profile=profile.name,
            base_url=self.config.base_url,
            model=model,
            capacity=self.config.capacity,
            created_at=float(self.clock()),
        )
        pages = {
            str(page): {"state": "pending", "epoch": 0, "attempt_id": None, "fingerprint": None}
            for page in range(1, page_count + 1)
        }
        self._write_manifest(manifest)
        self._write_indexes(manifest, pages, partial=False)
        return manifest, pages, renderer

    def enqueue(self, *, input_path: Path, output_path: Path, job_id: str | None = None) -> str:
        """Register a job (manifest + pending index) without processing it.

        A subsequent ``work`` process (this or another) claims and runs it. This is
        the work-queue front door for the multi-process claim tier.
        """
        manifest, _pages, _renderer = self._create_job(input_path=input_path, output_path=output_path, job_id=job_id)
        return manifest.job_id

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        job_id: str | None = None,
        allow_partial: bool = False,
    ) -> JobStatus:
        manifest, pages, renderer = self._create_job(input_path=input_path, output_path=output_path, job_id=job_id)
        return self._process_with_claim(
            manifest, pages, renderer=renderer, allow_partial=allow_partial, retry_ambiguous=False
        )

    def resume(self, job_id: str, *, retry_ambiguous: bool = False, allow_partial: bool = False) -> JobStatus:
        manifest = self._read_manifest(job_id)
        pages = self._read_pages_from_status(job_id)
        profile = get_builtin_profile(manifest.profile)
        renderer = self.renderer_factory(Path(manifest.input_path), dict(profile.render_options))
        # Reconcile (adopt-then-requeue) runs while we hold the claim so processing
        # sees real state even if the eventually-consistent index lagged a crash.
        return self._process_with_claim(
            manifest, pages, renderer=renderer, allow_partial=allow_partial,
            retry_ambiguous=retry_ambiguous, recover=True,
        )

    def work(self, *, output_dir: Path | None = None, max_jobs: int | None = None) -> list[JobStatus]:
        """Claim and run available incomplete jobs (single machine, multi-process).

        Scans the jobs tree, skips done/live-owned jobs, and resumes each job it can
        claim. Multiple ``paperscale work`` processes can run concurrently; the
        ``O_EXCL`` ClaimStore keeps each job single-owner and reclaims crashed owners
        at a higher epoch. Horizontal scale = many documents across processes.
        """
        store = self._claim_store()
        results: list[JobStatus] = []
        jobs_root = self.state.root / "jobs"
        if not jobs_root.exists():
            return results
        for job_dir in sorted(p for p in jobs_root.iterdir() if p.is_dir()):
            job_id = job_dir.name
            if store.is_done(job_id):
                continue
            try:
                manifest = self._read_manifest(job_id)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            claim = store.try_claim(job_id)
            if claim is None:
                continue  # done or owned by a live peer
            try:
                pages = self._read_pages_from_status(job_id)
                profile = get_builtin_profile(manifest.profile)
                renderer = self.renderer_factory(Path(manifest.input_path), dict(profile.render_options))
                status = self._drive_claimed(
                    store, claim, manifest, pages, renderer=renderer,
                    allow_partial=False, retry_ambiguous=False, recover=True,
                )
                results.append(status)
            finally:
                store.release(claim)
            if max_jobs is not None and len(results) >= max_jobs:
                break
        return results

    def _claim_store(self) -> Any:
        from paperscale.state.claim_store import ClaimStore

        return ClaimStore(
            self.state.root,
            worker_id=self.config.worker_id,
            clock=self.clock,
            lease_seconds=_CLAIM_LEASE_SECONDS,
            heartbeat_seconds=_CLAIM_HEARTBEAT_SECONDS,
        )

    def _process_with_claim(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
        recover: bool = False,
    ) -> JobStatus:
        store = self._claim_store()
        claim = store.try_claim(manifest.job_id, skip_if_done=False)
        if claim is None:
            raise RuntimeError(
                f"job {manifest.job_id!r} is claimed by a live worker; cannot acquire it"
            )
        try:
            return self._drive_claimed(
                store, claim, manifest, pages, renderer=renderer,
                allow_partial=allow_partial, retry_ambiguous=retry_ambiguous, recover=recover,
            )
        finally:
            store.release(claim)

    def _drive_claimed(
        self,
        store: Any,
        claim: Any,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
        recover: bool,
    ) -> JobStatus:
        if recover:
            self._recover_expired_attempts(manifest, pages)
        holder = {"claim": claim}

        def heartbeat() -> None:
            holder["claim"] = store.heartbeat(holder["claim"])

        status = self._process_pages(
            manifest, pages, renderer=renderer, allow_partial=allow_partial,
            retry_ambiguous=retry_ambiguous, heartbeat=heartbeat,
        )
        if status.complete:
            store.mark_done(manifest.job_id)
        return status

    def status(self, job_id: str) -> JobStatus:
        return JobStatus.from_index(self._read_index(job_id, "status"))

    def reconcile(self, job_id: str) -> Json:
        return self._read_index(job_id, "reconcile")

    def fsck(self, job_id: str) -> Json:
        """Scan-only ledger/index/artifact triangulation. Never mutates state."""
        manifest = self._read_manifest(job_id)
        status_index = self._read_index(job_id, "status")
        pages = status_index.get("pages", {})
        if not isinstance(pages, dict):
            raise CompactIndexError(f"missing pages in status index for {job_id}")
        ledger_by_page = self._ledger_attempts_by_page(job_id)
        issues: list[Json] = []
        for page_number in range(1, manifest.page_count + 1):
            page = pages.get(str(page_number))
            if not isinstance(page, dict):
                issues.append({"code": "missing_page", "page_number": page_number})
                continue
            state = page.get("state")
            if state == "succeeded":
                issues.extend(self._fsck_succeeded_page(job_id, page_number, page, ledger_by_page))
            elif state in {"reserved", "in_flight"}:
                issues.append({
                    "code": "stale_lease",
                    "page_number": page_number,
                    "state": state,
                    "lease_expires_at": self._ledger_lease(job_id, page.get("attempt_id")),
                })
        issues.extend(self._fsck_orphan_artifacts(job_id, pages))
        issues.extend(self._fsck_count_mismatches(manifest, status_index, pages))
        return {"job_id": job_id, "scanned": True, "pages_total": manifest.page_count, "issues": issues}

    def _fsck_succeeded_page(self, job_id: str, page_number: int, page: Json, ledger_by_page: dict[int, list[Json]]) -> list[Json]:
        found: list[Json] = []
        artifact_rel = self._artifact_rel(job_id, page_number)
        if not (self.state.root / artifact_rel).exists():
            found.append({"code": "missing_artifact", "page_number": page_number, "path": str(artifact_rel)})
        else:
            artifact = self.state.read_json(artifact_rel)
            ensure_known_schema(artifact)
            indexed = page.get("fingerprint")
            if indexed is not None and artifact.get("fingerprint") != indexed:
                found.append({
                    "code": "fingerprint_mismatch",
                    "page_number": page_number,
                    "indexed": indexed,
                    "artifact": artifact.get("fingerprint"),
                })
        if not any(attempt.get("state") == "succeeded" for attempt in ledger_by_page.get(page_number, [])):
            found.append({
                "code": "ledger_mismatch",
                "page_number": page_number,
                "detail": "succeeded page has no terminal-succeeded ledger attempt",
            })
        return found

    def _fsck_orphan_artifacts(self, job_id: str, pages: Json) -> list[Json]:
        artifact_dir = self._job_dir(job_id) / "artifacts" / "pages"
        if not artifact_dir.exists():
            return []
        known = set(pages)
        return [
            {"code": "orphan_artifact", "page_number": artifact.stem}
            for artifact in sorted(artifact_dir.glob("*.json"))
            if artifact.stem not in known
        ]

    def _fsck_count_mismatches(self, manifest: JobManifest, status_index: Json, pages: Json) -> list[Json]:
        found: list[Json] = []
        actual = _count_states(pages)
        for field in ("succeeded", "failed_retryable", "failed_terminal", "ambiguous", "pending"):
            indexed = int(status_index.get(field, 0))
            if actual.get(field, 0) != indexed:
                found.append({"code": "count_mismatch", "field": field, "indexed": indexed, "actual": actual.get(field, 0)})
        if len(pages) != manifest.page_count:
            found.append({"code": "count_mismatch", "field": "pages_total", "indexed": manifest.page_count, "actual": len(pages)})
        return found

    def _ledger_attempts_by_page(self, job_id: str) -> dict[int, list[Json]]:
        result: dict[int, list[Json]] = {}
        ledger_dir = self._job_dir(job_id) / "ledger"
        if not ledger_dir.exists():
            return result
        for path in sorted(ledger_dir.glob("*.json")):
            try:
                record = self.state.read_json(self._ledger_rel(job_id, path.stem))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            ensure_known_schema(record)
            page_number = record.get("page_number")
            if isinstance(page_number, int):
                result.setdefault(page_number, []).append(record)
        return result

    def _ledger_lease(self, job_id: str, attempt_id: Any) -> float | None:
        if not isinstance(attempt_id, str):
            return None
        try:
            attempt = self.state.read_json(self._ledger_rel(job_id, attempt_id))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        ensure_known_schema(attempt)
        lease = attempt.get("lease_expires_at")
        return float(lease) if isinstance(lease, (int, float)) else None

    def resolve_ambiguous(self, job_id: str, page_number: int, *, action: str) -> JobStatus:
        """Operator action to resolve an ambiguous page by superseding its attempt.

        ``supersede`` discards the uncertain attempt and requeues the page (pending);
        ``accept`` adopts the page's already-written artifact and marks it succeeded.
        Both record the prior attempt as ``superseded`` in the ledger.
        """
        if action not in {"supersede", "accept"}:
            raise ValueError(f"unknown reconcile action {action!r}")
        manifest = self._read_manifest(job_id)
        pages = self._read_pages_from_status(job_id)
        entry = pages.get(str(page_number))
        if not isinstance(entry, dict):
            raise ValueError(f"unknown page {page_number} for job {job_id!r}")
        if entry.get("state") != "ambiguous":
            raise ValueError(f"page {page_number} is not ambiguous (state={entry.get('state')!r})")
        if action == "accept":
            pages[str(page_number)] = self._accept_ambiguous_artifact(job_id, page_number, entry)
        else:
            pages[str(page_number)] = {
                "state": "pending",
                "epoch": int(entry.get("epoch") or 0),
                "attempt_id": None,
                "fingerprint": None,
                "duplicate_call_risk": False,
            }
        self._supersede_attempt(job_id, page_number, entry, resolution="accepted" if action == "accept" else "discarded")
        self._assemble_if_ready(manifest, pages, allow_partial=False)
        return self._write_indexes(manifest, pages, partial=False)

    def _accept_ambiguous_artifact(self, job_id: str, page_number: int, entry: Json) -> Json:
        artifact_rel = self._artifact_rel(job_id, page_number)
        if not (self.state.root / artifact_rel).exists():
            raise FileNotFoundError(f"no artifact to accept for page {page_number} of job {job_id!r}")
        artifact = self.state.read_json(artifact_rel)
        ensure_known_schema(artifact)
        return {
            "state": "succeeded",
            "artifact_path": str(artifact_rel),
            "fingerprint": artifact.get("fingerprint"),
            "epoch": int(entry.get("epoch") or 0),
        }

    def _supersede_attempt(self, job_id: str, page_number: int, entry: Json, *, resolution: str) -> None:
        attempt_id = entry.get("attempt_id")
        if not isinstance(attempt_id, str):
            return
        try:
            attempt = self.state.read_json(self._ledger_rel(job_id, attempt_id))
            ensure_known_schema(attempt)
        except (FileNotFoundError, json.JSONDecodeError):
            attempt = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "kind": "page_attempt",
                "attempt_id": attempt_id,
                "job_id": job_id,
                "page_number": page_number,
            }
        superseded = {**attempt, "state": "superseded", "resolution": resolution, "resolved_at": float(self.clock())}
        self._write_ledger(job_id, attempt_id, superseded)

    def repair_index(self, job_id: str) -> JobStatus:
        manifest = self._read_manifest(job_id)
        old_pages = self._read_pages_from_status(job_id)
        pages: Json = {}
        for page_number in range(1, manifest.page_count + 1):
            page_key = str(page_number)
            old = old_pages.get(page_key, {}) if isinstance(old_pages.get(page_key), dict) else {}
            artifact_path = self.state.root / self._artifact_rel(job_id, page_number)
            if artifact_path.exists():
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                ensure_known_schema(artifact)
                pages[page_key] = {
                    **old,
                    "state": "succeeded",
                    "artifact_path": str(self._artifact_rel(job_id, page_number)),
                    "fingerprint": artifact.get("fingerprint"),
                }
            else:
                pages[page_key] = {**old, "state": "failed_retryable", "artifact_path": None}
        return self._write_indexes(manifest, pages, partial=False)

    def _capacity_for(self, manifest: JobManifest) -> ProviderCapacityProfile:
        try:
            return builtin_capacity_profile(manifest.capacity)
        except ValueError:
            return builtin_capacity_profile("local-vllm-small")

    def _provider_for(self, manifest: JobManifest) -> Any:
        if self.provider is not None:
            return self.provider
        return self._build_async_provider(manifest)

    def _build_async_provider(self, manifest: JobManifest) -> Any:
        from paperscale.providers.async_openai_chat import build_async_chat_provider

        return build_async_chat_provider(
            _normalize_openai_base_url(manifest.base_url),
            max_connections=self.config.in_flight_limit,
            timeout=self._capacity_for(manifest).timeout_seconds,
        )

    def _process_pages(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
        heartbeat: Callable[[], None] | None = None,
    ) -> JobStatus:
        return asyncio.run(
            self._run_pool(
                manifest,
                pages,
                renderer=renderer,
                allow_partial=allow_partial,
                retry_ambiguous=retry_ambiguous,
                heartbeat=heartbeat,
            )
        )

    async def _run_pool(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
        heartbeat: Callable[[], None] | None,
    ) -> JobStatus:
        profile = get_builtin_profile(manifest.profile)
        capacity = self._capacity_for(manifest)
        ceiling = self.config.in_flight_limit
        floor = max(1, min(self.config.server_max_num_seqs, ceiling))
        overload = ProviderOverloadController(capacity, floor=floor, ceiling=ceiling)
        limiter = AdaptiveLimiter(overload.concurrency_limit, maximum=ceiling)
        provider = self._provider_for(manifest)

        queue: asyncio.Queue[int] = asyncio.Queue()
        for page_number in range(1, manifest.page_count + 1):
            state = pages[str(page_number)].get("state")
            if state == "succeeded":
                continue
            if state == "ambiguous" and not retry_ambiguous:
                continue
            if state not in {"pending", "failed_retryable", "reserved", "ambiguous"}:
                continue
            queue.put_nowait(page_number)

        writer = _IndexWriter(self, manifest, pages, partial=allow_partial)
        writer_task = asyncio.create_task(writer.run())
        ctx = _RunContext(
            profile=profile,
            provider=provider,
            renderer=renderer,
            overload=overload,
            limiter=limiter,
            writer=writer,
            render_lock=asyncio.Lock(),
            manifest=manifest,
            pages=pages,
            retry_ambiguous=retry_ambiguous,
        )
        worker_count = max(1, min(ceiling, self.config.render_ahead_limit, queue.qsize() or 1))
        heartbeat_task = (
            asyncio.create_task(self._heartbeat_loop(heartbeat)) if heartbeat is not None else None
        )
        try:
            workers = [
                asyncio.create_task(self._worker(ctx, queue, worker_index=index))
                for index in range(worker_count)
            ]
            await asyncio.gather(*workers)
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
            await writer.close()
            await writer_task
            await self._aclose_provider(provider)

        self._assemble_if_ready(manifest, pages, allow_partial=allow_partial)
        partial = allow_partial and _count_states(pages).get("succeeded", 0) < manifest.page_count
        return self._write_indexes(manifest, pages, partial=partial)

    async def _heartbeat_loop(self, heartbeat: Callable[[], None]) -> None:
        try:
            while True:
                await asyncio.sleep(_CLAIM_HEARTBEAT_SECONDS)
                await asyncio.to_thread(heartbeat)
        except asyncio.CancelledError:
            return

    async def _aclose_provider(self, provider: Any) -> None:
        # Only close providers we created (the injected test provider is the caller's).
        if provider is self.provider:
            return
        closer = getattr(getattr(provider, "_client", None), "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - best-effort cleanup
            return

    async def _worker(self, ctx: "_RunContext", queue: "asyncio.Queue[int]", *, worker_index: int) -> None:
        governor = self.governor if (worker_index == 0 and self._governor_injected) else ResourceGovernor()
        while True:
            if ctx.overload.circuit_open:
                return
            try:
                page_number = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await self._process_one_page(ctx, governor, page_number)
            finally:
                queue.task_done()

    async def _process_one_page(self, ctx: "_RunContext", governor: ResourceGovernor, page_number: int) -> None:
        key = str(page_number)
        entry = ctx.pages[key]
        accumulated: dict[str, Any] = dict(entry.get("overrides") or {})
        attempts = int(entry.get("epoch") or 0)
        while True:
            if ctx.overload.circuit_open:
                return
            if attempts >= self.config.max_attempts:
                await self._demote_terminal(ctx, page_number, accumulated)
                return
            attempts += 1
            eff_profile = (
                ctx.profile.with_overrides(
                    decoding=accumulated.get("decoding"), render_options=accumulated.get("render_options")
                )
                if accumulated
                else ctx.profile
            )
            result = await self._attempt_page(ctx, governor, page_number, attempts, eff_profile, accumulated)
            if result.kind in ("succeeded", "terminal"):
                return
            if result.kind == "transport":
                if ctx.overload.circuit_open or not result.should_retry:
                    return
                await self._maybe_sleep(result.backoff)
                continue
            # content_retryable: escalate remediation on the base profile, then re-attempt.
            accumulated = ctx.profile.remediation_for(result.diagnostic, accumulated=accumulated)

    async def _attempt_page(
        self,
        ctx: "_RunContext",
        governor: ResourceGovernor,
        page_number: int,
        epoch: int,
        eff_profile: Any,
        accumulated: dict[str, Any],
    ) -> "_AttemptResult":
        manifest = ctx.manifest
        page_id = f"{manifest.document_id}:{page_number}"
        attempt_id = str(uuid.uuid4())
        now = float(self.clock())
        render_override = accumulated.get("render_options")
        with governor.acquire(ResourceKind.SCHEDULER):
            with governor.acquire(ResourceKind.RENDER):
                async with ctx.render_lock:
                    if render_override:
                        renderer = await asyncio.to_thread(
                            self.renderer_factory, Path(manifest.input_path), dict(eff_profile.render_options)
                        )
                    else:
                        renderer = ctx.renderer
                    rendered = await asyncio.to_thread(renderer.render_page, page_number)
            request = eff_profile.build_request(
                page_id,
                rendered.image_bytes,
                getattr(rendered, "media_type", "image/png"),
                model=manifest.model,
            )
            reserved_entry = {
                "state": "reserved",
                "epoch": epoch,
                "attempt_id": attempt_id,
                "fingerprint": request.fingerprint,
                "overrides": accumulated,
            }
            base_attempt = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "kind": "page_attempt",
                "attempt_id": attempt_id,
                "job_id": manifest.job_id,
                "page_id": page_id,
                "page_number": page_number,
                "state": "reserved",
                "fingerprint": request.fingerprint,
                "worker_id": self.config.worker_id,
                "epoch": epoch,
                "lease_expires_at": now + self.config.lease_seconds,
                "heartbeat_at": now,
                "provider_started_at": None,
                "provider_response_committed_at": None,
                "result_pointer": None,
                "diagnostic": None,
            }
            with governor.acquire(ResourceKind.PROVIDER):
                with governor.acquire(ResourceKind.PAGE_LEASE):
                    await self._awrite_ledger(governor, manifest.job_id, attempt_id, base_attempt)
                    await ctx.writer.publish(page_number, reserved_entry)
                    in_flight = {**base_attempt, "state": "in_flight", "provider_started_at": float(self.clock())}
                    await self._awrite_ledger(governor, manifest.job_id, attempt_id, in_flight)
                    await ctx.writer.publish(page_number, {**reserved_entry, "state": "in_flight"})
                    try:
                        async with ctx.limiter.slot():
                            response = await self._send(ctx.provider, request)
                    except Exception as exc:  # noqa: BLE001 - transport/overload signal
                        await self._awrite_ledger(
                            governor, manifest.job_id, attempt_id,
                            {**in_flight, "state": "failed_retryable", "diagnostic": str(exc)},
                        )
                        await ctx.writer.publish(
                            page_number, {**reserved_entry, "state": "failed_retryable", "diagnostic": str(exc)}
                        )
                        decision = ctx.overload.record_failure(retryable=True)
                        await ctx.limiter.set_limit(ctx.overload.concurrency_limit)
                        return _AttemptResult("transport", str(exc), decision.should_retry, decision.backoff_seconds)

                    parsed = eff_profile.parse_and_validate(response.markdown)
                    if not parsed.ok:
                        blank = await self._maybe_commit_blank(
                            ctx, governor, page_number, attempt_id, in_flight, reserved_entry, request, response,
                            diagnostic=parsed.diagnostic, image_bytes=rendered.image_bytes,
                        )
                        if blank is not None:
                            return blank
                        return await self._record_content_failure(
                            ctx, governor, page_number, attempt_id, in_flight, reserved_entry,
                            terminal=parsed.retry_classification == "terminal", diagnostic=parsed.diagnostic,
                        )
                    finding = self.verifier.classify(parsed.markdown)
                    if not finding.accepted:
                        blank = await self._maybe_commit_blank(
                            ctx, governor, page_number, attempt_id, in_flight, reserved_entry, request, response,
                            diagnostic=finding.kind, image_bytes=rendered.image_bytes,
                        )
                        if blank is not None:
                            return blank
                        return await self._record_content_failure(
                            ctx, governor, page_number, attempt_id, in_flight, reserved_entry,
                            terminal=finding.retry_class == "terminal", diagnostic=finding.kind,
                        )
                    await self._commit_success(
                        ctx, governor, page_number, attempt_id, in_flight, reserved_entry, request, response, parsed, finding
                    )
                    ctx.overload.record_success()
                    await ctx.limiter.set_limit(ctx.overload.concurrency_limit)
                    return _AttemptResult("succeeded", "ok", False, 0.0)

    async def _record_content_failure(
        self,
        ctx: "_RunContext",
        governor: ResourceGovernor,
        page_number: int,
        attempt_id: str,
        attempt: Json,
        reserved_entry: Json,
        *,
        terminal: bool,
        diagnostic: str,
    ) -> "_AttemptResult":
        # Signal hygiene: a content/parse/verify rejection is NOT overload -> never throttle.
        ctx.overload.note_content_failure()
        state = "failed_terminal" if terminal else "failed_retryable"
        await self._awrite_ledger(
            governor, ctx.manifest.job_id, attempt_id, {**attempt, "state": state, "diagnostic": diagnostic}
        )
        await ctx.writer.publish(
            page_number, {**reserved_entry, "state": state, "diagnostic": diagnostic}, notable=terminal
        )
        return _AttemptResult("terminal" if terminal else "content_retryable", diagnostic, not terminal, 0.0)

    async def _commit_success(
        self,
        ctx: "_RunContext",
        governor: ResourceGovernor,
        page_number: int,
        attempt_id: str,
        attempt: Json,
        reserved_entry: Json,
        request: Any,
        response: Any,
        parsed: Any,
        finding: Any,
    ) -> None:
        manifest = ctx.manifest
        artifact_rel = self._artifact_rel(manifest.job_id, page_number)
        artifact = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "page_artifact",
            "document_id": manifest.document_id,
            "page_number": page_number,
            "page_id": request.page_id,
            "markdown": parsed.markdown,
            "result_pointer": str(artifact_rel),
            "verifier_metadata": [
                {
                    "accepted": finding.accepted,
                    "kind": finding.kind,
                    "retry_class": finding.retry_class,
                    "warnings": list(finding.warnings),
                }
            ],
            "fingerprint": request.fingerprint,
            "image_hash": request.image_hash,
            "provider_request_id": response.provider_request_id,
            "provider_metadata": response.metadata,
            "profile_metadata": parsed.metadata,
        }
        # Write the durable truth (artifact + ledger), fsync'd, before the index event.
        with governor.acquire(ResourceKind.STATE_STORE):
            await asyncio.to_thread(self.state.write_json_atomic, artifact_rel, artifact)
        committed = {
            **attempt,
            "state": "succeeded",
            "provider_response_committed_at": float(self.clock()),
            "result_pointer": str(artifact_rel),
        }
        await self._awrite_ledger(governor, manifest.job_id, attempt_id, committed)
        await ctx.writer.publish(
            page_number,
            {**reserved_entry, "state": "succeeded", "artifact_path": str(artifact_rel)},
            notable=False,
        )

    async def _maybe_commit_blank(
        self,
        ctx: "_RunContext",
        governor: ResourceGovernor,
        page_number: int,
        attempt_id: str,
        attempt: Json,
        reserved_entry: Json,
        request: Any,
        response: Any,
        *,
        diagnostic: str,
        image_bytes: bytes,
    ) -> "_AttemptResult | None":
        """Accept a genuinely-blank page as a successful empty artifact, or return None.

        Triggers only when the render is near-blank (the ink gate) AND the OCR produced
        no readable content — either empty output or a degenerate repetition loop, both
        of which a VLM emits on a blank/noise page. The ink gate protects real content
        pages (ink above threshold), which never reach here and remediate normally.
        """
        if diagnostic not in _BLANK_ELIGIBLE_DIAGNOSTICS or not _render_is_blank(image_bytes):
            return None
        from paperscale.profiles.base import ProfileValidationResult
        from paperscale.quality.verifier import VerificationFinding

        parsed = ProfileValidationResult(ok=True, markdown="", metadata={"blank_page": True})
        finding = VerificationFinding(True, "blank_page", "none", [])
        await self._commit_success(
            ctx, governor, page_number, attempt_id, attempt, reserved_entry, request, response, parsed, finding
        )
        ctx.overload.record_success()
        await ctx.limiter.set_limit(ctx.overload.concurrency_limit)
        return _AttemptResult("succeeded", "blank_page", False, 0.0)

    async def _demote_terminal(self, ctx: "_RunContext", page_number: int, accumulated: dict[str, Any]) -> None:
        diagnostic = f"exhausted {self.config.max_attempts} attempts"
        entry = dict(ctx.pages[str(page_number)])
        entry.update({"state": "failed_terminal", "diagnostic": diagnostic, "overrides": accumulated})
        await ctx.writer.publish(page_number, entry, notable=True)

    async def _awrite_ledger(self, governor: ResourceGovernor, job_id: str, attempt_id: str, payload: Json) -> None:
        with governor.acquire(ResourceKind.STATE_STORE):
            await asyncio.to_thread(self.state.write_json_atomic, self._ledger_rel(job_id, attempt_id), payload)

    async def _send(self, provider: Any, request: Any) -> Any:
        send = provider.send
        if inspect.iscoroutinefunction(send):
            return await send(request)
        result = await asyncio.to_thread(send, request)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _maybe_sleep(self, seconds: float) -> None:
        result = self._sleep(seconds)
        if inspect.isawaitable(result):
            await result

    def _assemble_if_ready(self, manifest: JobManifest, pages: Json, *, allow_partial: bool) -> None:
        succeeded_pages = [int(page) for page, entry in pages.items() if isinstance(entry, dict) and entry.get("state") == "succeeded"]
        if len(succeeded_pages) < manifest.page_count and not allow_partial:
            return
        if not succeeded_pages:
            return
        artifacts: list[PageArtifact] = []
        for page_number in sorted(succeeded_pages):
            payload = self.state.read_json(self._artifact_rel(manifest.job_id, page_number))
            ensure_known_schema(payload)
            artifacts.append(
                PageArtifact(
                    page_id=str(payload["page_id"]),
                    markdown=str(payload["markdown"]),
                    result_pointer=str(payload["result_pointer"]),
                    verifier_metadata=payload.get("verifier_metadata"),
                )
            )
        result = MarkdownAssembler(required_pages=list(range(1, manifest.page_count + 1))).assemble(artifacts, allow_partial=allow_partial)
        _atomic_write_text(Path(manifest.output_path), result.markdown)

    def _recover_expired_attempts(self, manifest: JobManifest, pages: Json) -> None:
        now = float(self.clock())
        changed = False
        for page_text, entry in list(pages.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("state") not in {"reserved", "in_flight"}:
                continue
            attempt_id = entry.get("attempt_id")
            if not isinstance(attempt_id, str):
                continue
            attempt_path = self._ledger_rel(manifest.job_id, attempt_id)
            try:
                attempt = self.state.read_json(attempt_path)
            except FileNotFoundError:
                continue
            ensure_known_schema(attempt)
            if float(attempt.get("lease_expires_at", now + 1)) > now:
                continue
            if attempt.get("state") not in {"reserved", "in_flight"}:
                continue

            # Adopt-then-requeue: the crash window is most likely *after* the artifact
            # is durably written but *before* the index recorded success. If the
            # artifact exists and its fingerprint matches this attempt's fingerprint,
            # adopt it (zero provider calls) rather than re-calling.
            page_number = int(page_text)
            adopted = self._adopt_if_artifact_matches(manifest, attempt, page_number)
            if adopted is not None:
                self._write_ledger(
                    manifest.job_id, attempt_id,
                    {**attempt, "state": "succeeded", "result_pointer": adopted["artifact_path"]},
                )
                pages[page_text] = adopted
                changed = True
                continue

            # No adoptable artifact. Auto-requeue by default; only a metered endpoint
            # (billable / non-idempotent) escalates a crashed in_flight to ambiguous.
            if attempt.get("state") == "reserved" and attempt.get("provider_started_at") is None:
                new_state = "pending"
            elif attempt.get("state") == "in_flight":
                new_state = "ambiguous" if self.config.metered else "pending"
            else:
                new_state = "pending"
            duplicate_call_risk = new_state == "ambiguous"
            attempt = {**attempt, "state": new_state, "duplicate_call_risk": duplicate_call_risk}
            self._write_ledger(manifest.job_id, attempt_id, attempt)
            requeued = {
                "state": new_state,
                "epoch": int(entry.get("epoch") or attempt.get("epoch") or 0),
                "attempt_id": None if new_state == "pending" else attempt_id,
                "fingerprint": None if new_state == "pending" else entry.get("fingerprint"),
                "duplicate_call_risk": duplicate_call_risk,
            }
            if entry.get("overrides"):
                requeued["overrides"] = entry["overrides"]
            pages[page_text] = requeued
            changed = True
        if changed:
            self._assemble_if_ready(manifest, pages, allow_partial=False)
            self._write_indexes(manifest, pages, partial=False)

    def _adopt_if_artifact_matches(self, manifest: JobManifest, attempt: Json, page_number: int) -> Json | None:
        """Return a succeeded page entry if a fingerprint-matching artifact exists.

        The fingerprint match proves the artifact came from *this* attempt's exact
        request (not a stale/superseded epoch). Mirrors ``repair_index`` adoption.
        """
        artifact_rel = self._artifact_rel(manifest.job_id, page_number)
        if not (self.state.root / artifact_rel).exists():
            return None
        artifact = self.state.read_json(artifact_rel)
        ensure_known_schema(artifact)
        if artifact.get("fingerprint") != attempt.get("fingerprint"):
            return None
        return {
            "state": "succeeded",
            "artifact_path": str(artifact_rel),
            "fingerprint": artifact.get("fingerprint"),
            "epoch": int(attempt.get("epoch") or 0),
        }

    def _write_manifest(self, manifest: JobManifest) -> None:
        with self.governor.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._manifest_rel(manifest.job_id), manifest.to_json())

    def _read_manifest(self, job_id: str) -> JobManifest:
        payload = self.state.read_json(self._manifest_rel(job_id))
        ensure_known_schema(payload)
        return JobManifest.from_json(payload)

    def _read_index(self, job_id: str, name: str) -> Json:
        try:
            payload = self.state.read_json(self._index_rel(job_id, name))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise CompactIndexError(f"missing or corrupt {name} index for {job_id}") from exc
        if not isinstance(payload, dict):
            raise CompactIndexError(f"missing or corrupt {name} index for {job_id}")
        ensure_known_schema(payload)
        return payload

    def _read_pages_from_status(self, job_id: str) -> Json:
        status = self._read_index(job_id, "status")
        pages = status.get("pages")
        if not isinstance(pages, dict):
            raise CompactIndexError(f"missing pages in status index for {job_id}")
        return {str(key): dict(value) for key, value in pages.items() if isinstance(value, dict)}

    def _write_indexes(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        partial: bool,
        fsync: bool = True,
        governor: ResourceGovernor | None = None,
    ) -> JobStatus:
        counts = _count_states(pages)
        status_index: Json = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "status_index",
            "job_id": manifest.job_id,
            "pages_total": manifest.page_count,
            "succeeded": counts.get("succeeded", 0),
            "failed_retryable": counts.get("failed_retryable", 0),
            "failed_terminal": counts.get("failed_terminal", 0),
            "ambiguous": counts.get("ambiguous", 0),
            "pending": counts.get("pending", 0),
            "in_flight": counts.get("in_flight", 0) + counts.get("reserved", 0),
            "partial": partial,
            "output_path": manifest.output_path,
            "pages": pages,
        }
        resume_index: Json = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "resume_index",
            "job_id": manifest.job_id,
            "pending_pages": _pages_in_states(pages, {"pending", "failed_retryable"}),
            "ambiguous_pages": _pages_in_states(pages, {"ambiguous"}),
            "in_flight_pages": _pages_in_states(pages, {"reserved", "in_flight"}),
        }
        reconcile_index: Json = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "reconcile_index",
            "job_id": manifest.job_id,
            "ambiguous_attempts": [
                {
                    "page_number": int(page),
                    "attempt_id": entry.get("attempt_id"),
                    "duplicate_call_risk": True,
                    "recommended_actions": ["supersede", "accept"],
                }
                for page, entry in pages.items()
                if isinstance(entry, dict) and entry.get("state") == "ambiguous"
            ],
        }
        gov = governor or self.governor
        with gov.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "status"), status_index, fsync=fsync)
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "resume"), resume_index, fsync=fsync)
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "reconcile"), reconcile_index, fsync=fsync)
        return JobStatus.from_index(status_index)

    def _write_ledger(self, job_id: str, attempt_id: str, payload: Json) -> None:
        with self.governor.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._ledger_rel(job_id, attempt_id), payload)

    def _job_dir(self, job_id: str) -> Path:
        return self.state.root / "jobs" / job_id

    @staticmethod
    def _manifest_rel(job_id: str) -> Path:
        return Path("jobs") / job_id / "manifest.json"

    @staticmethod
    def _index_rel(job_id: str, name: str) -> Path:
        return Path("jobs") / job_id / "indexes" / f"{name}.json"

    @staticmethod
    def _ledger_rel(job_id: str, attempt_id: str) -> Path:
        return Path("jobs") / job_id / "ledger" / f"{attempt_id}.json"

    @staticmethod
    def _artifact_rel(job_id: str, page_number: int) -> Path:
        return Path("jobs") / job_id / "artifacts" / "pages" / f"{page_number}.json"


def doctor_provider(*, base_url: str, model: str, capacity: str, profile: str, http_client: Any | None = None) -> Json:
    selected_profile = get_builtin_profile(profile)
    server = InferenceServerProfile(endpoint=_normalize_openai_base_url(base_url), served_model=model)
    capacity_profile = builtin_capacity_profile(capacity)
    provider = SelfHostedOpenAICompatibleProvider(server, capacity_profile, http_client=http_client or _UrllibHttpClient())
    result = provider.health_check()
    return {
        "endpoint": result.endpoint,
        "observed_models": list(result.observed_models),
        "model": model,
        "ocr_profile": selected_profile.name,
        "capacity_profile": capacity_profile.name,
        "compatible": result.ok,
        "diagnostic": result.diagnostic,
    }


class _UrllibHttpClient:
    def get(self, url: str, *, timeout: float) -> Any:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - user/local provider URL
            status_code = int(getattr(response, "status", 0) or response.getcode())
            payload = json.loads(response.read().decode("utf-8"))
        return _HttpResponse(status_code, payload)


class _HttpResponse:
    def __init__(self, status_code: int, payload: Any) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _normalize_openai_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        return stripped
    return f"{stripped}/v1"


def _new_job_id(input_path: Path) -> str:
    return f"{input_path.stem or 'job'}-{uuid.uuid4().hex[:8]}"


def _count_states(pages: Json) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in pages.values():
        if isinstance(entry, dict):
            state = str(entry.get("state", "pending"))
            counts[state] = counts.get(state, 0) + 1
    return counts


def _pages_in_states(pages: Json, states: set[str]) -> list[int]:
    return [int(page) for page, entry in sorted(pages.items(), key=lambda item: int(item[0])) if isinstance(entry, dict) and entry.get("state") in states]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
        _fsync_parent(path.parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_parent(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
