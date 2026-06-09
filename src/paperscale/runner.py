"""Durable local-first document-to-Markdown OCR runner."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable
import uuid
import urllib.request

from paperscale.assembly import MarkdownAssembler
from paperscale.contracts import CURRENT_SCHEMA_VERSION, PageArtifact, ensure_known_schema
from paperscale.profiles.builtin import get_builtin_profile
from paperscale.providers.base import PageOcrProvider
from paperscale.providers.openai_chat import OpenAIChatProvider
from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
)
from paperscale.quality.verifier import DeterministicQualityVerifier
from paperscale.rendering import PdfPageRenderer
from paperscale.scheduler import CompactIndexError
from paperscale.state.fs_store import FileSystemStateStore

Json = dict[str, Any]
RendererFactory = Callable[[Path, dict[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    state_root: Path = Path(".paperscale")
    profile: str = "generic_vlm_markdown"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str | None = None
    capacity: str = "local-vllm-small"
    lease_seconds: float = 300.0
    worker_id: str = "local"


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
    ) -> None:
        self.config = config or RunnerConfig()
        self.state = FileSystemStateStore(Path(self.config.state_root))
        self.provider = provider
        self.renderer_factory = renderer_factory or (lambda path, options: PdfPageRenderer(path, render_options=options))
        self.clock = clock or time.time
        self.verifier = DeterministicQualityVerifier()

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        job_id: str | None = None,
        allow_partial: bool = False,
    ) -> JobStatus:
        job_id = job_id or _new_job_id(input_path)
        job_dir = self._job_dir(job_id)
        if job_dir.exists():
            raise FileExistsError(f"job {job_id!r} already exists")
        job_dir.mkdir(parents=True)
        profile = get_builtin_profile(self.config.profile)
        render_options = dict(profile.render_options)
        renderer = self.renderer_factory(Path(input_path), render_options)
        page_count = int(getattr(renderer, "page_count"))
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
        return self._process_pages(manifest, pages, renderer=renderer, allow_partial=allow_partial, retry_ambiguous=False)

    def resume(self, job_id: str, *, retry_ambiguous: bool = False, allow_partial: bool = False) -> JobStatus:
        manifest = self._read_manifest(job_id)
        pages = self._read_pages_from_status(job_id)
        self._recover_expired_attempts(manifest, pages)
        profile = get_builtin_profile(manifest.profile)
        renderer = self.renderer_factory(Path(manifest.input_path), dict(profile.render_options))
        return self._process_pages(manifest, pages, renderer=renderer, allow_partial=allow_partial, retry_ambiguous=retry_ambiguous)

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

    def _process_pages(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
    ) -> JobStatus:
        profile = get_builtin_profile(manifest.profile)
        for page_number in range(1, manifest.page_count + 1):
            page = pages[str(page_number)]
            state = page.get("state")
            if state == "succeeded":
                continue
            if state == "ambiguous" and not retry_ambiguous:
                continue
            if state not in {"pending", "failed_retryable", "reserved", "ambiguous"}:
                continue
            rendered = renderer.render_page(page_number)
            request = profile.build_request(
                f"{manifest.document_id}:{page_number}",
                rendered.image_bytes,
                getattr(rendered, "media_type", "image/png"),
                model=manifest.model,
            )
            self._process_page(manifest, pages, page_number, request)
        self._assemble_if_ready(manifest, pages, allow_partial=allow_partial)
        return self._write_indexes(manifest, pages, partial=allow_partial and _count_states(pages).get("succeeded", 0) < manifest.page_count)


    def _provider_for(self, manifest: JobManifest) -> PageOcrProvider:
        if self.provider is not None:
            return self.provider
        return _default_provider(manifest.base_url)

    def _process_page(self, manifest: JobManifest, pages: Json, page_number: int, request: Any) -> None:
        page_key = str(page_number)
        previous = pages[page_key]
        epoch = int(previous.get("epoch") or 0) + 1
        attempt_id = str(uuid.uuid4())
        now = float(self.clock())
        attempt = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "page_attempt",
            "attempt_id": attempt_id,
            "job_id": manifest.job_id,
            "page_id": request.page_id,
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
        self._write_ledger(manifest.job_id, attempt_id, attempt)
        pages[page_key] = {
            "state": "reserved",
            "epoch": epoch,
            "attempt_id": attempt_id,
            "fingerprint": request.fingerprint,
        }
        self._write_indexes(manifest, pages, partial=False)

        attempt = {**attempt, "state": "in_flight", "provider_started_at": float(self.clock())}
        self._write_ledger(manifest.job_id, attempt_id, attempt)
        pages[page_key] = {**pages[page_key], "state": "in_flight"}
        self._write_indexes(manifest, pages, partial=False)

        try:
            response = self._provider_for(manifest).send(request)
        except Exception as exc:
            self._fail_attempt(manifest, pages, page_number, attempt, state="failed_retryable", diagnostic=str(exc))
            return

        parsed = get_builtin_profile(manifest.profile).parse_and_validate(response.markdown)
        if not parsed.ok:
            state = "failed_terminal" if parsed.retry_classification == "terminal" else "failed_retryable"
            self._fail_attempt(manifest, pages, page_number, attempt, state=state, diagnostic=parsed.diagnostic)
            return
        finding = self.verifier.classify(parsed.markdown)
        if not finding.accepted:
            state = "failed_terminal" if finding.retry_class == "terminal" else "failed_retryable"
            self._fail_attempt(manifest, pages, page_number, attempt, state=state, diagnostic=finding.kind)
            return

        artifact_rel = self._artifact_rel(manifest.job_id, page_number)
        artifact = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "page_artifact",
            "document_id": manifest.document_id,
            "page_number": page_number,
            "page_id": request.page_id,
            "markdown": parsed.markdown,
            "result_pointer": str(artifact_rel),
            "verifier_metadata": [finding.__dict__ if hasattr(finding, "__dict__") else {
                "accepted": finding.accepted,
                "kind": finding.kind,
                "retry_class": finding.retry_class,
                "warnings": list(finding.warnings),
            }],
            "fingerprint": request.fingerprint,
            "image_hash": request.image_hash,
            "provider_request_id": response.provider_request_id,
            "provider_metadata": response.metadata,
            "profile_metadata": parsed.metadata,
        }
        self.state.write_json_atomic(artifact_rel, artifact)
        committed = {
            **attempt,
            "state": "succeeded",
            "provider_response_committed_at": float(self.clock()),
            "result_pointer": str(artifact_rel),
        }
        self._write_ledger(manifest.job_id, attempt_id, committed)
        pages[page_key] = {
            **pages[page_key],
            "state": "succeeded",
            "artifact_path": str(artifact_rel),
            "fingerprint": request.fingerprint,
        }
        self._write_indexes(manifest, pages, partial=False)

    def _fail_attempt(self, manifest: JobManifest, pages: Json, page_number: int, attempt: Json, *, state: str, diagnostic: str) -> None:
        failed = {**attempt, "state": state, "diagnostic": diagnostic}
        self._write_ledger(manifest.job_id, str(attempt["attempt_id"]), failed)
        pages[str(page_number)] = {**pages[str(page_number)], "state": state, "diagnostic": diagnostic}
        self._write_indexes(manifest, pages, partial=False)

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
            if attempt.get("state") == "reserved" and attempt.get("provider_started_at") is None:
                new_state = "pending"
            elif attempt.get("state") == "in_flight" and attempt.get("result_pointer") is None:
                new_state = "ambiguous"
            else:
                continue
            attempt = {**attempt, "state": new_state, "duplicate_call_risk": new_state == "ambiguous"}
            self._write_ledger(manifest.job_id, attempt_id, attempt)
            pages[page_text] = {**entry, "state": new_state, "duplicate_call_risk": new_state == "ambiguous"}
            changed = True
        if changed:
            self._write_indexes(manifest, pages, partial=False)

    def _write_manifest(self, manifest: JobManifest) -> None:
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

    def _write_indexes(self, manifest: JobManifest, pages: Json, *, partial: bool) -> JobStatus:
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
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "status"), status_index)
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "resume"), resume_index)
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "reconcile"), reconcile_index)
        return JobStatus.from_index(status_index)

    def _write_ledger(self, job_id: str, attempt_id: str, payload: Json) -> None:
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


def _default_provider(base_url: str) -> OpenAIChatProvider:
    from openai import OpenAI

    return OpenAIChatProvider(client=OpenAI(base_url=_normalize_openai_base_url(base_url), api_key="paperscale-local"))


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
