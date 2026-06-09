from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.contracts import UnknownSchemaVersionError
from paperscale.providers.base import PageOcrResponse
from paperscale.runner import DocumentOcrRunner, RunnerConfig


@dataclass(frozen=True)
class _FakeRenderedPage:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class _FakeRenderer:
    def __init__(self, pages: list[bytes]) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def render_page(self, page_number: int) -> _FakeRenderedPage:
        image = self._pages[page_number - 1]
        return _FakeRenderedPage(page_number, image, hashlib.sha256(image).hexdigest())


class _OkProvider:
    name = "ok-provider"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def send(self, request: Any) -> PageOcrResponse:
        return PageOcrResponse(markdown=self._responses.pop(0), provider_request_id=request.fingerprint)


def _bump_schema(path: Path, *, version: int = 2) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = version
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _run_completed_job(state_root: Path, job_id: str, *, pages: int = 1) -> DocumentOcrRunner:
    runner = DocumentOcrRunner(
        RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
        provider=_OkProvider([f"# Page {n}\n\nBody {n}" for n in range(1, pages + 1)]),
        renderer_factory=lambda _path, _options: _FakeRenderer([f"page-{n}".encode() for n in range(1, pages + 1)]),
    )
    runner.run(input_path=state_root / "in.pdf", output_path=state_root / "out.md", job_id=job_id)
    return runner


class SchemaFailClosedTests(unittest.TestCase):
    def test_status_fails_closed_on_future_status_index_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-status")
            _bump_schema(state_root / "jobs" / "job-status" / "indexes" / "status.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.status("job-status")

    def test_resume_fails_closed_on_future_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-manifest")
            _bump_schema(state_root / "jobs" / "job-manifest" / "manifest.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.resume("job-manifest")

    def test_reconcile_fails_closed_on_future_reconcile_index_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-reconcile")
            _bump_schema(state_root / "jobs" / "job-reconcile" / "indexes" / "reconcile.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.reconcile("job-reconcile")

    def test_fsck_fails_closed_on_future_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-fsck")
            _bump_schema(state_root / "jobs" / "job-fsck" / "manifest.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.fsck("job-fsck")

    def test_repair_index_fails_closed_on_future_artifact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-repair")
            _bump_schema(state_root / "jobs" / "job-repair" / "artifacts" / "pages" / "1.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.repair_index("job-repair")

    def test_resume_assembly_fails_closed_on_future_artifact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-assemble")
            _bump_schema(state_root / "jobs" / "job-assemble" / "artifacts" / "pages" / "1.json")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.resume("job-assemble")

    def test_resume_recovery_fails_closed_on_future_ledger_attempt_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-ledger")
            job_dir = state_root / "jobs" / "job-ledger"
            attempt_path = next((job_dir / "ledger").glob("*.json"))
            attempt_id = attempt_path.stem
            _bump_schema(attempt_path)
            # Force the page back into an in-flight state so resume must read the ledger attempt.
            status_path = job_dir / "indexes" / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["pages"]["1"] = {"state": "in_flight", "epoch": 1, "attempt_id": attempt_id}
            status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
            with self.assertRaises(UnknownSchemaVersionError):
                runner.resume("job-ledger")

    def test_status_succeeds_on_current_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_completed_job(state_root, "job-ok")
            self.assertTrue(runner.status("job-ok").complete)


if __name__ == "__main__":
    unittest.main()
