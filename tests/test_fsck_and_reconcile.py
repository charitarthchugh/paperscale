from __future__ import annotations

import hashlib
import io
import contextlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _run_job(state_root: Path, job_id: str, *, pages: int = 2) -> DocumentOcrRunner:
    runner = DocumentOcrRunner(
        RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
        provider=_OkProvider([f"# Page {n}\n\nBody {n}" for n in range(1, pages + 1)]),
        renderer_factory=lambda _path, _options: _FakeRenderer([f"page-{n}".encode() for n in range(1, pages + 1)]),
    )
    runner.run(input_path=state_root / "in.pdf", output_path=state_root / "out.md", job_id=job_id)
    return runner


def _read_status(job_dir: Path) -> dict[str, Any]:
    return json.loads((job_dir / "indexes" / "status.json").read_text(encoding="utf-8"))


def _write_status(job_dir: Path, status: dict[str, Any]) -> None:
    (job_dir / "indexes" / "status.json").write_text(json.dumps(status, sort_keys=True), encoding="utf-8")


def _ledger_attempt_id_for_page(job_dir: Path, page_number: int) -> str:
    for path in (job_dir / "ledger").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("page_number") == page_number:
            return str(record["attempt_id"])
    raise AssertionError(f"no ledger attempt for page {page_number}")


def _make_page_ambiguous(job_dir: Path, page_number: int) -> str:
    attempt_id = _ledger_attempt_id_for_page(job_dir, page_number)
    status = _read_status(job_dir)
    page = status["pages"][str(page_number)]
    page.update({"state": "ambiguous", "attempt_id": attempt_id, "duplicate_call_risk": True})
    _write_status(job_dir, status)
    return attempt_id


def _make_page_inflight_expired(job_dir: Path, page_number: int) -> str:
    """Force a page back into a crashed in-flight state with an expired lease."""
    attempt_id = _ledger_attempt_id_for_page(job_dir, page_number)
    attempt_path = job_dir / "ledger" / f"{attempt_id}.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    attempt.update({"state": "in_flight", "result_pointer": None, "provider_started_at": 1.0, "lease_expires_at": 1.0})
    attempt_path.write_text(json.dumps(attempt, sort_keys=True), encoding="utf-8")
    status = _read_status(job_dir)
    status["pages"][str(page_number)] = {"state": "in_flight", "epoch": 1, "attempt_id": attempt_id}
    _write_status(job_dir, status)
    return attempt_id


def _codes(report: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


class FsckTriangulationTests(unittest.TestCase):
    def test_count_mismatch_when_summary_disagrees_with_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            status = _read_status(job_dir)
            status["succeeded"] = 5  # lie about the count
            _write_status(job_dir, status)
            report = runner.fsck("job")
            self.assertIn("count_mismatch", _codes(report))

    def test_ledger_mismatch_when_succeeded_page_has_no_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            attempt_id = _ledger_attempt_id_for_page(job_dir, 2)
            (job_dir / "ledger" / f"{attempt_id}.json").unlink()
            report = runner.fsck("job")
            self.assertIn("ledger_mismatch", _codes(report))

    def test_fingerprint_mismatch_between_index_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            artifact_path = job_dir / "artifacts" / "pages" / "2.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["fingerprint"] = "tampered-fingerprint"
            artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
            report = runner.fsck("job")
            self.assertIn("fingerprint_mismatch", _codes(report))

    def test_stale_lease_for_stuck_in_flight_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            status = _read_status(job_dir)
            status["pages"]["2"]["state"] = "in_flight"
            _write_status(job_dir, status)
            report = runner.fsck("job")
            self.assertIn("stale_lease", _codes(report))

    def test_fsck_does_not_write_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            status = _read_status(job_dir)
            status["succeeded"] = 5
            _write_status(job_dir, status)
            before = (job_dir / "indexes" / "status.json").read_text(encoding="utf-8")
            runner.fsck("job")
            after = (job_dir / "indexes" / "status.json").read_text(encoding="utf-8")
            self.assertEqual(before, after)


class ReconcileSupersedeTests(unittest.TestCase):
    def test_supersede_marks_attempt_superseded_and_requeues_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            attempt_id = _make_page_ambiguous(job_dir, 2)

            status = runner.resolve_ambiguous("job", 2, action="supersede")

            self.assertEqual(status.pending, 1)
            self.assertEqual(status.succeeded, 1)
            self.assertEqual(_read_status(job_dir)["pages"]["2"]["state"], "pending")
            attempt = json.loads((job_dir / "ledger" / f"{attempt_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(attempt["state"], "superseded")

    def test_accept_adopts_artifact_marks_succeeded_and_reassembles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            attempt_id = _make_page_ambiguous(job_dir, 2)
            (state_root / "out.md").unlink()

            status = runner.resolve_ambiguous("job", 2, action="accept")

            self.assertTrue(status.complete)
            self.assertEqual(_read_status(job_dir)["pages"]["2"]["state"], "succeeded")
            attempt = json.loads((job_dir / "ledger" / f"{attempt_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(attempt["state"], "superseded")
            self.assertEqual(attempt["resolution"], "accepted")
            output = (state_root / "out.md").read_text(encoding="utf-8")
            self.assertIn("Body 1", output)
            self.assertIn("Body 2", output)

    def test_accept_without_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            _make_page_ambiguous(job_dir, 2)
            (job_dir / "artifacts" / "pages" / "2.json").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.resolve_ambiguous("job", 2, action="accept")

    def test_resolve_rejects_non_ambiguous_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            with self.assertRaises(ValueError):
                runner.resolve_ambiguous("job", 1, action="supersede")

    def test_recovery_marks_inflight_ambiguous_with_recommended_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            runner = _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            _make_page_inflight_expired(job_dir, 2)

            runner.resume("job", allow_partial=True)
            payload = runner.reconcile("job")

            attempts = payload["ambiguous_attempts"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["page_number"], 2)
            self.assertIn("supersede", attempts[0]["recommended_actions"])
            self.assertIn("accept", attempts[0]["recommended_actions"])


class ReconcileCliTests(unittest.TestCase):
    def test_cli_reconcile_supersede_requeues_page(self) -> None:
        from paperscale.cli import main as paperscale_main

        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            _run_job(state_root, "job", pages=2)
            job_dir = state_root / "jobs" / "job"
            _make_page_ambiguous(job_dir, 2)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = paperscale_main(["reconcile", "job", "--state-root", str(state_root), "--supersede", "2"])
            self.assertEqual(code, 0)
            self.assertEqual(_read_status(job_dir)["pages"]["2"]["state"], "pending")


if __name__ == "__main__":
    unittest.main()
