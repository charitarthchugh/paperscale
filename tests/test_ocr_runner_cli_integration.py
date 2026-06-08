from __future__ import annotations

import contextlib
import io
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.providers.base import PageOcrResponse, ProviderError


@dataclass(frozen=True)
class FakeRenderedPage:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class FakeRenderer:
    def __init__(self, pages: list[bytes]) -> None:
        self._pages = pages
        self.rendered: list[int] = []

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def render_page(self, page_number: int) -> FakeRenderedPage:
        import hashlib

        self.rendered.append(page_number)
        image = self._pages[page_number - 1]
        return FakeRenderedPage(page_number, image, hashlib.sha256(image).hexdigest())


class RecordingProvider:
    name = "fake-provider"

    def __init__(self, job_dir: Path, responses: list[str] | None = None) -> None:
        self.job_dir = job_dir
        self.responses = responses or ["# Page 1\n\nAlpha", "# Page 2\n\nBeta"]
        self.calls: list[Any] = []

    def send(self, request: Any) -> PageOcrResponse:
        ledger_records = [json.loads(path.read_text()) for path in (self.job_dir / "ledger").glob("*.json")]
        matching = [record for record in ledger_records if record["page_id"] == request.page_id]
        assert matching, "provider called before durable reservation"
        assert matching[-1]["state"] == "in_flight", "provider called before durable in_flight mark"
        assert matching[-1]["fingerprint"] == request.fingerprint
        self.calls.append(request)
        if not self.responses:
            raise ProviderError("fake exhausted")
        return PageOcrResponse(markdown=self.responses.pop(0), provider_request_id=request.fingerprint)


class OcrRunnerCliIntegrationTests(unittest.TestCase):
    def test_runner_writes_durable_artifacts_indexes_and_final_markdown_for_two_pages(self) -> None:
        from paperscale.runner import DocumentOcrRunner, RunnerConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            job_dir = state_root / "jobs" / "job-two"
            renderer = FakeRenderer([b"page-one", b"page-two"])
            provider = RecordingProvider(job_dir)
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
                provider=provider,
                renderer_factory=lambda _path, _options: renderer,
            )

            status = runner.run(input_path=root / "input.pdf", output_path=root / "out.md", job_id="job-two")

            self.assertEqual(status.succeeded, 2)
            self.assertEqual(status.pages_total, 2)
            self.assertEqual(renderer.rendered, [1, 2])
            self.assertEqual(len(provider.calls), 2)
            markdown = (root / "out.md").read_text(encoding="utf-8")
            self.assertIn("# Page 1", markdown)
            self.assertIn("<!-- page-break -->", markdown)
            self.assertTrue((job_dir / "manifest.json").exists())
            self.assertTrue((job_dir / "artifacts" / "pages" / "1.json").exists())
            self.assertTrue((job_dir / "artifacts" / "pages" / "2.json").exists())
            index = json.loads((job_dir / "indexes" / "status.json").read_text())
            self.assertEqual(index["succeeded"], 2)
            self.assertEqual(index["pages"]["1"]["state"], "succeeded")

    def test_status_and_reconcile_read_compact_indexes_without_artifact_tree_scan(self) -> None:
        from paperscale.runner import DocumentOcrRunner, RunnerConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            job_dir = state_root / "jobs" / "job-index-only"
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
                provider=RecordingProvider(job_dir, ["# Only\n\nBody"]),
                renderer_factory=lambda _path, _options: FakeRenderer([b"page"]),
            )
            runner.run(input_path=root / "input.pdf", output_path=root / "out.md", job_id="job-index-only")
            (job_dir / "artifacts").rename(job_dir / "artifacts.off")

            status = runner.status("job-index-only")
            reconcile = runner.reconcile("job-index-only")

            self.assertEqual(status.succeeded, 1)
            self.assertEqual(reconcile["ambiguous_attempts"], [])

    def test_quality_failure_never_commits_success_artifact(self) -> None:
        from paperscale.runner import DocumentOcrRunner, RunnerConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            job_dir = state_root / "jobs" / "job-quality"
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
                provider=RecordingProvider(job_dir, ["same same same same same same"]),
                renderer_factory=lambda _path, _options: FakeRenderer([b"page"]),
            )

            status = runner.run(input_path=root / "input.pdf", output_path=root / "out.md", job_id="job-quality", allow_partial=True)

            self.assertEqual(status.succeeded, 0)
            self.assertEqual(status.failed_retryable, 1)
            self.assertFalse((job_dir / "artifacts" / "pages" / "1.json").exists())
            self.assertFalse((root / "out.md").exists())

    def test_fsck_scans_without_writing_and_repair_index_rebuilds_compact_indexes(self) -> None:
        from paperscale.runner import DocumentOcrRunner, RunnerConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_root = root / ".paperscale"
            job_dir = state_root / "jobs" / "job-repair"
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm"),
                provider=RecordingProvider(job_dir, ["# Only\n\nBody"]),
                renderer_factory=lambda _path, _options: FakeRenderer([b"page"]),
            )
            runner.run(input_path=root / "input.pdf", output_path=root / "out.md", job_id="job-repair")
            (job_dir / "artifacts" / "pages" / "1.json").unlink()
            before = (job_dir / "indexes" / "status.json").read_text(encoding="utf-8")

            fsck = runner.fsck("job-repair")
            after_fsck = (job_dir / "indexes" / "status.json").read_text(encoding="utf-8")
            repaired = runner.repair_index("job-repair")

            self.assertTrue(fsck["scanned"])
            self.assertIn("missing_artifact", fsck["issues"][0]["code"])
            self.assertEqual(before, after_fsck)
            self.assertEqual(repaired.succeeded, 0)
            self.assertEqual(repaired.failed_retryable, 1)

    def test_cli_doctor_provider_passes_for_served_model_and_fails_for_missing_model(self) -> None:
        from paperscale.cli import main as paperscale_main

        port = _free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paperscale.mock_api.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                "mock-vlm",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_models(port)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                ok = paperscale_main(
                    [
                        "doctor",
                        "provider",
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--model",
                        "mock-vlm",
                    ]
                )
            self.assertEqual(ok, 0)
            self.assertIn("compatible: true", stdout.getvalue())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                bad = paperscale_main(
                    [
                        "doctor",
                        "provider",
                        "--base-url",
                        f"http://127.0.0.1:{port}",
                        "--model",
                        "missing-vlm",
                    ]
                )
            self.assertEqual(bad, 1)
            self.assertIn("compatible: false", stdout.getvalue())
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr:
                process.stderr.close()

    def test_full_cli_run_smoke_against_mock_api_subprocess(self) -> None:
        from paperscale.cli import main as paperscale_main

        port = _free_port()
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paperscale.mock_api.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--model",
                "mock-vlm",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_models(port)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                pdf_path = root / "sample.pdf"
                pdf_path.write_bytes(_two_page_pdf_bytes())
                out_path = root / "out.md"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = paperscale_main(
                        [
                            "run",
                            "--input",
                            str(pdf_path),
                            "--output",
                            str(out_path),
                            "--job-id",
                            "smoke",
                            "--state-root",
                            str(root / ".paperscale"),
                            "--base-url",
                            f"http://127.0.0.1:{port}",
                            "--model",
                            "mock-vlm",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertIn("succeeded=2/2", stdout.getvalue())
                self.assertIn("Mock OCR Page", out_path.read_text(encoding="utf-8"))
                self.assertTrue((root / ".paperscale" / "jobs" / "smoke" / "indexes" / "status.json").exists())
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr:
                process.stderr.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_models(port: int) -> None:
    deadline = time.monotonic() + 10
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - localhost test URL
                response.read()
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.1)
    raise AssertionError("mock API server did not become ready")


def _two_page_pdf_bytes() -> bytes:
    objects: list[tuple[int, str]] = []

    def obj(number: int, body: str) -> None:
        objects.append((number, body))

    stream1 = "BT /F1 24 Tf 72 720 Td (Page One) Tj ET"
    stream2 = "BT /F1 24 Tf 72 720 Td (Page Two) Tj ET"
    obj(1, "<< /Type /Catalog /Pages 2 0 R >>")
    obj(2, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>")
    obj(3, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>")
    obj(4, "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>")
    obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    obj(6, f"<< /Length {len(stream1)} >>\nstream\n{stream1}\nendstream")
    obj(7, f"<< /Length {len(stream2)} >>\nstream\n{stream2}\nendstream")
    output = b"%PDF-1.4\n"
    offsets = {0: 0}
    for number, body in objects:
        offsets[number] = len(output)
        output += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref_start = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for number in range(1, len(objects) + 1):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode()
    return output


if __name__ == "__main__":
    unittest.main()
