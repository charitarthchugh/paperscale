from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.providers.base import PageOcrResponse, ProviderError
from paperscale.runner import DocumentOcrRunner, RunnerConfig


@dataclass(frozen=True)
class _Rendered:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class _Renderer:
    def __init__(self, n: int) -> None:
        self._pages = [f"page-{i}".encode() for i in range(1, n + 1)]

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def render_page(self, page_number: int) -> _Rendered:
        image = self._pages[page_number - 1]
        return _Rendered(page_number, image, hashlib.sha256(image).hexdigest())


class _ScriptedProvider:
    """Returns markdown per call; records the request decoding/fingerprint each time."""

    name = "scripted"

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def send(self, request: Any) -> PageOcrResponse:
        self.calls.append({"decoding": dict(request.decoding), "fingerprint": request.fingerprint})
        if not self._outputs:
            raise ProviderError("exhausted")
        return PageOcrResponse(markdown=self._outputs.pop(0), provider_request_id=request.fingerprint)


def _runner(state_root: Path, provider: Any, *, pages: int = 1, **cfg: Any) -> DocumentOcrRunner:
    base = dict(state_root=state_root, base_url="http://fake/v1", model="m", profile="deepseek_ocr_2", max_in_flight_requests=1)
    base.update(cfg)
    return DocumentOcrRunner(
        RunnerConfig(**base),
        provider=provider,
        renderer_factory=lambda _p, _o: _Renderer(pages),
        sleeper=lambda _s: None,
    )


class RetryCapTests(unittest.TestCase):
    def test_repeated_content_failure_is_capped_then_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider(["same same same same same same"] * 10)
            runner = _runner(root, provider, max_attempts=3)
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j", allow_partial=True)
            self.assertEqual(status.failed_terminal, 1)
            self.assertEqual(status.succeeded, 0)
            self.assertEqual(len(provider.calls), 3)  # capped at max_attempts
            self.assertFalse((root / "jobs" / "j" / "artifacts" / "pages" / "1.json").exists())

    def test_remediation_escalates_decoding_across_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider(["same same same same same same"] * 10)
            runner = _runner(root, provider, max_attempts=4)
            runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j", allow_partial=True)
            # repeated_ngram -> repetition_penalty raised on each subsequent attempt.
            first = provider.calls[0]["decoding"].get("repetition_penalty", 1.0)
            later = provider.calls[-1]["decoding"].get("repetition_penalty", 1.0)
            self.assertGreater(later, first)
            # Each remediated attempt changes the request fingerprint (intended).
            fingerprints = [c["fingerprint"] for c in provider.calls]
            self.assertEqual(len(set(fingerprints)), len(fingerprints))

    def test_content_failure_recovers_after_remediation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider(["same same same same same same", "same same same same same same", "# Good\n\nReal text body."])
            runner = _runner(root, provider, max_attempts=8)
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j")
            self.assertEqual(status.succeeded, 1)
            self.assertEqual(len(provider.calls), 3)


def _png(shade: int) -> bytes:
    from PIL import Image
    import io as _io

    buf = _io.BytesIO()
    Image.new("L", (64, 64), color=shade).save(buf, "PNG")
    return buf.getvalue()


class _PngRenderer:
    def __init__(self, n: int, shade: int) -> None:
        self._n = n
        self._png = _png(shade)

    @property
    def page_count(self) -> int:
        return self._n

    def render_page(self, page_number: int) -> _Rendered:
        return _Rendered(page_number, self._png, hashlib.sha256(self._png).hexdigest())


class BlankPageTests(unittest.TestCase):
    def test_blank_page_with_empty_ocr_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider([""])  # model returns empty for a blank page
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", profile="deepseek_ocr_2", max_in_flight_requests=1),
                provider=provider,
                renderer_factory=lambda _p, _o: _PngRenderer(1, shade=255),  # all-white = blank
                sleeper=lambda _s: None,
            )
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j")
            self.assertEqual(status.succeeded, 1)
            self.assertTrue(status.complete)
            self.assertEqual(len(provider.calls), 1)  # accepted on first sight, no re-rolls

    def test_inky_page_with_empty_ocr_is_not_treated_as_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider([""] * 10)  # empty on a content-bearing page
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", profile="deepseek_ocr_2", max_in_flight_requests=1, max_attempts=3),
                provider=provider,
                renderer_factory=lambda _p, _o: _PngRenderer(1, shade=0),  # all-black = lots of ink
                sleeper=lambda _s: None,
            )
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j", allow_partial=True)
            self.assertEqual(status.succeeded, 0)
            self.assertEqual(status.failed_terminal, 1)


class AdoptThenRequeueTests(unittest.TestCase):
    def test_resume_adopts_matching_artifact_without_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _ScriptedProvider(["# Page\n\nBody text here."])
            runner = _runner(root, provider, pages=1)
            runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j")
            job_dir = root / "jobs" / "j"

            # Force the page back into a crashed in_flight state (lease expired) while
            # the durable artifact survives with a matching fingerprint.
            status_idx = json.loads((job_dir / "indexes" / "status.json").read_text())
            artifact = json.loads((job_dir / "artifacts" / "pages" / "1.json").read_text())
            attempt_id = status_idx["pages"]["1"]["attempt_id"]
            ledger_path = job_dir / "ledger" / f"{attempt_id}.json"
            attempt = json.loads(ledger_path.read_text())
            attempt.update({"state": "in_flight", "result_pointer": None, "lease_expires_at": 1.0, "fingerprint": artifact["fingerprint"]})
            ledger_path.write_text(json.dumps(attempt, sort_keys=True))
            status_idx["pages"]["1"] = {"state": "in_flight", "epoch": 1, "attempt_id": attempt_id, "fingerprint": artifact["fingerprint"]}
            (job_dir / "indexes" / "status.json").write_text(json.dumps(status_idx, sort_keys=True))

            runner.provider = _ScriptedProvider([])  # any send would raise -> proves no call
            status = runner.resume("j")
            self.assertTrue(status.complete)


class SettledReportingTests(unittest.TestCase):
    def test_terminal_failure_makes_job_settled_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            # page 1 refuses (terminal on first sight), page 2 succeeds.
            provider = _ScriptedProvider(["I'm sorry, I can't assist with that.", "# Page 2\n\nReal body."])
            runner = _runner(root, provider, pages=2)
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j", allow_partial=True)
            self.assertEqual(status.failed_terminal, 1)
            self.assertEqual(status.succeeded, 1)
            self.assertFalse(status.complete)
            self.assertTrue(status.settled)
            self.assertTrue(status.has_terminal_failures)


if __name__ == "__main__":
    unittest.main()
