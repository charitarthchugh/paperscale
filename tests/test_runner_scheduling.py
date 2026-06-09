from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.providers.base import PageOcrResponse, ProviderError
from paperscale.resources import ResourceGovernor, ResourceKind
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


class _ScriptedProvider:
    """Raises ProviderError for the first `fail_first` calls, then succeeds."""

    name = "scripted"

    def __init__(self, *, fail_first: int = 0, always_fail: bool = False) -> None:
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.calls = 0

    def send(self, request: Any) -> PageOcrResponse:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise ProviderError(f"boom {self.calls}")
        return PageOcrResponse(markdown=f"# Page\n\nBody {request.page_id}", provider_request_id=request.fingerprint)


class RecordingGovernor(ResourceGovernor):
    def __init__(self) -> None:
        super().__init__()
        self.acquired: list[ResourceKind | str] = []

    def acquire(self, kind: ResourceKind | str):  # type: ignore[override]
        self.acquired.append(kind)
        return super().acquire(kind)


def _runner(state_root: Path, provider: Any, *, governor: Any = None, sleeper: Any = None, capacity: str = "local-vllm-small") -> DocumentOcrRunner:
    return DocumentOcrRunner(
        RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm", capacity=capacity),
        provider=provider,
        renderer_factory=lambda _path, _options: _FakeRenderer([b"page-1"]),
        governor=governor,
        sleeper=sleeper,
    )


class GovernedOrderingTests(unittest.TestCase):
    def test_page_processing_acquires_resources_in_global_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            governor = RecordingGovernor()
            runner = _runner(state_root, _ScriptedProvider(), governor=governor)
            status = runner.run(input_path=state_root / "in.pdf", output_path=state_root / "out.md", job_id="job")
            self.assertEqual(status.succeeded, 1)
            self.assertIn(ResourceKind.SCHEDULER, governor.acquired)
            self.assertIn(ResourceKind.RENDER, governor.acquired)
            self.assertIn(ResourceKind.PROVIDER, governor.acquired)
            self.assertIn(ResourceKind.PAGE_LEASE, governor.acquired)
            self.assertIn(ResourceKind.STATE_STORE, governor.acquired)
            self.assertLess(
                governor.acquired.index(ResourceKind.RENDER),
                governor.acquired.index(ResourceKind.PROVIDER),
            )


if __name__ == "__main__":
    unittest.main()
