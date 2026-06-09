from __future__ import annotations

import hashlib
import json
import tempfile
import threading
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
        self._n = n

    @property
    def page_count(self) -> int:
        return self._n

    def render_page(self, page_number: int) -> _Rendered:
        image = f"page-{page_number}".encode()
        return _Rendered(page_number, image, hashlib.sha256(image).hexdigest())


class _GoodProvider:
    name = "good"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self._active = 0
        self.calls = 0

    def send(self, request: Any) -> PageOcrResponse:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            self.calls += 1
        try:
            return PageOcrResponse(markdown=f"# {request.page_id}\n\nBody for {request.page_id}.", provider_request_id=request.fingerprint)
        finally:
            with self._lock:
                self._active -= 1


class _AlwaysFail:
    name = "fail"

    def send(self, request: Any) -> PageOcrResponse:
        raise ProviderError("down")


class ConcurrencyPoolTests(unittest.TestCase):
    def test_many_pages_all_succeed_and_index_is_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _GoodProvider()
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", max_in_flight_requests=8, server_max_num_seqs=8),
                provider=provider,
                renderer_factory=lambda _p, _o: _Renderer(40),
                sleeper=lambda _s: None,
            )
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="big")
            self.assertEqual(status.succeeded, 40)
            self.assertTrue(status.complete)
            self.assertEqual(provider.calls, 40)  # each page sent exactly once
            # The single index writer folded every event into a consistent rollup.
            idx = json.loads((root / "jobs" / "big" / "indexes" / "status.json").read_text())
            self.assertEqual(idx["succeeded"], 40)
            self.assertEqual(sum(1 for p in idx["pages"].values() if p["state"] == "succeeded"), 40)

    def test_in_flight_semaphore_bounds_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _GoodProvider()
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", max_in_flight_requests=3),
                provider=provider,
                renderer_factory=lambda _p, _o: _Renderer(30),
                sleeper=lambda _s: None,
            )
            runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="bounded")
            # Never more than max_in_flight_requests concurrent provider calls.
            self.assertLessEqual(provider.max_concurrent, 3)


class WorkClaimTests(unittest.TestCase):
    def test_work_resumes_incomplete_job_then_skips_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            # First pass: the server is down, circuit opens, job left incomplete.
            failing = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", max_in_flight_requests=1),
                provider=_AlwaysFail(),
                renderer_factory=lambda _p, _o: _Renderer(2),
                sleeper=lambda _s: None,
            )
            first = failing.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="wjob", allow_partial=True)
            self.assertFalse(first.complete)

            # A worker process with a healthy provider claims and finishes it.
            worker = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", worker_id="worker-2", max_in_flight_requests=4),
                provider=_GoodProvider(),
                renderer_factory=lambda _p, _o: _Renderer(2),
                sleeper=lambda _s: None,
            )
            results = worker.work()
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].complete)
            self.assertTrue((root / "jobs" / "wjob" / "done.json").exists())

            # A subsequent worker skips the done job (no claimable work).
            self.assertEqual(worker.work(), [])


if __name__ == "__main__":
    unittest.main()
