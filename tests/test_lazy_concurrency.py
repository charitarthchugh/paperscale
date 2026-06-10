"""In-process concurrency approximation for the lazy plan->work drain.

NOTE: This is an *in-process* approximation of real multi-process ``paperscale
work`` concurrency. Two ``DocumentOcrRunner`` instances (distinct ``worker_id``)
share one state root and one ``candidates/*.jsonl`` manifest, draining it from
separate threads. The ``O_EXCL`` ClaimStore is the same filesystem substrate a
real subprocess would use, so claim races, done-precedence, and per-job
single-ownership are exercised faithfully; only the OS-process boundary is
simulated. Small page counts keep it deterministic and fast.
"""

from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.candidates import plan_candidates
from paperscale.providers.base import PageOcrResponse
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

    def send(self, request: Any) -> PageOcrResponse:
        return PageOcrResponse(
            markdown=f"# {request.page_id}\n\nbody",
            provider_request_id=request.fingerprint,
        )


def _runner(root: Path, worker_id: str) -> DocumentOcrRunner:
    return DocumentOcrRunner(
        RunnerConfig(
            state_root=root,
            base_url="http://fake/v1",
            model="m",
            worker_id=worker_id,
            max_in_flight_requests=4,
            server_max_num_seqs=4,
        ),
        provider=_GoodProvider(),
        renderer_factory=lambda _p, _o: _Renderer(2),
        sleeper=lambda _s: None,
    )


class LazyConcurrencyTests(unittest.TestCase):
    def test_two_workers_drain_one_manifest_without_double_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            in_dir = base / "in"
            in_dir.mkdir(parents=True)
            inputs = []
            for name in ("a", "b", "c", "d", "e", "f"):
                p = in_dir / f"{name}.pdf"
                p.write_text("%PDF-1.4 placeholder\n", encoding="utf-8")
                inputs.append(p)
            out_dir = base / "out"
            plan_candidates(
                inputs,
                output_dir=out_dir,
                state_root=root,
                profile="generic_vlm_markdown",
                model="m",
                base_url="http://fake/v1",
                capacity="local-vllm-small",
            )

            errors: list[BaseException] = []
            barrier = threading.Barrier(2)

            def drain(worker_id: str) -> None:
                runner = _runner(root, worker_id)
                try:
                    barrier.wait()
                    runner.work()
                except BaseException as exc:  # noqa: BLE001 - surface any claim-race crash
                    errors.append(exc)

            threads = [
                threading.Thread(target=drain, args=("worker-1",)),
                threading.Thread(target=drain, args=("worker-2",)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"worker crashed under claim race: {errors}")
            store = _runner(root, "verify")._claim_store()
            for name in ("a", "b", "c", "d", "e", "f"):
                self.assertTrue(store.is_done(name), f"{name} not done")
                self.assertTrue((out_dir / f"{name}.md").exists(), f"{name}.md missing")
                # exactly one done.json marker per job (single-ownership completion)
                done_markers = list((root / "jobs" / name).glob("done.json"))
                self.assertEqual(len(done_markers), 1)
                # page artifacts present
                artifacts = list((root / "jobs" / name / "artifacts" / "pages").glob("*.json"))
                self.assertEqual(len(artifacts), 2, f"{name} missing page artifacts")


if __name__ == "__main__":
    unittest.main()
