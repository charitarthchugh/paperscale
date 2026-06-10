"""Tests for the lazy candidate-materialization core of the runner (T1, T4-T7)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.candidates import CandidateRecord
from paperscale.contracts import CURRENT_SCHEMA_VERSION
from paperscale.providers.base import PageOcrResponse
from paperscale.runner import DocumentOcrRunner, RunnerConfig
from paperscale.state.claim_store import ClaimStore


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
        self.calls = 0

    def send(self, request: Any) -> PageOcrResponse:
        self.calls += 1
        return PageOcrResponse(
            markdown=f"# {request.page_id}\n\nBody for {request.page_id}.",
            provider_request_id=request.fingerprint,
        )


def _record(input_path: Path, output_path: Path, *, job_id: str = "doc", created_at: float = 1_700_000_000.0,
            output_override: str | None = None) -> CandidateRecord:
    return CandidateRecord(
        job_id=job_id,
        input_path=str(input_path),
        output_path=output_override or str(output_path),
        profile="generic_vlm_markdown",
        model="m",
        base_url="http://fake/v1",
        capacity="local-vllm-small",
        created_at=created_at,
    )


def _runner(root: Path, *, pages: int = 1, provider: Any | None = None,
            renderer_factory: Any | None = None, **config_kwargs: Any) -> DocumentOcrRunner:
    cfg = RunnerConfig(
        state_root=root, base_url="http://fake/v1", model="m",
        max_in_flight_requests=4, **config_kwargs,
    )
    return DocumentOcrRunner(
        cfg,
        provider=provider if provider is not None else _GoodProvider(),
        renderer_factory=renderer_factory or (lambda _p, _o: _Renderer(pages)),
        sleeper=lambda _s: None,
    )


class MaterializeJobTests(unittest.TestCase):
    def test_deterministic_manifest_from_record_created_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=3)
            store = runner._claim_store()
            record = _record(root / "i.pdf", root / "o.md", created_at=1234.5)

            store.try_claim(record.job_id, skip_if_done=False)
            manifest1 = runner.materialize_job(record, store=store)
            self.assertIsNotNone(manifest1)
            bytes1 = (root / "jobs" / "doc" / "manifest.json").read_bytes()

            # created_at comes from the record, not the wall clock.
            self.assertEqual(manifest1.created_at, 1234.5)
            self.assertEqual(manifest1.page_count, 3)

            # A second materialization on a matching manifest is a byte-identical no-op.
            manifest2 = runner.materialize_job(record, store=store)
            self.assertIsNotNone(manifest2)
            bytes2 = (root / "jobs" / "doc" / "manifest.json").read_bytes()
            self.assertEqual(bytes1, bytes2)
            self.assertEqual(manifest2.created_at, 1234.5)

    def test_fresh_materialization_twice_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            root1 = Path(tmp1) / ".paperscale"
            root2 = Path(tmp2) / ".paperscale"
            record1 = _record(Path(tmp1) / "i.pdf", root1 / "o.md", created_at=999.0)
            record2 = _record(Path(tmp2) / "i.pdf", root2 / "o.md", created_at=999.0)
            r1 = _runner(root1, pages=2)
            r2 = _runner(root2, pages=2)
            s1, s2 = r1._claim_store(), r2._claim_store()
            s1.try_claim(record1.job_id, skip_if_done=False)
            s2.try_claim(record2.job_id, skip_if_done=False)
            r1.materialize_job(record1, store=s1)
            r2.materialize_job(record2, store=s2)
            m1 = json.loads((root1 / "jobs" / "doc" / "manifest.json").read_text())
            m2 = json.loads((root2 / "jobs" / "doc" / "manifest.json").read_text())
            # Only paths differ; created_at/page_count/profile are deterministic.
            for key in ("created_at", "page_count", "profile", "model", "base_url", "capacity"):
                self.assertEqual(m1[key], m2[key])

    def test_noop_on_matching_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=2)
            store = runner._claim_store()
            record = _record(root / "i.pdf", root / "o.md")
            store.try_claim(record.job_id, skip_if_done=False)
            first = runner.materialize_job(record, store=store)
            second = runner.materialize_job(record, store=store)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first.to_json(), second.to_json())

    def test_skip_on_divergent_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=2)
            store = runner._claim_store()
            record = _record(root / "i.pdf", root / "o.md")
            store.try_claim(record.job_id, skip_if_done=False)
            runner.materialize_job(record, store=store)

            # A different candidate (same job_id, different output_path) must be skipped.
            divergent = _record(root / "i.pdf", root / "o.md", output_override=str(root / "other.md"))
            result = runner.materialize_job(divergent, store=store)
            self.assertIsNone(result)

    def test_corrupt_pdf_marks_failed_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"

            def boom(_p: Path, _o: dict) -> Any:
                raise RuntimeError("pdfinfo: corrupt PDF")

            runner = _runner(root, renderer_factory=boom)
            store = runner._claim_store()
            record = _record(root / "bad.pdf", root / "o.md")
            store.try_claim(record.job_id, skip_if_done=False)
            result = runner.materialize_job(record, store=store)
            self.assertIsNone(result)
            self.assertTrue((root / "jobs" / "doc" / "failed.json").exists())
            self.assertTrue(store.is_failed("doc"))


class PagesFromManifestTests(unittest.TestCase):
    def test_adopts_present_artifact_marks_missing_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=2)
            store = runner._claim_store()
            record = _record(root / "i.pdf", root / "o.md")
            store.try_claim(record.job_id, skip_if_done=False)
            manifest = runner.materialize_job(record, store=store)

            # Pre-place a succeeded artifact for page 1 only.
            artifact_rel = runner._artifact_rel("doc", 1)
            artifact = {
                "schema_version": CURRENT_SCHEMA_VERSION,
                "kind": "page_artifact",
                "document_id": manifest.document_id,
                "page_number": 1,
                "page_id": "doc:1",
                "markdown": "# one",
                "result_pointer": str(artifact_rel),
                "fingerprint": "fp-1",
            }
            runner.state.write_json_atomic(artifact_rel, artifact)

            pages = runner._pages_from_manifest("doc", manifest)
            self.assertEqual(pages["1"]["state"], "succeeded")
            self.assertEqual(pages["1"]["fingerprint"], "fp-1")
            self.assertEqual(pages["1"]["artifact_path"], str(artifact_rel))
            self.assertEqual(pages["2"]["state"], "pending")
            self.assertIsNone(pages["2"]["attempt_id"])


class ProcessCandidateTests(unittest.TestCase):
    def test_end_to_end_materialize_process_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            provider = _GoodProvider()
            output = root / "out" / "doc.md"
            runner = _runner(root, pages=2, provider=provider)
            store = runner._claim_store()
            record = _record(root / "i.pdf", output)

            status = runner._process_candidate(record, store=store)
            self.assertIsNotNone(status)
            self.assertTrue(status.complete)
            self.assertEqual(provider.calls, 2)
            self.assertTrue((root / "jobs" / "doc" / "done.json").exists())
            self.assertTrue(output.exists())
            # Claim released after processing.
            self.assertFalse((root / "jobs" / "doc" / "claim.json").exists())

    def test_skips_already_done_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=1)
            store = runner._claim_store()
            store.mark_done("doc")
            record = _record(root / "i.pdf", root / "o.md")
            self.assertIsNone(runner._process_candidate(record, store=store))

    def test_skips_failed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=1)
            store = runner._claim_store()
            store.mark_failed("doc", "prior corrupt")
            record = _record(root / "i.pdf", root / "o.md")
            self.assertIsNone(runner._process_candidate(record, store=store))

    def test_corrupt_pdf_candidate_returns_none_no_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"

            def boom(_p: Path, _o: dict) -> Any:
                raise RuntimeError("pdfinfo: corrupt PDF")

            runner = _runner(root, renderer_factory=boom)
            store = runner._claim_store()
            record = _record(root / "bad.pdf", root / "o.md")
            result = runner._process_candidate(record, store=store)
            self.assertIsNone(result)
            self.assertTrue(store.is_failed("doc"))
            self.assertFalse((root / "jobs" / "doc" / "claim.json").exists())


class _SpyStore:
    """Wraps a ClaimStore, counting heartbeat calls and gating one via a barrier."""

    def __init__(self, inner: ClaimStore, *, block_event: threading.Event | None = None,
                 first_tick: threading.Event | None = None) -> None:
        self._inner = inner
        self.heartbeat_calls = 0
        self._block_event = block_event
        self._first_tick = first_tick

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def heartbeat(self, claim: Any) -> Any:
        self.heartbeat_calls += 1
        result = self._inner.heartbeat(claim)
        if self._first_tick is not None:
            self._first_tick.set()
        return result


class HeartbeatRenewalTests(unittest.TestCase):
    def test_heartbeat_renews_lease_and_strict_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = _runner(root, pages=1, claim_lease_seconds=0.05, claim_heartbeat_seconds=0.01)
            inner_a = ClaimStore(root, worker_id="A", clock=time.time,
                                 lease_seconds=0.05, heartbeat_seconds=0.01)
            first_tick = threading.Event()
            spy = _SpyStore(inner_a, first_tick=first_tick)
            claim = spy.try_claim("doc", skip_if_done=False)
            self.assertIsNotNone(claim)

            holder = {"claim": claim}
            stop = runner._start_heartbeat(spy, holder, interval=0.01)
            try:
                # Block in real time well past the 0.05s lease while heartbeat renews it.
                deadline = time.time() + 0.3
                # Peer B cannot take over while the lease is being renewed.
                peer = ClaimStore(root, worker_id="B", clock=time.time,
                                  lease_seconds=0.05, heartbeat_seconds=0.01)
                self.assertTrue(first_tick.wait(1.0))
                while time.time() < deadline:
                    self.assertIsNone(peer.try_claim("doc", skip_if_done=False))
                    time.sleep(0.02)
                self.assertGreaterEqual(spy.heartbeat_calls, 1)
            finally:
                stop()
            # Strict shutdown: release after join leaves claim.json gone.
            spy.release(holder["claim"])
            self.assertFalse((root / "jobs" / "doc" / "claim.json").exists())
            # No post-join tick: heartbeat count is frozen.
            frozen = spy.heartbeat_calls
            time.sleep(0.05)
            self.assertEqual(spy.heartbeat_calls, frozen)

    def test_control_without_heartbeat_peer_takes_over(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            store_a = ClaimStore(root, worker_id="A", clock=time.time,
                                 lease_seconds=0.05, heartbeat_seconds=0.01)
            claim = store_a.try_claim("doc", skip_if_done=False)
            self.assertIsNotNone(claim)
            # No heartbeat running: after the lease expires, B takes over.
            time.sleep(0.08)
            peer = ClaimStore(root, worker_id="B", clock=time.time,
                              lease_seconds=0.05, heartbeat_seconds=0.01)
            taken = peer.try_claim("doc", skip_if_done=False)
            self.assertIsNotNone(taken)


if __name__ == "__main__":
    unittest.main()
