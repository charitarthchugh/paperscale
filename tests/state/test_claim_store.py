from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paperscale.state.claim_store import ClaimStore


class ClaimStoreTests(unittest.TestCase):
    def _store(self, root: Path, now_box: list[float], worker: str) -> ClaimStore:
        return ClaimStore(
            root,
            worker_id=worker,
            clock=lambda: now_box[0],
            lease_seconds=60.0,
            heartbeat_seconds=20.0,
        )

    def test_fresh_claim_succeeds_and_held_claim_blocks_other_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            b = self._store(root, now, "worker-b")

            claim = a.try_claim("job-1")
            self.assertIsNotNone(claim)
            self.assertEqual(claim.worker_id, "worker-a")
            # b cannot co-own a fresh, unexpired claim (O_EXCL exclusivity).
            self.assertIsNone(b.try_claim("job-1"))

    def test_done_marker_skips_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            self.assertFalse(a.is_done("job-2"))
            a.mark_done("job-2")
            self.assertTrue(a.is_done("job-2"))
            self.assertIsNone(a.try_claim("job-2"))

    def test_expired_lease_is_reclaimed_with_higher_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            b = self._store(root, now, "worker-b")

            first = a.try_claim("job-3")
            self.assertEqual(first.epoch, 1)
            # a goes silent; lease expires.
            now[0] = 200.0
            reclaimed = b.try_claim("job-3")
            self.assertIsNotNone(reclaimed)
            self.assertEqual(reclaimed.worker_id, "worker-b")
            self.assertGreater(reclaimed.epoch, first.epoch)

    def test_heartbeat_extends_lease_and_prevents_takeover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            b = self._store(root, now, "worker-b")

            claim = a.try_claim("job-4")
            now[0] = 150.0
            a.heartbeat(claim)  # extends lease to 150+60=210
            now[0] = 200.0
            self.assertIsNone(b.try_claim("job-4"))

    def test_release_allows_reclaim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            b = self._store(root, now, "worker-b")

            claim = a.try_claim("job-5")
            a.release(claim)
            again = b.try_claim("job-5")
            self.assertIsNotNone(again)
            self.assertEqual(again.worker_id, "worker-b")

    def test_mark_failed_writes_discoverable_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            self.assertFalse(a.is_failed("job-6"))
            a.mark_failed("job-6", "boom")
            self.assertTrue(a.is_failed("job-6"))
            marker = root / "jobs" / "job-6" / "failed.json"
            self.assertTrue(marker.exists())

    def test_clear_failed_removes_marker_and_is_noop_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            a.mark_failed("job-7", "boom")
            self.assertTrue(a.is_failed("job-7"))
            a.clear_failed("job-7")
            self.assertFalse(a.is_failed("job-7"))
            # Clearing an absent marker is a no-op (no error).
            a.clear_failed("job-7")
            self.assertFalse(a.is_failed("job-7"))

    def test_terminal_outcome_none_when_no_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            self.assertIsNone(a.terminal_outcome("job-8"))

    def test_terminal_outcome_failed_when_only_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            a.mark_failed("job-9", "boom")
            self.assertEqual(a.terminal_outcome("job-9"), "failed")

    def test_terminal_outcome_done_when_only_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            a.mark_done("job-10")
            self.assertEqual(a.terminal_outcome("job-10"), "done")

    def test_terminal_outcome_done_precedence_when_both_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            a.mark_failed("job-11", "boom")
            a.mark_done("job-11")
            self.assertTrue(a.is_done("job-11"))
            self.assertTrue(a.is_failed("job-11"))
            self.assertEqual(a.terminal_outcome("job-11"), "done")

    def test_failed_marker_does_not_block_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = [100.0]
            a = self._store(root, now, "worker-a")
            a.mark_failed("job-12", "boom")
            # try_claim only auto-skips on is_done, NOT is_failed.
            claim = a.try_claim("job-12")
            self.assertIsNotNone(claim)
            self.assertEqual(claim.worker_id, "worker-a")


if __name__ == "__main__":
    unittest.main()
