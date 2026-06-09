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


if __name__ == "__main__":
    unittest.main()
