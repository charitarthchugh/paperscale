from __future__ import annotations

import unittest

from tests.harness.fakes import FakeClock, FakeProvider, RecordingStateStore
from tests.harness.imports import require_symbol


class LedgerRecoveryTests(unittest.TestCase):
    def test_provider_call_requires_durable_reservation_before_io(self) -> None:
        PageLedger = require_symbol("paperscale.ledger", "PageLedger")
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        store = RecordingStateStore()
        ledger = PageLedger(store=store, clock=FakeClock())
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        attempt = ledger.reserve_page_attempt(task, worker_id="w1", fingerprint="fp1", lease_seconds=30)
        provider = FakeProvider(ledger_probe=lambda: ledger.has_durable_reservation(attempt.attempt_id))
        provider.send({"attempt_id": attempt.attempt_id})
        self.assertIn(("attempt_reserved", attempt.attempt_id), store.mutations)

    def test_crash_after_reservation_before_provider_start_requeues_after_lease(self) -> None:
        PageLedger = require_symbol("paperscale.ledger", "PageLedger")
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        clock = FakeClock()
        ledger = PageLedger(store=RecordingStateStore(), clock=clock)
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        attempt = ledger.reserve_page_attempt(task, worker_id="w1", fingerprint="fp1", lease_seconds=10)
        clock.advance(11)
        recovered = ledger.recover_expired_attempts()
        self.assertEqual(recovered[attempt.attempt_id].state.value, "pending")
        self.assertFalse(recovered[attempt.attempt_id].duplicate_call_risk)

    def test_in_flight_without_committed_result_becomes_ambiguous_not_auto_retried(self) -> None:
        PageLedger = require_symbol("paperscale.ledger", "PageLedger")
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        clock = FakeClock()
        ledger = PageLedger(store=RecordingStateStore(), clock=clock)
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        attempt = ledger.reserve_page_attempt(task, worker_id="w1", fingerprint="fp1", lease_seconds=10)
        ledger.mark_provider_started(attempt.attempt_id)
        clock.advance(11)
        recovered = ledger.recover_expired_attempts()
        self.assertEqual(recovered[attempt.attempt_id].state.value, "ambiguous")
        self.assertTrue(recovered[attempt.attempt_id].duplicate_call_risk)
        self.assertFalse(ledger.next_retryable_page(allow_ambiguous=False))

    def test_stale_worker_heartbeat_and_commit_are_rejected_by_epoch(self) -> None:
        PageLedger = require_symbol("paperscale.ledger", "PageLedger")
        StaleEpochError = require_symbol("paperscale.ledger", "StaleEpochError")
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        ledger = PageLedger(store=RecordingStateStore(), clock=FakeClock())
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        old_attempt = ledger.reserve_page_attempt(task, worker_id="w1", fingerprint="fp1", lease_seconds=1)
        new_attempt = ledger.steal_expired_attempt(old_attempt.attempt_id, worker_id="w2")
        with self.assertRaises(StaleEpochError):
            ledger.heartbeat(old_attempt.attempt_id, epoch=old_attempt.epoch)
        with self.assertRaises(StaleEpochError):
            ledger.commit_success(old_attempt.attempt_id, epoch=old_attempt.epoch, result_pointer="old")
        ledger.commit_success(new_attempt.attempt_id, epoch=new_attempt.epoch, result_pointer="new")


if __name__ == "__main__":
    unittest.main()
