from __future__ import annotations

import unittest

from paperscale.ledger import InMemoryLedgerStore, Ledger, LedgerState


class RecoveryTests(unittest.TestCase):
    def test_reserved_without_provider_start_requeues_after_lease_expiry(self) -> None:
        current_time = 10.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: current_time)

        first = ledger.reserve_page(
            page_id="doc-3:page-1",
            provider_request_fingerprint="fp-3",
            worker_id="worker-a",
            lease_seconds=3.0,
        )
        current_time = 14.0
        report = ledger.recover_expired_leases(now=current_time)

        latest = store.require_latest("doc-3:page-1")
        self.assertEqual(LedgerState.PENDING, latest.state)
        self.assertEqual(["doc-3:page-1"], report.requeued_pages)
        self.assertEqual([], report.ambiguous_pages)
        self.assertIsNone(latest.provider_call_started_at)

        second = ledger.reserve_page(
            page_id="doc-3:page-1",
            provider_request_fingerprint="fp-3",
            worker_id="worker-b",
            lease_seconds=3.0,
        )
        self.assertEqual(first.epoch + 1, second.epoch)
        self.assertNotEqual(first.attempt_id, second.attempt_id)

    def test_in_flight_without_committed_result_becomes_ambiguous_after_lease_expiry(self) -> None:
        current_time = 20.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: current_time)

        reservation = ledger.reserve_page(
            page_id="doc-4:page-9",
            provider_request_fingerprint="fp-4",
            worker_id="worker-a",
            lease_seconds=3.0,
        )
        ledger.mark_provider_call_started(
            reservation.attempt_id,
            worker_id="worker-a",
            epoch=reservation.epoch,
        )
        current_time = 24.0
        report = ledger.recover_expired_leases(now=current_time)

        latest = store.require_latest("doc-4:page-9")
        self.assertEqual(LedgerState.AMBIGUOUS, latest.state)
        self.assertEqual([], report.requeued_pages)
        self.assertEqual(["doc-4:page-9"], report.ambiguous_pages)
        self.assertIsNotNone(latest.provider_call_started_at)
        self.assertIsNone(latest.provider_response_committed_at)


if __name__ == "__main__":
    unittest.main()
