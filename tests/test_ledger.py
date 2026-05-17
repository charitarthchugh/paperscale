from __future__ import annotations

import unittest

from paperscale.ledger import (
    AmbiguousAttemptError,
    InMemoryLedgerStore,
    Ledger,
    LedgerState,
    StaleEpochError,
)


class FakeProvider:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def send(self) -> None:
        self.events.append("provider_send")


class LedgerReservationTests(unittest.TestCase):
    def test_reservation_is_durably_persisted_before_provider_io(self) -> None:
        events: list[str] = []
        store = InMemoryLedgerStore(events=events)
        ledger = Ledger(store, now=lambda: 100.0)
        provider = FakeProvider(events)

        reservation = ledger.reserve_page(
            page_id="doc-1:page-1",
            provider_request_fingerprint="profile=v1:pagehash=abc",
            worker_id="worker-a",
            lease_seconds=30.0,
        )
        provider.send()
        ledger.mark_provider_call_started(
            attempt_id=reservation.attempt_id,
            worker_id="worker-a",
            epoch=reservation.epoch,
        )

        self.assertLess(events.index("write:reserved"), events.index("provider_send"))
        self.assertEqual(LedgerState.IN_FLIGHT, store.require_latest("doc-1:page-1").state)

    def test_stale_heartbeat_and_commit_are_rejected_after_higher_epoch_takeover(self) -> None:
        current_time = 100.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: current_time)

        stale = ledger.reserve_page(
            page_id="doc-1:page-2",
            provider_request_fingerprint="fp-1",
            worker_id="worker-a",
            lease_seconds=5.0,
        )
        current_time = 106.0
        ledger.recover_expired_leases(now=current_time)
        fresh = ledger.reserve_page(
            page_id="doc-1:page-2",
            provider_request_fingerprint="fp-1",
            worker_id="worker-b",
            lease_seconds=5.0,
        )

        self.assertGreater(fresh.epoch, stale.epoch)
        with self.assertRaises(StaleEpochError):
            ledger.heartbeat(stale.attempt_id, worker_id="worker-a", epoch=stale.epoch)
        with self.assertRaises(StaleEpochError):
            ledger.commit_success(
                stale.attempt_id,
                worker_id="worker-a",
                epoch=stale.epoch,
                result_pointer="results/doc-1/page-2.md",
            )


class LedgerDuplicatePolicyTests(unittest.TestCase):
    def test_ambiguous_attempt_is_not_retried_without_explicit_policy(self) -> None:
        current_time = 200.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: current_time)

        reservation = ledger.reserve_page(
            page_id="doc-2:page-1",
            provider_request_fingerprint="fp-2",
            worker_id="worker-a",
            lease_seconds=10.0,
        )
        ledger.mark_provider_call_started(
            reservation.attempt_id,
            worker_id="worker-a",
            epoch=reservation.epoch,
        )
        current_time = 211.0
        ledger.recover_expired_leases(now=current_time)

        latest = store.require_latest("doc-2:page-1")
        self.assertEqual(LedgerState.AMBIGUOUS, latest.state)
        with self.assertRaises(AmbiguousAttemptError):
            ledger.reserve_page(
                page_id="doc-2:page-1",
                provider_request_fingerprint="fp-2",
                worker_id="worker-b",
                lease_seconds=10.0,
            )


if __name__ == "__main__":
    unittest.main()
