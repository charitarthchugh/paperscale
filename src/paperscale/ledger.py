"""Ledger reservation, lease, and crash-recovery semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import uuid


class LedgerState(StrEnum):
    PENDING = "pending"
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    AMBIGUOUS = "ambiguous"
    SUPERSEDED = "superseded"


class StaleEpochError(RuntimeError):
    pass


class AmbiguousAttemptError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LedgerAttempt:
    attempt_id: str
    page_id: str
    provider_request_fingerprint: str
    worker_id: str
    epoch: int
    lease_expires_at: float
    heartbeat_at: float
    state: LedgerState
    provider_call_started_at: float | None = None
    provider_response_committed_at: float | None = None
    result_pointer: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    requeued_pages: list[str]
    ambiguous_pages: list[str]


class InMemoryLedgerStore:
    """Small fake durable store used by tests and early invariant harnesses."""

    def __init__(self, *, events: list[str] | None = None) -> None:
        self._latest_by_page: dict[str, LedgerAttempt] = {}
        self._by_attempt: dict[str, LedgerAttempt] = {}
        self._max_epoch_by_page: dict[str, int] = {}
        self.events = events if events is not None else []

    def write(self, attempt: LedgerAttempt) -> LedgerAttempt:
        self._latest_by_page[attempt.page_id] = attempt
        self._by_attempt[attempt.attempt_id] = attempt
        self._max_epoch_by_page[attempt.page_id] = max(self._max_epoch_by_page.get(attempt.page_id, 0), attempt.epoch)
        self.events.append(f"write:{attempt.state.value}")
        return attempt

    def require_latest(self, page_id: str) -> LedgerAttempt:
        return self._latest_by_page[page_id]

    def require_attempt(self, attempt_id: str) -> LedgerAttempt:
        return self._by_attempt[attempt_id]

    def next_epoch(self, page_id: str) -> int:
        return self._max_epoch_by_page.get(page_id, 0) + 1

    def all_latest(self) -> list[LedgerAttempt]:
        return list(self._latest_by_page.values())


class Ledger:
    def __init__(self, store: InMemoryLedgerStore, *, now) -> None:
        self.store = store
        self._now = now

    def reserve_page(
        self,
        *,
        page_id: str,
        provider_request_fingerprint: str,
        worker_id: str,
        lease_seconds: float,
        retry_ambiguous: bool = False,
    ) -> LedgerAttempt:
        try:
            latest = self.store.require_latest(page_id)
        except KeyError:
            latest = None
        if latest and latest.state == LedgerState.AMBIGUOUS and not retry_ambiguous:
            raise AmbiguousAttemptError(f"ambiguous attempt for {page_id} requires explicit retry policy")
        if latest and latest.state in {LedgerState.RESERVED, LedgerState.IN_FLIGHT, LedgerState.SUCCEEDED}:
            raise RuntimeError(f"page {page_id} is not reservable from state {latest.state}")

        current = float(self._now())
        attempt = LedgerAttempt(
            attempt_id=str(uuid.uuid4()),
            page_id=page_id,
            provider_request_fingerprint=provider_request_fingerprint,
            worker_id=worker_id,
            epoch=self.store.next_epoch(page_id),
            lease_expires_at=current + lease_seconds,
            heartbeat_at=current,
            state=LedgerState.RESERVED,
        )
        return self.store.write(attempt)

    def mark_provider_call_started(self, attempt_id: str, *, worker_id: str, epoch: int) -> LedgerAttempt:
        attempt = self._require_current_attempt(attempt_id, worker_id, epoch)
        updated = replace(attempt, state=LedgerState.IN_FLIGHT, provider_call_started_at=float(self._now()))
        return self.store.write(updated)

    def heartbeat(self, attempt_id: str, *, worker_id: str, epoch: int, lease_seconds: float = 30.0) -> LedgerAttempt:
        attempt = self._require_current_attempt(attempt_id, worker_id, epoch)
        current = float(self._now())
        return self.store.write(replace(attempt, heartbeat_at=current, lease_expires_at=current + lease_seconds))

    def commit_success(self, attempt_id: str, *, worker_id: str, epoch: int, result_pointer: str) -> LedgerAttempt:
        attempt = self._require_current_attempt(attempt_id, worker_id, epoch)
        updated = replace(
            attempt,
            state=LedgerState.SUCCEEDED,
            provider_response_committed_at=float(self._now()),
            result_pointer=result_pointer,
        )
        return self.store.write(updated)

    def recover_expired_leases(self, *, now: float | None = None) -> RecoveryReport:
        current = float(self._now() if now is None else now)
        requeued: list[str] = []
        ambiguous: list[str] = []
        for attempt in list(self.store.all_latest()):
            if attempt.lease_expires_at > current:
                continue
            if attempt.state == LedgerState.RESERVED and attempt.provider_call_started_at is None:
                self.store.write(replace(attempt, state=LedgerState.PENDING))
                requeued.append(attempt.page_id)
            elif attempt.state == LedgerState.IN_FLIGHT and attempt.provider_response_committed_at is None:
                self.store.write(replace(attempt, state=LedgerState.AMBIGUOUS))
                ambiguous.append(attempt.page_id)
        return RecoveryReport(requeued_pages=requeued, ambiguous_pages=ambiguous)

    def _require_current_attempt(self, attempt_id: str, worker_id: str, epoch: int) -> LedgerAttempt:
        attempt = self.store.require_attempt(attempt_id)
        latest = self.store.require_latest(attempt.page_id)
        if latest.attempt_id != attempt_id or attempt.worker_id != worker_id or attempt.epoch != epoch:
            raise StaleEpochError(f"stale attempt {attempt_id}")
        return attempt
