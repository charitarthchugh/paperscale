"""Ledger reservation, lease, and crash-recovery semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import uuid

from paperscale.contracts import PageAttemptState, PageTask


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


@dataclass(frozen=True, slots=True)
class PageAttempt:
    attempt_id: str
    task: PageTask
    worker_id: str
    fingerprint: str
    epoch: int
    lease_expires_at: float
    heartbeat_at: float
    state: PageAttemptState
    duplicate_call_risk: bool = False
    provider_started_at: float | None = None
    result_pointer: str | None = None


class PageLedger:
    """High-level page ledger API used by crash/recovery acceptance tests."""

    def __init__(self, *, store, clock) -> None:
        self.store = store
        self.clock = clock
        self._attempts: dict[str, PageAttempt] = {}
        self._latest_by_page: dict[str, str] = {}
        self._max_epoch_by_page: dict[str, int] = {}

    def reserve_page_attempt(
        self,
        task: PageTask,
        *,
        worker_id: str,
        fingerprint: str,
        lease_seconds: float,
    ) -> PageAttempt:
        latest = self._latest_attempt_for_page(task.page_id)
        if latest is not None and latest.state == PageAttemptState.AMBIGUOUS:
            raise AmbiguousAttemptError(f"ambiguous attempt for {task.page_id} requires explicit retry")
        epoch = self._max_epoch_by_page.get(task.page_id, 0) + 1
        now = float(self.clock.now())
        attempt = PageAttempt(
            attempt_id=str(uuid.uuid4()),
            task=task,
            worker_id=worker_id,
            fingerprint=fingerprint,
            epoch=epoch,
            lease_expires_at=now + lease_seconds,
            heartbeat_at=now,
            state=PageAttemptState.RESERVED,
        )
        self._write_attempt(attempt, mutation="attempt_reserved")
        return attempt

    def has_durable_reservation(self, attempt_id: str) -> bool:
        return ("attempt_reserved", attempt_id) in getattr(self.store, "mutations", [])

    def mark_provider_started(self, attempt_id: str) -> PageAttempt:
        attempt = self._require_current_attempt(attempt_id, epoch=None)
        updated = replace(
            attempt,
            state=PageAttemptState.IN_FLIGHT,
            provider_started_at=float(self.clock.now()),
        )
        self._write_attempt(updated, mutation="attempt_in_flight")
        return updated

    def recover_expired_attempts(self) -> dict[str, PageAttempt]:
        now = float(self.clock.now())
        recovered: dict[str, PageAttempt] = {}
        for attempt_id, attempt in list(self._attempts.items()):
            if self._latest_by_page.get(attempt.task.page_id) != attempt_id:
                continue
            if attempt.lease_expires_at > now:
                continue
            if attempt.state == PageAttemptState.RESERVED and attempt.provider_started_at is None:
                updated = replace(attempt, state=PageAttemptState.PENDING, duplicate_call_risk=False)
                self._write_attempt(updated, mutation="attempt_requeued")
                recovered[attempt_id] = updated
            elif attempt.state == PageAttemptState.IN_FLIGHT and attempt.result_pointer is None:
                updated = replace(attempt, state=PageAttemptState.AMBIGUOUS, duplicate_call_risk=True)
                self._write_attempt(updated, mutation="attempt_ambiguous")
                recovered[attempt_id] = updated
        return recovered

    def next_retryable_page(self, *, allow_ambiguous: bool = False) -> PageTask | None:
        for attempt_id in self._latest_by_page.values():
            attempt = self._attempts[attempt_id]
            if attempt.state == PageAttemptState.PENDING:
                return attempt.task
            if allow_ambiguous and attempt.state == PageAttemptState.AMBIGUOUS:
                return attempt.task
        return None

    def steal_expired_attempt(self, attempt_id: str, *, worker_id: str) -> PageAttempt:
        old = self._attempts[attempt_id]
        epoch = self._max_epoch_by_page.get(old.task.page_id, old.epoch) + 1
        now = float(self.clock.now())
        stolen = PageAttempt(
            attempt_id=str(uuid.uuid4()),
            task=old.task,
            worker_id=worker_id,
            fingerprint=old.fingerprint,
            epoch=epoch,
            lease_expires_at=now + max(old.lease_expires_at - old.heartbeat_at, 1.0),
            heartbeat_at=now,
            state=PageAttemptState.RESERVED,
        )
        self._write_attempt(stolen, mutation="attempt_stolen")
        return stolen

    def heartbeat(self, attempt_id: str, *, epoch: int, lease_seconds: float = 30.0) -> PageAttempt:
        attempt = self._require_current_attempt(attempt_id, epoch=epoch)
        now = float(self.clock.now())
        updated = replace(attempt, heartbeat_at=now, lease_expires_at=now + lease_seconds)
        self._write_attempt(updated, mutation="attempt_heartbeat")
        return updated

    def commit_success(self, attempt_id: str, *, epoch: int, result_pointer: str) -> PageAttempt:
        attempt = self._require_current_attempt(attempt_id, epoch=epoch)
        updated = replace(attempt, state=PageAttemptState.SUCCEEDED, result_pointer=result_pointer)
        self._write_attempt(updated, mutation="attempt_succeeded")
        return updated

    def _latest_attempt_for_page(self, page_id: str) -> PageAttempt | None:
        latest_id = self._latest_by_page.get(page_id)
        return self._attempts.get(latest_id) if latest_id else None

    def _write_attempt(self, attempt: PageAttempt, *, mutation: str) -> None:
        self._attempts[attempt.attempt_id] = attempt
        self._latest_by_page[attempt.task.page_id] = attempt.attempt_id
        self._max_epoch_by_page[attempt.task.page_id] = max(
            self._max_epoch_by_page.get(attempt.task.page_id, 0), attempt.epoch
        )
        mutate = getattr(self.store, "mutate", None)
        if callable(mutate):
            mutate(mutation, attempt.attempt_id)

    def _require_current_attempt(self, attempt_id: str, *, epoch: int | None) -> PageAttempt:
        attempt = self._attempts[attempt_id]
        if self._latest_by_page.get(attempt.task.page_id) != attempt_id:
            raise StaleEpochError(f"stale attempt {attempt_id}")
        if epoch is not None and attempt.epoch != epoch:
            raise StaleEpochError(f"stale epoch {epoch} for {attempt_id}")
        return attempt
