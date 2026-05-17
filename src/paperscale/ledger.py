"""Replayable page-attempt ledger and crash-recovery helpers.

The ledger is deliberately provider-call aware: callers reserve an attempt and
persist it before provider I/O, then transition to ``in_flight`` immediately
before the provider request. Recovery distinguishes never-started reservations
from requests that may have reached a non-idempotent provider.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol
from uuid import uuid4


class LedgerState(StrEnum):
    """Durable state for one page attempt."""

    PENDING = "pending"
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    AMBIGUOUS = "ambiguous"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """Versioned compact ledger record for a page attempt."""

    schema_version: int
    attempt_id: str
    page_id: str
    provider_request_fingerprint: str
    worker_id: str | None
    epoch: int
    state: LedgerState
    lease_expires_at: float | None
    heartbeat_at: float | None
    provider_call_started_at: float | None = None
    provider_response_committed_at: float | None = None
    result_pointer: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Summary of lease-expiry recovery transitions."""

    requeued_pages: list[str]
    ambiguous_pages: list[str]


class LedgerError(RuntimeError):
    """Base class for ledger transition failures."""


class AmbiguousAttemptError(LedgerError):
    """Raised when a duplicate provider call would be unsafe."""


class StaleEpochError(LedgerError):
    """Raised when an older worker epoch attempts to mutate current state."""


class InvalidLedgerTransitionError(LedgerError):
    """Raised for state-machine violations."""


class LedgerStore(Protocol):
    """Minimal store contract needed by the ledger state machine."""

    def append(self, record: LedgerRecord) -> None:
        """Persist a compact record transition durably."""

    def latest(self, page_id: str) -> LedgerRecord | None:
        """Return the latest compact index record for a page, if any."""

    def latest_by_attempt(self, attempt_id: str) -> LedgerRecord | None:
        """Return the latest record for an attempt, if any."""

    def active_records(self) -> Iterable[LedgerRecord]:
        """Return compact-index records that may need recovery."""


class InMemoryLedgerStore:
    """Deterministic ledger store for tests and fake local flows.

    Production filesystem persistence belongs behind the same ``LedgerStore``
    protocol. This test store keeps a transition log plus compact indexes and
    intentionally exposes event hooks so tests can prove reservation-before-I/O
    ordering without real provider calls.
    """

    def __init__(self, *, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self._records: list[LedgerRecord] = []
        self._latest_by_page: dict[str, LedgerRecord] = {}
        self._latest_by_attempt: dict[str, LedgerRecord] = {}

    def append(self, record: LedgerRecord) -> None:
        self.events.append(f"write:{record.state.value}")
        self._records.append(record)
        self._latest_by_page[record.page_id] = record
        self._latest_by_attempt[record.attempt_id] = record

    def latest(self, page_id: str) -> LedgerRecord | None:
        return self._latest_by_page.get(page_id)

    def require_latest(self, page_id: str) -> LedgerRecord:
        record = self.latest(page_id)
        if record is None:
            msg = f"no ledger record for page {page_id!r}"
            raise KeyError(msg)
        return record

    def latest_by_attempt(self, attempt_id: str) -> LedgerRecord | None:
        return self._latest_by_attempt.get(attempt_id)

    def active_records(self) -> Iterable[LedgerRecord]:
        for record in self._latest_by_page.values():
            if record.state in {LedgerState.RESERVED, LedgerState.IN_FLIGHT}:
                yield record

    @property
    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)


class Ledger:
    """Page-attempt ledger enforcing recovery and duplicate-call policy."""

    def __init__(self, store: LedgerStore, *, now: Callable[[], float]) -> None:
        self._store = store
        self._now = now

    def reserve_page(
        self,
        *,
        page_id: str,
        provider_request_fingerprint: str,
        worker_id: str,
        lease_seconds: float,
        allow_ambiguous_retry: bool = False,
    ) -> LedgerRecord:
        """Reserve a page attempt before provider I/O.

        Ambiguous work is fail-closed by default because a provider request may
        have reached a non-idempotent model server before the crash.
        """

        now = self._now()
        latest = self._store.latest(page_id)
        if latest is not None:
            if latest.state == LedgerState.AMBIGUOUS and not allow_ambiguous_retry:
                msg = (
                    f"page {page_id!r} has ambiguous attempt {latest.attempt_id!r}; "
                    "retry requires explicit duplicate-call policy"
                )
                raise AmbiguousAttemptError(msg)
            if latest.state in {
                LedgerState.RESERVED,
                LedgerState.IN_FLIGHT,
                LedgerState.SUCCEEDED,
                LedgerState.FAILED_TERMINAL,
            }:
                msg = f"cannot reserve page {page_id!r} from state {latest.state.value!r}"
                raise InvalidLedgerTransitionError(msg)
            epoch = latest.epoch + 1
        else:
            epoch = 1

        record = LedgerRecord(
            schema_version=1,
            attempt_id=str(uuid4()),
            page_id=page_id,
            provider_request_fingerprint=provider_request_fingerprint,
            worker_id=worker_id,
            epoch=epoch,
            state=LedgerState.RESERVED,
            lease_expires_at=now + lease_seconds,
            heartbeat_at=now,
        )
        self._store.append(record)
        return record

    def mark_provider_call_started(self, attempt_id: str, *, worker_id: str, epoch: int) -> LedgerRecord:
        current = self._require_current_attempt(attempt_id, worker_id=worker_id, epoch=epoch)
        if current.state != LedgerState.RESERVED:
            msg = f"provider call can start only from reserved, got {current.state.value!r}"
            raise InvalidLedgerTransitionError(msg)
        updated = replace(current, state=LedgerState.IN_FLIGHT, provider_call_started_at=self._now())
        self._store.append(updated)
        return updated

    def heartbeat(self, attempt_id: str, *, worker_id: str, epoch: int, lease_seconds: float = 30.0) -> LedgerRecord:
        current = self._require_current_attempt(attempt_id, worker_id=worker_id, epoch=epoch)
        if current.state not in {LedgerState.RESERVED, LedgerState.IN_FLIGHT}:
            msg = f"heartbeat allowed only for active attempts, got {current.state.value!r}"
            raise InvalidLedgerTransitionError(msg)
        now = self._now()
        updated = replace(current, heartbeat_at=now, lease_expires_at=now + lease_seconds)
        self._store.append(updated)
        return updated

    def commit_success(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        epoch: int,
        result_pointer: str,
    ) -> LedgerRecord:
        current = self._require_current_attempt(attempt_id, worker_id=worker_id, epoch=epoch)
        if current.state != LedgerState.IN_FLIGHT:
            msg = f"success commit allowed only from in_flight, got {current.state.value!r}"
            raise InvalidLedgerTransitionError(msg)
        updated = replace(
            current,
            state=LedgerState.SUCCEEDED,
            lease_expires_at=None,
            provider_response_committed_at=self._now(),
            result_pointer=result_pointer,
        )
        self._store.append(updated)
        return updated

    def fail_attempt(
        self,
        attempt_id: str,
        *,
        worker_id: str,
        epoch: int,
        retryable: bool,
        reason: str,
    ) -> LedgerRecord:
        current = self._require_current_attempt(attempt_id, worker_id=worker_id, epoch=epoch)
        if current.state not in {LedgerState.RESERVED, LedgerState.IN_FLIGHT}:
            msg = f"failure commit allowed only from active states, got {current.state.value!r}"
            raise InvalidLedgerTransitionError(msg)
        updated = replace(
            current,
            state=LedgerState.FAILED_RETRYABLE if retryable else LedgerState.FAILED_TERMINAL,
            lease_expires_at=None,
            provider_response_committed_at=self._now() if current.state == LedgerState.IN_FLIGHT else None,
            failure_reason=reason,
        )
        self._store.append(updated)
        return updated

    def recover_expired_leases(self, *, now: float | None = None) -> RecoveryReport:
        """Recover expired active attempts using compact active records only."""

        observed_now = self._now() if now is None else now
        requeued_pages: list[str] = []
        ambiguous_pages: list[str] = []
        for record in list(self._store.active_records()):
            if record.lease_expires_at is None or record.lease_expires_at > observed_now:
                continue
            if record.state == LedgerState.RESERVED and record.provider_call_started_at is None:
                self._store.append(
                    replace(
                        record,
                        state=LedgerState.PENDING,
                        worker_id=None,
                        lease_expires_at=None,
                        heartbeat_at=None,
                    )
                )
                requeued_pages.append(record.page_id)
            elif record.state == LedgerState.IN_FLIGHT and record.provider_response_committed_at is None:
                self._store.append(replace(record, state=LedgerState.AMBIGUOUS, lease_expires_at=None))
                ambiguous_pages.append(record.page_id)
        return RecoveryReport(requeued_pages=requeued_pages, ambiguous_pages=ambiguous_pages)

    def _require_current_attempt(self, attempt_id: str, *, worker_id: str, epoch: int) -> LedgerRecord:
        current = self._store.latest_by_attempt(attempt_id)
        if current is None:
            msg = f"unknown attempt {attempt_id!r}"
            raise KeyError(msg)
        latest_for_page = self._store.latest(current.page_id)
        if latest_for_page is None or latest_for_page.attempt_id != attempt_id or latest_for_page.epoch != epoch:
            msg = f"attempt {attempt_id!r} is stale for page {current.page_id!r}"
            raise StaleEpochError(msg)
        if current.worker_id != worker_id or current.epoch != epoch:
            msg = f"attempt {attempt_id!r} has stale worker/epoch"
            raise StaleEpochError(msg)
        return current
