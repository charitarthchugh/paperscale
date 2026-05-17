"""Compact-index scheduler for bounded lazy queueing."""

from __future__ import annotations

from dataclasses import dataclass


class CompactIndexError(RuntimeError):
    """Raised when compact index reads fail closed."""


@dataclass(frozen=True, slots=True)
class ProviderCapacity:
    max_in_flight: int
    max_provider_queue: int
    circuit_breaker_threshold: int = 3


class RetryStormController:
    def __init__(self, capacity: ProviderCapacity) -> None:
        self._capacity = capacity
        self._overload_events = 0

    def record_overload(self, _signal: str) -> None:
        self._overload_events += 1

    @property
    def circuit_open(self) -> bool:
        return self._overload_events >= self._capacity.circuit_breaker_threshold


class Scheduler:
    def __init__(self, index, queue_size: int, capacity: ProviderCapacity | None = None) -> None:
        self._index = index
        self._queue_size = queue_size
        self._capacity = capacity
        self._pending_page_ids = iter(index.iter_pending_page_ids())
        self._queued_count = 0
        self._paused = False

    @property
    def queued_count(self) -> int:
        return self._queued_count

    def status(self):
        return self._index.read_status_index()

    def resume_plan(self):
        try:
            return self._index.read_resume_index()
        except Exception as exc:  # fail closed on any compact-index corruption
            raise CompactIndexError(str(exc)) from exc

    def repair_index(self):
        return self._index.full_tree_scan()

    def fill_queue(self) -> None:
        if self._paused:
            return
        hard_cap = self._queue_size
        if self._capacity is not None:
            hard_cap = min(hard_cap, self._capacity.max_provider_queue)
        while self._queued_count < hard_cap:
            try:
                next(self._pending_page_ids)
            except StopIteration:
                break
            self._queued_count += 1

    def on_provider_overload(self, _controller: RetryStormController) -> None:
        self._paused = True
