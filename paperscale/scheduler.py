"""Compact-index scheduler for bounded lazy queueing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


class CompactIndexError(RuntimeError):
    """Raised when compact-index reads fail closed."""


CorruptIndexError = CompactIndexError


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


class LazyPageQueue:
    def __init__(self, *, document_id: str, page_count: int, buffer_size: int) -> None:
        self.document_id = document_id
        self.total_pages = page_count
        self._buffer_size = max(buffer_size, 0)
        self._buffer = [f"{document_id}:{page}" for page in range(1, min(page_count, self._buffer_size) + 1)]

    def peek_buffer(self) -> list[str]:
        return list(self._buffer)

    def __iter__(self) -> Iterator[str]:
        for page in range(1, self.total_pages + 1):
            yield f"{self.document_id}:{page}"


class JobScheduler:
    def __init__(self, *, store: Any, queue_size: int) -> None:
        self.store = store
        self.queue_size = queue_size

    def status(self, _job_id: str) -> dict[str, Any]:
        payload = self.store.read_index("job-index")
        if not isinstance(payload, dict):
            raise CorruptIndexError("missing or corrupt job status index")
        return payload

    def resume(self, _job_id: str) -> dict[str, Any]:
        payload = self.store.read_index("resume-index")
        if not isinstance(payload, dict):
            raise CorruptIndexError("missing or corrupt resume index")
        return payload

    def build_lazy_queue(self, *, document_id: str, page_count: int) -> LazyPageQueue:
        return LazyPageQueue(document_id=document_id, page_count=page_count, buffer_size=self.queue_size)


class Scheduler:
    def __init__(self, index: Any, queue_size: int, capacity: ProviderCapacity | None = None) -> None:
        self._index = index
        self._queue_size = queue_size
        self._capacity = capacity
        self._pending_page_ids = list(index.iter_pending_page_ids())
        self._queued_count = 0

    @property
    def queued_count(self) -> int:
        return self._queued_count

    def status(self) -> Any:
        return self._index.read_status_index()

    def resume_plan(self) -> Any:
        try:
            return self._index.read_resume_index()
        except Exception as exc:
            raise CompactIndexError(str(exc)) from exc

    def repair_index(self) -> Any:
        return self._index.full_tree_scan()

    def fill_queue(self) -> None:
        if self._capacity is not None and self._capacity.max_provider_queue <= 0:
            self._queued_count = 0
            return
        hard_cap = self._queue_size
        if self._capacity is not None:
            hard_cap = min(hard_cap, self._capacity.max_provider_queue)
        self._queued_count = min(hard_cap, len(self._pending_page_ids))

    def on_provider_overload(self, controller: RetryStormController) -> None:
        if controller.circuit_open:
            self._queued_count = min(self._queued_count, self._capacity.max_provider_queue if self._capacity else self._queue_size)
