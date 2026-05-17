"""Resource governor for fixed acquisition-order enforcement."""

from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Iterator, Callable, Any


class ResourceOrderError(RuntimeError):
    """Raised when a resource is acquired or released out of order."""


class ResourceKind(IntEnum):
    CANCELLATION = 1
    SCHEDULER = 2
    RENDER = 3
    FILE_DESCRIPTOR = 4
    PROVIDER = 5
    PAGE_LEASE = 6
    STATE_STORE = 7


class ResourceGovernor:
    """Tracks held resources and enforces the global acquisition order."""

    def __init__(self, file_opener: Callable[..., Any] | None = None) -> None:
        self._file_opener = file_opener or Path.open
        self._held_stack: list[ResourceKind] = []

    def is_held(self, kind: ResourceKind) -> bool:
        return kind in self._held_stack

    @contextmanager
    def acquire(self, kind: ResourceKind) -> Iterator[None]:
        self._acquire(kind)
        try:
            yield
        finally:
            self._release(kind)

    @contextmanager
    def open_file(self, path: Path, mode: str, **kwargs: Any) -> Iterator[Any]:
        with self.acquire(ResourceKind.FILE_DESCRIPTOR):
            handle = self._file_opener(path, mode, **kwargs)
            try:
                yield handle
            finally:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()

    def _acquire(self, kind: ResourceKind) -> None:
        if self._held_stack and kind < self._held_stack[-1]:
            raise ResourceOrderError(f"cannot acquire {kind.name} after {self._held_stack[-1].name}")
        self._held_stack.append(kind)

    def _release(self, kind: ResourceKind) -> None:
        if not self._held_stack or self._held_stack[-1] is not kind:
            raise ResourceOrderError(f"must release resources in reverse order: {kind.name}")
        self._held_stack.pop()
