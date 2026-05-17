"""Resource governor for acquisition-order enforcement.

The governor models the fixed resource order in the consensus plan:
cancellation -> scheduler -> render -> file descriptor -> provider ->
page lease -> state store.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Iterator


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
    """Tracks held resources and enforces a global acquisition order."""

    def __init__(self, file_opener=Path.open) -> None:
        self._file_opener = file_opener
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

    def open_file(self, path: Path, mode: str, **kwargs):
        with self.acquire(ResourceKind.FILE_DESCRIPTOR):
            return self._file_opener(path, mode, **kwargs)

    def _acquire(self, kind: ResourceKind) -> None:
        if self._held_stack and kind < self._held_stack[-1]:
            raise ResourceOrderError(
                f"cannot acquire {kind.name} after {self._held_stack[-1].name}"
            )
        self._held_stack.append(kind)

    def _release(self, kind: ResourceKind) -> None:
        if not self._held_stack or self._held_stack[-1] is not kind:
            raise ResourceOrderError(f"must release resources in reverse order: {kind.name}")
        self._held_stack.pop()

