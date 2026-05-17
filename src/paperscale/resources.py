"""Resource governor for fixed acquisition-order enforcement."""

from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterator


class ResourceOrderViolation(RuntimeError):
    """Raised when a resource is acquired or released out of order."""


class UnauthorizedResourceError(RuntimeError):
    """Raised when code opens unmanaged resources outside a governor token."""


ResourceOrderError = ResourceOrderViolation


class ResourceKind(IntEnum):
    CANCELLATION = 1
    SCHEDULER = 2
    RENDER = 3
    FILE_DESCRIPTOR = 4
    PROVIDER = 5
    PAGE_LEASE = 6
    STATE_STORE = 7


_RESOURCE_ORDER = {
    "cancellation_token": 1,
    "scheduler_slot": 2,
    "pdf_render_slot": 3,
    "render": 3,
    "file_descriptor": 4,
    "provider_concurrency": 5,
    "provider": 5,
    "page_lease": 6,
    "state_store_lock": 7,
}
_KIND_TO_NAME = {
    ResourceKind.CANCELLATION: "cancellation_token",
    ResourceKind.SCHEDULER: "scheduler_slot",
    ResourceKind.RENDER: "pdf_render_slot",
    ResourceKind.FILE_DESCRIPTOR: "file_descriptor",
    ResourceKind.PROVIDER: "provider_concurrency",
    ResourceKind.PAGE_LEASE: "page_lease",
    ResourceKind.STATE_STORE: "state_store_lock",
}


def _resource_name(kind: str | ResourceKind) -> str:
    if isinstance(kind, ResourceKind):
        return _KIND_TO_NAME[kind]
    if kind not in _RESOURCE_ORDER:
        raise ValueError(f"unknown resource kind {kind!r}")
    return kind


class ResourceGovernor:
    """Tracks held resources and enforces the global acquisition order."""

    def __init__(self, file_opener: Callable[..., Any] | None = None) -> None:
        self._file_opener = file_opener or Path.open
        self._held_stack: list[str] = []
        self.debug_release_order: list[str] = []

    @property
    def debug_active_order(self) -> list[str]:
        return list(self._held_stack)

    def is_held(self, kind: str | ResourceKind) -> bool:
        return _resource_name(kind) in self._held_stack

    @contextmanager
    def acquire(self, kind: str | ResourceKind) -> Iterator[None]:
        name = _resource_name(kind)
        self._acquire(name)
        try:
            yield
        finally:
            self._release(name)

    @contextmanager
    def acquire_many(self, resources: list[str | ResourceKind]) -> Iterator[None]:
        acquired: list[str] = []
        try:
            for resource in resources:
                name = _resource_name(resource)
                self._acquire(name)
                acquired.append(name)
            yield
        finally:
            while acquired:
                self._release(acquired.pop())

    def open_file(self, path: str | Path, mode: str, **kwargs: Any) -> Any:
        if not self.is_held("file_descriptor"):
            raise UnauthorizedResourceError("file opens must occur inside a file_descriptor token")
        return self._file_opener(Path(path), mode, **kwargs)

    @contextmanager
    def managed_open_file(self, path: str | Path, mode: str, **kwargs: Any) -> Iterator[Any]:
        with self.acquire("file_descriptor"):
            handle = self.open_file(path, mode, **kwargs)
            try:
                yield handle
            finally:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()

    def _acquire(self, name: str) -> None:
        if self._held_stack and _RESOURCE_ORDER[name] < _RESOURCE_ORDER[self._held_stack[-1]]:
            raise ResourceOrderViolation(f"cannot acquire {name} after {self._held_stack[-1]}")
        self._held_stack.append(name)

    def _release(self, name: str) -> None:
        if not self._held_stack or self._held_stack[-1] != name:
            raise ResourceOrderViolation(f"must release resources in reverse order: {name}")
        released = self._held_stack.pop()
        self.debug_release_order.append(released)
