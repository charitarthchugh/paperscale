"""Deterministic fakes for safety-invariant tests.

These helpers deliberately avoid provider/network/file side effects unless a test
explicitly asks for them. Production code should depend on protocols/duck typing
so these fakes can observe ordering, durability, and scan behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class FakeClock:
    now_value: float = 1_700_000_000.0

    def now(self) -> float:
        return self.now_value

    def advance(self, seconds: float) -> None:
        self.now_value += seconds


@dataclass
class FakeCrashHook:
    crash_at: str | None = None
    seen: list[str] = field(default_factory=list)

    def checkpoint(self, name: str) -> None:
        self.seen.append(name)
        if self.crash_at == name:
            raise RuntimeError(f"synthetic crash at {name}")


@dataclass
class FakeProviderResponse:
    request_id: str = "fake-provider-request-1"
    markdown: str = "# Page 1\n\nHello from fake OCR."
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeProvider:
    responses: list[FakeProviderResponse] = field(default_factory=lambda: [FakeProviderResponse()])
    calls: list[Any] = field(default_factory=list)
    ledger_probe: Any | None = None

    def send(self, request: Any) -> FakeProviderResponse:
        if self.ledger_probe is not None:
            assert self.ledger_probe(), "provider called before durable ledger reservation"
        self.calls.append(request)
        if not self.responses:
            raise RuntimeError("fake provider exhausted")
        return self.responses.pop(0)


@dataclass
class RecordingStateStore:
    """Minimal store fake that distinguishes compact-index reads from tree scans."""

    records: dict[str, Any] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    mutations: list[tuple[str, Any]] = field(default_factory=list)
    index_reads: int = 0
    artifact_reads: int = 0
    tree_scans: int = 0
    lock_depth: int = 0

    def read_index(self, name: str) -> Any:
        self.index_reads += 1
        self.events.append(f"read_index:{name}")
        return self.records.get(name)

    def read_artifact(self, name: str) -> Any:
        self.artifact_reads += 1
        self.events.append(f"read_artifact:{name}")
        return self.records.get(name)

    def scan_tree(self, root: str | Path = ".") -> Iterable[str]:
        self.tree_scans += 1
        self.events.append(f"scan_tree:{root}")
        return list(self.records)

    def mutate(self, name: str, value: Any) -> None:
        self.mutations.append((name, value))
        self.records[name] = value

    def locked(self):
        store = self

        class _Lock:
            def __enter__(self) -> RecordingStateStore:
                store.lock_depth += 1
                store.events.append("lock:enter")
                return store

            def __exit__(self, exc_type, exc, tb) -> None:
                store.events.append("lock:exit")
                store.lock_depth -= 1

        return _Lock()


@dataclass
class FakeHttpClient:
    models: list[str]
    calls: list[str] = field(default_factory=list)
    status_code: int = 200

    def get(self, path: str) -> dict[str, Any]:
        self.calls.append(path)
        if self.status_code >= 400:
            return {"status_code": self.status_code, "data": []}
        return {"status_code": self.status_code, "data": [{"id": model} for model in self.models]}


@dataclass
class RecordingResourceGovernor:
    events: list[str] = field(default_factory=list)

    def acquire(self, name: str):
        governor = self

        class _Token:
            def __enter__(self):
                governor.events.append(f"acquire:{name}")
                return self

            def __exit__(self, exc_type, exc, tb):
                governor.events.append(f"release:{name}")

        return _Token()

    def assert_order(self, expected: list[str]) -> None:
        assert self.events == expected, f"resource order mismatch: {self.events!r} != {expected!r}"
