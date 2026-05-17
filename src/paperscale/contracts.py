"""Versioned contracts and schema guards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, Callable

CURRENT_SCHEMA_VERSION = 1


class UnknownSchemaVersionError(RuntimeError):
    """Raised when persisted data uses an unknown future schema."""


class PageAttemptState(StrEnum):
    PENDING = "pending"
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    AMBIGUOUS = "ambiguous"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class VersionedRecord:
    schema_version: int


@dataclass(frozen=True, slots=True)
class PageTask:
    document_id: str
    page_number: int
    image_hash: str

    @property
    def page_id(self) -> str:
        return f"{self.document_id}:{self.page_number}"


def ensure_known_schema(record: dict[str, Any], *, current: int = CURRENT_SCHEMA_VERSION) -> None:
    version = record.get("schema_version")
    if not isinstance(version, int):
        raise UnknownSchemaVersionError("missing or invalid schema_version")
    if version > current:
        raise UnknownSchemaVersionError(f"unknown future schema version {version}")


def load_versioned_record(
    record: dict[str, Any],
    *,
    expected_kind: str,
    current_version: int = CURRENT_SCHEMA_VERSION,
    on_mutation: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load a persisted record, failing closed before any mutation callback."""

    ensure_known_schema(record, current=current_version)
    if record.get("kind") != expected_kind:
        raise ValueError(f"expected record kind {expected_kind!r}, got {record.get('kind')!r}")
    if on_mutation is not None:
        # Known schemas may be normalized/migrated by callers. The critical
        # invariant is that this hook is never reached for unknown future data.
        on_mutation("validated", record)
    return dict(record)


def build_provider_request_fingerprint(**parts: Any) -> str:
    """Build the request key from profile, provider, render, decoding, and image inputs."""

    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PageArtifact:
    """Durable page OCR artifact consumed by assembly and diagnostics."""

    page_id: str
    markdown: str
    result_pointer: str
    verifier_metadata: list[Any] | None = None

    @property
    def page_number(self) -> int:
        try:
            return int(self.page_id.rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"page_id {self.page_id!r} does not end with a numeric page number") from exc
