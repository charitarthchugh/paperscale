"""Versioned contracts and schema guards."""

from __future__ import annotations

from dataclasses import dataclass

CURRENT_SCHEMA_VERSION = 1


class UnknownSchemaVersionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VersionedRecord:
    schema_version: int


def ensure_known_schema(record: dict, *, current: int = CURRENT_SCHEMA_VERSION) -> None:
    version = record.get("schema_version")
    if not isinstance(version, int):
        raise UnknownSchemaVersionError("missing or invalid schema_version")
    if version > current:
        raise UnknownSchemaVersionError(f"unknown future schema version {version}")
