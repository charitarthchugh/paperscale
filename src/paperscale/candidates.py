"""Lazy plan manifest: one immutable candidate file that ``work`` materializes.

The ``plan`` phase resolves a batch of input documents into a single durable
``candidates/<workload_id>.jsonl`` manifest instead of eagerly creating one
on-disk job directory per input (the multi-minute startup stall). Each line is a
:class:`CandidateRecord`; ``work`` reads them later via :func:`read_candidates`
and creates job dirs lazily.

The atomic-durable commit discipline (temp + fsync + ``os.replace`` + parent-dir
fsync) mirrors ``state.fs_store.write_json_atomic`` / ``runner._fsync_parent`` but
is kept self-contained here.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from paperscale.observability import get_logger


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One planned-but-not-yet-materialized job."""

    job_id: str
    input_path: str
    output_path: str
    profile: str
    model: str | None
    base_url: str
    capacity: str
    created_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "profile": self.profile,
            "model": self.model,
            "base_url": self.base_url,
            "capacity": self.capacity,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_json(payload: dict[str, Any]) -> CandidateRecord:
        return CandidateRecord(
            job_id=payload["job_id"],
            input_path=payload["input_path"],
            output_path=payload["output_path"],
            profile=payload["profile"],
            model=payload["model"],
            base_url=payload["base_url"],
            capacity=payload["capacity"],
            created_at=float(payload["created_at"]),
        )


def _unique_job_id(input_path: Path, used: set[str]) -> str:
    """Derive a stable job id from the file stem, de-duped against ``used``.

    Mirrors ``runner._unique_job_id`` but checks only the in-memory ``used`` set —
    ``plan`` never touches job directories.
    """

    base = input_path.stem or "job"
    candidate = base
    suffix = 1
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def plan_candidates(
    inputs: list[Path],
    *,
    output_dir: Path,
    state_root: Path,
    profile: str,
    model: str | None,
    base_url: str,
    capacity: str,
    clock: Callable[[], float] = time.time,
) -> Path:
    """Plan a batch of inputs into one immutable candidate manifest.

    Resolves each input to an absolute path, drops exact duplicates, derives a
    de-duplicated ``job_id`` per surviving input, and writes all records as JSONL
    to ``<state_root>/candidates/<workload_id>.jsonl`` via a single atomic-durable
    commit. Returns the path to the written manifest.
    """

    output_dir = Path(output_dir)
    state_root = Path(state_root)
    logger = get_logger()
    total = len(inputs)
    logger.info("planning %d document(s)...", total)

    # abspath dedup: keep first occurrence, drop exact duplicates by resolved path.
    seen_paths: set[str] = set()
    resolved_inputs: list[Path] = []
    dropped = 0
    for raw in inputs:
        resolved = Path(raw).resolve()
        key = str(resolved)
        if key in seen_paths:
            dropped += 1
            logger.info("planning: dropping duplicate input %s", resolved)
            continue
        seen_paths.add(key)
        resolved_inputs.append(resolved)

    created_at = float(clock())
    workload_id = f"{int(created_at)}-{uuid.uuid4().hex[:8]}"

    used: set[str] = set()
    records: list[CandidateRecord] = []
    for resolved in resolved_inputs:
        job_id = _unique_job_id(resolved, used)
        used.add(job_id)
        output_path = (output_dir / f"{job_id}.md").resolve()
        records.append(
            CandidateRecord(
                job_id=job_id,
                input_path=str(resolved),
                output_path=str(output_path),
                profile=profile,
                model=model,
                base_url=base_url,
                capacity=capacity,
                created_at=created_at,
            )
        )

    target = (state_root / "candidates" / f"{workload_id}.jsonl").resolve()
    _write_jsonl_atomic(target, records)

    logger.info(
        "planned %d candidate(s) into workload %s%s",
        len(records),
        workload_id,
        f" ({dropped} duplicate(s) dropped)" if dropped else "",
    )
    return target


def _write_jsonl_atomic(target: Path, records: list[CandidateRecord]) -> None:
    """Atomically and durably write ``records`` as JSONL to ``target``.

    temp-file in the target dir -> write+flush+fsync -> ``os.replace`` ->
    fsync the parent directory; unlink the temp on any failure.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record.to_json(), sort_keys=True, separators=(",", ":")))
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, target)
        _fsync_parent(target.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _fsync_parent(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_candidates(path: Path) -> list[CandidateRecord]:
    """Parse a candidates JSONL manifest, skipping blank lines."""

    records: list[CandidateRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(CandidateRecord.from_json(json.loads(line)))
    return records
