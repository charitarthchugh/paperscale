"""Single-machine, multi-process document claim tier.

Scope is always a single machine with multiple ``paperscale`` processes pulling
work; no cross-machine / NFS / object-store backends (explicitly YAGNI). The unit
of claim is the document/job, so the entire intra-process design (single index
writer, async pool, backpressure) runs unchanged within a claim.

Substrate: an ``O_EXCL`` filesystem claim record ``jobs/<job_id>/claim.json``,
atomic on a local fs, carrying ``worker_id``/``epoch``/``lease_expires_at``/
``heartbeat_at``. Correctness does NOT rest on the lock: a paused owner can wake
after takeover and still write, and a dumb filesystem cannot fence that write.
Safety rests instead on (1) page-keyed deterministic artifacts (a zombie write is
redundant, not corrupting) and (2) epoch-aware reconcile (the index writer/recovery
ignores attempts from a superseded epoch). A brief two-owner overlap therefore only
yields a stale index the higher-epoch owner overwrites and resume reconciles.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

from paperscale.contracts import CURRENT_SCHEMA_VERSION, ensure_known_schema


@dataclass(frozen=True, slots=True)
class Claim:
    job_id: str
    worker_id: str
    epoch: int
    lease_expires_at: float
    heartbeat_at: float


class ClaimStore:
    """O_EXCL document claims with leases, heartbeats, and epoch reclaim."""

    def __init__(
        self,
        root: Path | str,
        *,
        worker_id: str,
        clock: Callable[[], float],
        lease_seconds: float = 60.0,
        heartbeat_seconds: float = 20.0,
    ) -> None:
        self.root = Path(root)
        self.worker_id = worker_id
        self._clock = clock
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = heartbeat_seconds

    # -- paths -------------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def _claim_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "claim.json"

    def _done_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "done.json"

    # -- done marker (durable truth) --------------------------------------
    def is_done(self, job_id: str) -> bool:
        return self._done_path(job_id).exists()

    def mark_done(self, job_id: str) -> None:
        """Write a durable, fsync'd ``done`` marker on true completion.

        Truth, unlike the eventually-consistent index. The claim path checks the
        marker before attempting a claim, so finished documents are skipped.
        """
        payload = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "kind": "job_done_marker",
            "job_id": job_id,
            "worker_id": self.worker_id,
            "completed_at": float(self._clock()),
        }
        _write_json_fsync(self._done_path(job_id), payload)

    # -- claim lifecycle ---------------------------------------------------
    def try_claim(self, job_id: str, *, skip_if_done: bool = True) -> Claim | None:
        """Attempt to claim ``job_id``; return a ``Claim`` or ``None`` if unavailable.

        Returns ``None`` when held by a live owner (any process), or when the job is
        already done and ``skip_if_done`` is set. The auto-scanning ``work`` loop uses
        ``skip_if_done=True`` to skip finished jobs; an operator-initiated
        ``run``/``resume`` passes ``skip_if_done=False`` so a stale done marker never
        blocks an explicit re-run. Reclaims an expired lease at a strictly higher epoch.
        """
        if skip_if_done and self.is_done(job_id):
            return None
        self._job_dir(job_id).mkdir(parents=True, exist_ok=True)
        now = float(self._clock())

        # Fast path: nobody holds the claim -> atomic exclusive create.
        created = self._exclusive_create(job_id, epoch=1, now=now)
        if created is not None:
            return created

        # Contended: read the incumbent. Skip if its lease is still live.
        incumbent = self._read_claim(job_id)
        if incumbent is None:
            # File appeared/vanished mid-race; one more exclusive attempt.
            return self._exclusive_create(job_id, epoch=1, now=now)
        if incumbent.lease_expires_at > now:
            return None
        # Expired lease: reclaim at a higher epoch (last-write-wins on a dumb fs;
        # the read-side epoch fence makes the brief overlap safe).
        return self._overwrite_claim(job_id, epoch=incumbent.epoch + 1, now=now)

    def heartbeat(self, claim: Claim) -> Claim:
        now = float(self._clock())
        return self._overwrite_claim(claim.job_id, epoch=claim.epoch, now=now)

    def release(self, claim: Claim) -> None:
        """Drop our claim so another process may take the job.

        Only removes the record if we still own it (epoch + worker match), so a
        late release after takeover does not evict the new owner.
        """
        current = self._read_claim(claim.job_id)
        if current is None:
            return
        if current.worker_id == claim.worker_id and current.epoch == claim.epoch:
            try:
                self._claim_path(claim.job_id).unlink()
            except FileNotFoundError:
                pass

    # -- internals ---------------------------------------------------------
    def _exclusive_create(self, job_id: str, *, epoch: int, now: float) -> Claim | None:
        claim = Claim(
            job_id=job_id,
            worker_id=self.worker_id,
            epoch=epoch,
            lease_expires_at=now + self.lease_seconds,
            heartbeat_at=now,
        )
        path = self._claim_path(job_id)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return None
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(_claim_payload(claim), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        return claim

    def _overwrite_claim(self, job_id: str, *, epoch: int, now: float) -> Claim:
        claim = Claim(
            job_id=job_id,
            worker_id=self.worker_id,
            epoch=epoch,
            lease_expires_at=now + self.lease_seconds,
            heartbeat_at=now,
        )
        _write_json_fsync(self._claim_path(job_id), _claim_payload(claim))
        return claim

    def _read_claim(self, job_id: str) -> Claim | None:
        path = self._claim_path(job_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        ensure_known_schema(payload)
        try:
            return Claim(
                job_id=str(payload["job_id"]),
                worker_id=str(payload["worker_id"]),
                epoch=int(payload["epoch"]),
                lease_expires_at=float(payload["lease_expires_at"]),
                heartbeat_at=float(payload["heartbeat_at"]),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _claim_payload(claim: Claim) -> dict[str, object]:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "kind": "job_claim",
        "job_id": claim.job_id,
        "worker_id": claim.worker_id,
        "epoch": claim.epoch,
        "lease_expires_at": claim.lease_expires_at,
        "heartbeat_at": claim.heartbeat_at,
    }


def _write_json_fsync(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
