"""Filesystem StateStore helpers with atomic write discipline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from paperscale.contracts import ensure_known_schema


class FileSystemStateStore:
    def __init__(self, root: Path, *, allow_weak_fs: bool = False) -> None:
        self.root = Path(root)
        self.allow_weak_fs = allow_weak_fs
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json_atomic(self, relative_path: str | Path, payload: dict[str, Any]) -> Path:
        ensure_known_schema(payload)
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, sort_keys=True, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp, target)
            self._fsync_dir(target.parent)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return target

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        payload = json.loads((self.root / relative_path).read_text(encoding="utf-8"))
        ensure_known_schema(payload)
        return payload

    def _fsync_dir(self, directory: Path) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            if self.allow_weak_fs:
                return
            raise
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
