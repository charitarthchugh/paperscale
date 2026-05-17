"""Filesystem StateStore helpers with atomic write discipline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from paperscale.contracts import load_versioned_record


class FileSystemStateStore:
    def __init__(self, root: Path, *, allow_weak_fs: bool = False) -> None:
        self.root = Path(root)
        self.allow_weak_fs = allow_weak_fs
        self.root.mkdir(parents=True, exist_ok=True)
        self.debug_tree_scan_count = 0

    def write_json_atomic(self, relative_path: str | Path, payload: dict[str, Any], *, crash_hook: Any | None = None) -> Path:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, sort_keys=True, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            if crash_hook is not None:
                crash_hook.checkpoint("after_temp_fsync_before_replace")
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
        return json.loads((self.root / relative_path).read_text(encoding="utf-8"))

    def read_versioned_json(self, relative_path: str | Path, *, expected_kind: str, current_version: int) -> dict[str, Any]:
        payload = self.read_json(relative_path)
        return load_versioned_record(payload, expected_kind=expected_kind, current_version=current_version)

    def list_committed(self, relative_dir: str | Path) -> list[str]:
        directory = self.root / relative_dir
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.iterdir() if path.is_file() and not path.name.endswith(".tmp"))

    def status_from_index(self) -> dict[str, Any]:
        index = self.root / "index.json"
        if index.exists():
            return self.read_json("index.json")
        return {}

    def repair_index(self) -> dict[str, Any]:
        self.debug_tree_scan_count += 1
        files = [str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()]
        return {"files": files}

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
