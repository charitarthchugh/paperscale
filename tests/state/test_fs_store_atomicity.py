from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.harness.fakes import FakeCrashHook
from tests.harness.imports import require_symbol


class FileSystemStateStoreAtomicityTests(unittest.TestCase):
    def test_crash_after_temp_fsync_before_replace_leaves_old_target_and_ignores_temp(self) -> None:
        FileSystemStateStore = require_symbol("paperscale.state.fs_store", "FileSystemStateStore")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = FileSystemStateStore(root, allow_weak_fs=False)
            store.write_json_atomic("results/page-1.json", {"schema_version": 1, "text": "old"})
            crash = FakeCrashHook("after_temp_fsync_before_replace")
            with self.assertRaises(RuntimeError):
                store.write_json_atomic(
                    "results/page-1.json",
                    {"schema_version": 1, "text": "new"},
                    crash_hook=crash,
                )
            self.assertEqual(store.read_json("results/page-1.json")["text"], "old")
            self.assertEqual(store.list_committed("results"), ["page-1.json"])

    def test_unknown_schema_in_normal_read_fails_closed_without_repair_scan(self) -> None:
        FileSystemStateStore = require_symbol("paperscale.state.fs_store", "FileSystemStateStore")
        UnknownSchemaVersionError = require_symbol("paperscale.contracts", "UnknownSchemaVersionError")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStateStore(Path(tmpdir), allow_weak_fs=False)
            store.write_json_atomic("ledger/page-1.json", {"schema_version": 999, "kind": "page_attempt"})
            with self.assertRaises(UnknownSchemaVersionError):
                store.read_versioned_json("ledger/page-1.json", expected_kind="page_attempt", current_version=1)
            self.assertEqual(store.debug_tree_scan_count, 0, "normal schema failure must not repair-scan")

    def test_repair_index_is_the_only_path_allowed_to_scan_artifact_tree(self) -> None:
        FileSystemStateStore = require_symbol("paperscale.state.fs_store", "FileSystemStateStore")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileSystemStateStore(Path(tmpdir), allow_weak_fs=False)
            store.write_json_atomic("artifacts/aa/page-1.json", {"schema_version": 1, "text": "ok"})
            store.status_from_index()
            self.assertEqual(store.debug_tree_scan_count, 0)
            store.repair_index()
            self.assertGreater(store.debug_tree_scan_count, 0)


if __name__ == "__main__":
    unittest.main()
