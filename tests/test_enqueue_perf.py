"""Enqueue is the batch 'startup' phase; it must stay cheap and observable.

A batch ``run``/``enqueue`` registers one job per input *before* any OCR begins.
For large lists (thousands of PDFs) the dominant cost is durable disk writes, so
the derived/rebuildable compact indexes must be written without fsync at enqueue
time (only the manifest — the truth — is fsync'd), and the phase must log so it is
not an invisible multi-minute stall.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paperscale.runner import DocumentOcrRunner, RunnerConfig


class _FakeRenderer:
    def __init__(self, path: Path, render_options: dict) -> None:
        self.path = path
        self.render_options = render_options
        self.page_count = 3


def _runner(state_root: Path) -> tuple[DocumentOcrRunner, list[tuple[str, bool]]]:
    runner = DocumentOcrRunner(
        RunnerConfig(state_root=state_root),
        renderer_factory=lambda path, options: _FakeRenderer(path, options),
    )
    calls: list[tuple[str, bool]] = []
    original = runner.state.write_json_atomic

    def spy(relative_path, payload, *, crash_hook=None, fsync=True):
        calls.append((str(relative_path), fsync))
        return original(relative_path, payload, crash_hook=crash_hook, fsync=fsync)

    runner.state.write_json_atomic = spy  # type: ignore[method-assign]
    return runner, calls


class EnqueueDurabilityTests(unittest.TestCase):
    def test_manifest_is_fsynced_but_indexes_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "doc.pdf").write_bytes(b"%PDF-1.4\n")
            runner, calls = _runner(root / ".paperscale")

            runner.enqueue(input_path=root / "doc.pdf", output_path=root / "out.md", job_id="j1")

            by_name = {
                Path(rel).name: fsync
                for rel, fsync in calls
                if Path(rel).name in {"manifest.json", "status.json", "resume.json", "reconcile.json"}
            }
            self.assertTrue(by_name.get("manifest.json"), "manifest (the truth) must be fsync'd")
            for index in ("status.json", "resume.json", "reconcile.json"):
                self.assertIn(index, by_name, f"{index} should be written at enqueue")
                self.assertFalse(
                    by_name[index],
                    f"{index} is derived/rebuildable and must NOT be fsync'd at enqueue",
                )


class EnqueueObservabilityTests(unittest.TestCase):
    def test_enqueue_many_logs_start_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                (root / f"doc{i}.pdf").write_bytes(b"%PDF-1.4\n")
            runner, _ = _runner(root / ".paperscale")
            inputs = [root / f"doc{i}.pdf" for i in range(3)]

            with self.assertLogs("paperscale", level="INFO") as captured:
                enqueued, skipped = runner.enqueue_many(inputs, output_dir=root / "out")

            self.assertEqual(len(enqueued), 3)
            self.assertEqual(skipped, [])
            joined = "\n".join(captured.output)
            self.assertIn("enqueuing", joined.lower())
            self.assertIn("3", joined)


if __name__ == "__main__":
    unittest.main()
