"""Tests for the local work queue: grouping, locking, and resume-by-skip."""

import tempfile
import unittest
from pathlib import Path

from paperscale.work_queue import DONE_FLAGS_DIR, LocalBackend, WorkItem, WorkQueue


class WorkQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = self._tmp.name

    async def asyncTearDown(self):
        self._tmp.cleanup()

    def _queue(self) -> WorkQueue:
        return WorkQueue(LocalBackend(self.workspace))

    async def test_populate_groups_by_items_per_group(self):
        queue = self._queue()
        paths = [f"/docs/{i}.pdf" for i in range(5)]
        await queue.populate_queue(paths, items_per_group=2)
        size = await queue.initialize_queue()
        # 5 paths / 2 per group -> 3 groups (2, 2, 1).
        self.assertEqual(size, 3)

    async def test_populate_is_idempotent_for_known_paths(self):
        queue = self._queue()
        await queue.populate_queue(["/docs/a.pdf", "/docs/b.pdf"], items_per_group=1)
        # Re-adding the same paths must not create new groups.
        await queue.populate_queue(["/docs/a.pdf", "/docs/b.pdf"], items_per_group=1)
        size = await queue.initialize_queue()
        self.assertEqual(size, 2)

    async def test_get_work_then_mark_done_creates_flag(self):
        queue = self._queue()
        await queue.populate_queue(["/docs/a.pdf"], items_per_group=1)
        await queue.initialize_queue()

        item = await queue.get_work()
        self.assertIsInstance(item, WorkItem)
        flag = Path(self.workspace) / DONE_FLAGS_DIR / f"done_{item.hash}.flag"
        self.assertFalse(flag.exists())

        await queue.mark_done(item)
        self.assertTrue(flag.exists())

    async def test_resume_skips_completed_items(self):
        # Run one item to completion, then a fresh queue over the same workspace
        # must not re-surface it (this is the default resume behavior).
        first = self._queue()
        await first.populate_queue(["/docs/a.pdf", "/docs/b.pdf"], items_per_group=1)
        await first.initialize_queue()
        done_item = await first.get_work()
        await first.mark_done(done_item)

        resumed = self._queue()
        remaining = await resumed.initialize_queue()
        self.assertEqual(remaining, 1)

    async def test_worker_lock_blocks_second_getter(self):
        queue = self._queue()
        await queue.populate_queue(["/docs/a.pdf"], items_per_group=1)
        await queue.initialize_queue()

        first = await queue.get_work()
        self.assertIsNotNone(first)

        # A second independent queue over the same workspace sees the lock and,
        # with no other work, returns None.
        other = self._queue()
        await other.initialize_queue()
        self.assertIsNone(await other.get_work())


if __name__ == "__main__":
    unittest.main()
