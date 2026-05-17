from __future__ import annotations

import unittest

from tests.harness.fakes import RecordingStateStore
from tests.harness.imports import require_symbol


class SchedulerTests(unittest.TestCase):
    def test_status_reads_compact_indexes_not_artifact_tree(self) -> None:
        JobScheduler = require_symbol("paperscale.scheduler", "JobScheduler")
        store = RecordingStateStore(records={"job-index": {"pages_total": 1_000_000, "succeeded": 7}})
        scheduler = JobScheduler(store=store, queue_size=8)
        status = scheduler.status("job-1")
        self.assertEqual(status["pages_total"], 1_000_000)
        self.assertGreater(store.index_reads, 0)
        self.assertEqual(store.artifact_reads, 0)
        self.assertEqual(store.tree_scans, 0)

    def test_resume_with_corrupt_or_missing_index_fails_closed(self) -> None:
        JobScheduler = require_symbol("paperscale.scheduler", "JobScheduler")
        CorruptIndexError = require_symbol("paperscale.scheduler", "CorruptIndexError")
        scheduler = JobScheduler(store=RecordingStateStore(records={}), queue_size=8)
        with self.assertRaises(CorruptIndexError):
            scheduler.resume("job-1")

    def test_lazy_queue_never_materializes_entire_large_job(self) -> None:
        JobScheduler = require_symbol("paperscale.scheduler", "JobScheduler")
        scheduler = JobScheduler(store=RecordingStateStore(), queue_size=4)
        queue = scheduler.build_lazy_queue(document_id="doc", page_count=10_000_000)
        self.assertLessEqual(len(queue.peek_buffer()), 4)
        self.assertEqual(queue.total_pages, 10_000_000)


if __name__ == "__main__":
    unittest.main()
