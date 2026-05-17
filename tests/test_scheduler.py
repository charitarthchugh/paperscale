"""Scheduler tests for compact-index and bounded-queue invariants.

Trace: plan acceptance tests 12, 13, 14, 20, and 24.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paperscale.scheduler import (
    CompactIndexError,
    ProviderCapacity,
    RetryStormController,
    Scheduler,
)


@dataclass
class FakeIndex:
    pending: list[str]
    status_payload: dict[str, int] | Exception
    resume_payload: dict[str, str] | Exception
    status_reads: int = 0
    resume_reads: int = 0
    tree_scans: int = 0

    def iter_pending_page_ids(self):
        yield from self.pending

    def read_status_index(self):
        self.status_reads += 1
        if isinstance(self.status_payload, Exception):
            raise self.status_payload
        return self.status_payload

    def read_resume_index(self):
        self.resume_reads += 1
        if isinstance(self.resume_payload, Exception):
            raise self.resume_payload
        return self.resume_payload

    def full_tree_scan(self):
        self.tree_scans += 1
        return {"rebuilt": len(self.pending)}


class SchedulerTests(unittest.TestCase):
    def test_status_reads_only_compact_index(self) -> None:
        index = FakeIndex(
            pending=[f"page-{n}" for n in range(1000)],
            status_payload={"pending": 1000, "completed": 0},
            resume_payload={},
        )
        scheduler = Scheduler(index, queue_size=4)

        self.assertEqual(scheduler.status(), {"pending": 1000, "completed": 0})
        self.assertEqual(index.status_reads, 1)
        self.assertEqual(index.tree_scans, 0)

    def test_resume_with_corrupt_index_fails_closed(self) -> None:
        index = FakeIndex(
            pending=["page-1"],
            status_payload={},
            resume_payload=ValueError("bad json"),
        )
        scheduler = Scheduler(index, queue_size=4)

        with self.assertRaises(CompactIndexError):
            scheduler.resume_plan()
        self.assertEqual(index.tree_scans, 0)

    def test_repair_index_is_explicit_tree_scan_boundary(self) -> None:
        index = FakeIndex(
            pending=["page-1", "page-2"],
            status_payload={},
            resume_payload={},
        )
        scheduler = Scheduler(index, queue_size=4)

        self.assertEqual(index.tree_scans, 0)
        self.assertEqual(scheduler.repair_index(), {"rebuilt": 2})
        self.assertEqual(index.tree_scans, 1)

    def test_lazy_queue_never_materializes_beyond_capacity(self) -> None:
        index = FakeIndex(
            pending=[f"page-{n}" for n in range(100)],
            status_payload={},
            resume_payload={},
        )
        scheduler = Scheduler(
            index,
            queue_size=10,
            capacity=ProviderCapacity(max_in_flight=2, max_provider_queue=3),
        )

        scheduler.fill_queue()
        self.assertEqual(scheduler.queued_count, 3)
        self.assertEqual(index.tree_scans, 0)

    def test_provider_overload_opens_circuit_without_queue_growth(self) -> None:
        index = FakeIndex(
            pending=[f"page-{n}" for n in range(100)],
            status_payload={},
            resume_payload={},
        )
        capacity = ProviderCapacity(
            max_in_flight=2,
            max_provider_queue=4,
            circuit_breaker_threshold=2,
        )
        scheduler = Scheduler(index, queue_size=50, capacity=capacity)
        controller = RetryStormController(capacity)

        scheduler.fill_queue()
        before = scheduler.queued_count
        controller.record_overload("429")
        controller.record_overload("503")
        scheduler.on_provider_overload(controller)
        scheduler.fill_queue()

        self.assertTrue(controller.circuit_open)
        self.assertLessEqual(scheduler.queued_count, capacity.max_provider_queue)
        self.assertEqual(scheduler.queued_count, before)


if __name__ == "__main__":
    unittest.main()
