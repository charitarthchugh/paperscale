from __future__ import annotations

import unittest

from tests.harness.fakes import FakeProvider, RecordingResourceGovernor, RecordingStateStore
from tests.harness.imports import require_symbol


class EndToEndFakeJobTests(unittest.TestCase):
    def test_fake_pdf_to_markdown_job_preserves_reservation_resource_and_no_network_invariants(self) -> None:
        FakeDocumentRunner = require_symbol("paperscale.testing", "FakeDocumentRunner")
        store = RecordingStateStore()
        resources = RecordingResourceGovernor()
        provider = FakeProvider()
        runner = FakeDocumentRunner(store=store, resources=resources, provider=provider)
        result = runner.run_document_to_markdown(document_id="doc", pages=[b"fake-page-image"])
        self.assertIn("Hello from fake OCR", result.markdown)
        self.assertIn(("attempt_reserved", "doc:1:attempt-1"), store.mutations)
        self.assertEqual(len(provider.calls), 1)
        self.assertLess(
            store.events.index("read_index:ledger-reservation-doc:1:attempt-1"),
            provider.calls[0].debug_event_index,
            "reservation durability must be observable before provider call",
        )
        self.assertIn("acquire:provider_concurrency", resources.events)
        self.assertNotIn("network", result.debug_side_effects)


if __name__ == "__main__":
    unittest.main()
