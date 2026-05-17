from __future__ import annotations

import unittest

from tests.harness.fakes import RecordingStateStore
from tests.harness.imports import require_symbol


class CliContractTests(unittest.TestCase):
    def test_public_v1_cli_exposes_document_markdown_commands_only(self) -> None:
        build_parser = require_symbol("paperscale.cli", "build_parser")
        parser = build_parser()
        help_text = parser.format_help()
        for command in ["run", "status", "resume", "reconcile", "fsck", "repair-index", "doctor"]:
            self.assertIn(command, help_text)
        for forbidden in ["free-ocr", "visual-qa", "extract-kv", "prompt"]:
            self.assertNotIn(forbidden, help_text)

    def test_status_command_uses_compact_index_only(self) -> None:
        CliApp = require_symbol("paperscale.cli", "CliApp")
        store = RecordingStateStore(records={"job-index": {"pages_total": 5, "succeeded": 4}})
        exit_code = CliApp(store=store).run(["status", "job-1"])
        self.assertEqual(exit_code, 0)
        self.assertGreater(store.index_reads, 0)
        self.assertEqual(store.tree_scans, 0)

    def test_ambiguous_attempts_are_operator_visible(self) -> None:
        format_ambiguous_attempts = require_symbol("paperscale.cli", "format_ambiguous_attempts")
        message = format_ambiguous_attempts(count=2, page_sample=["doc:3", "doc:9"])
        self.assertIn("ambiguous", message.lower())
        self.assertIn("duplicate", message.lower())
        self.assertIn("doc:3", message)
        self.assertIn("--retry-ambiguous", message)


if __name__ == "__main__":
    unittest.main()
