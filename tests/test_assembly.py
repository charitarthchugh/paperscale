from __future__ import annotations

import unittest

from tests.harness.imports import require_symbol


class MarkdownAssemblyTests(unittest.TestCase):
    def test_page_ocr_artifact_survives_document_assembly_failure(self) -> None:
        MarkdownAssembler = require_symbol("paperscale.assembly", "MarkdownAssembler")
        PageArtifact = require_symbol("paperscale.contracts", "PageArtifact")
        AssemblyError = require_symbol("paperscale.assembly", "AssemblyError")
        artifact = PageArtifact(page_id="doc:1", markdown="# Page 1", result_pointer="artifacts/doc/1.md")
        assembler = MarkdownAssembler(required_pages=[1, 2])
        with self.assertRaises(AssemblyError):
            assembler.assemble([artifact], allow_partial=False)
        self.assertEqual(artifact.markdown, "# Page 1")
        self.assertEqual(artifact.result_pointer, "artifacts/doc/1.md")

    def test_partial_assembly_is_marked_and_cannot_masquerade_as_complete(self) -> None:
        MarkdownAssembler = require_symbol("paperscale.assembly", "MarkdownAssembler")
        PageArtifact = require_symbol("paperscale.contracts", "PageArtifact")
        artifact = PageArtifact(page_id="doc:1", markdown="# Page 1", result_pointer="artifacts/doc/1.md")
        result = MarkdownAssembler(required_pages=[1, 2]).assemble([artifact], allow_partial=True)
        self.assertTrue(result.partial)
        self.assertIn("PARTIAL", result.markdown)
        self.assertEqual(result.missing_pages, [2])

    def test_pages_are_assembled_in_document_order(self) -> None:
        MarkdownAssembler = require_symbol("paperscale.assembly", "MarkdownAssembler")
        PageArtifact = require_symbol("paperscale.contracts", "PageArtifact")
        artifacts = [
            PageArtifact(page_id="doc:2", markdown="Page 2", result_pointer="artifacts/doc/2.md"),
            PageArtifact(page_id="doc:1", markdown="Page 1", result_pointer="artifacts/doc/1.md"),
        ]
        result = MarkdownAssembler(required_pages=[1, 2]).assemble(artifacts)
        self.assertLess(result.markdown.index("Page 1"), result.markdown.index("Page 2"))
        self.assertFalse(result.partial)


if __name__ == "__main__":
    unittest.main()
