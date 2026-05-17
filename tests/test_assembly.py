import unittest

from paperscale.assembly import PageMarkdownArtifact, assemble_document_markdown


class AssemblyTests(unittest.TestCase):
    def test_assembles_successful_pages_in_document_order(self) -> None:
        markdown = assemble_document_markdown(
            [
                PageMarkdownArtifact(document_id="doc-1", page_number=2, markdown="Second page"),
                PageMarkdownArtifact(document_id="doc-1", page_number=1, markdown="# Title"),
            ],
            title="Example",
        )

        self.assertEqual(markdown, "# Example\n\n# Title\n\n<!-- page-break -->\n\nSecond page\n")

    def test_rejects_duplicate_or_cross_document_pages(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate page"):
            assemble_document_markdown(
                [
                    PageMarkdownArtifact(document_id="doc-1", page_number=1, markdown="A"),
                    PageMarkdownArtifact(document_id="doc-1", page_number=1, markdown="B"),
                ]
            )

        with self.assertRaisesRegex(ValueError, "single document"):
            assemble_document_markdown(
                [
                    PageMarkdownArtifact(document_id="doc-1", page_number=1, markdown="A"),
                    PageMarkdownArtifact(document_id="doc-2", page_number=2, markdown="B"),
                ]
            )

    def test_quality_gate_can_block_bad_fragments(self) -> None:
        with self.assertRaisesRegex(ValueError, "quality check failed"):
            assemble_document_markdown(
                [PageMarkdownArtifact(document_id="doc-1", page_number=1, markdown="Noise " * 100)],
                enforce_quality=True,
            )


if __name__ == "__main__":
    unittest.main()
