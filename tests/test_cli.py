import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_help_exposes_public_document_markdown_commands_only(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "paperscale.cli", "--help"],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("document-to-Markdown", result.stdout)
        self.assertIn("assemble", result.stdout)
        self.assertNotIn("visual-qa", result.stdout.lower())
        self.assertNotIn("key-value", result.stdout.lower())

    def test_assemble_reads_page_artifacts_and_writes_markdown_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "pages.jsonl"
            output_path = root / "document.md"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"document_id": "doc", "page_number": 2, "markdown": "World"}),
                        json.dumps({"document_id": "doc", "page_number": 1, "markdown": "# Hello"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "paperscale.cli",
                    "assemble",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--title",
                    "Demo",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            self.assertIn("assembled 2 pages", result.stdout)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "# Demo\n\n# Hello\n\n<!-- page-break -->\n\nWorld\n")


if __name__ == "__main__":
    unittest.main()
