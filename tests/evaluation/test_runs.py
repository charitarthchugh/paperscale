"""Tests for the Dolma-JSONL run loader."""

import tempfile
import unittest
from pathlib import Path

from paperscale.evaluation.runs import DuplicateSourceFileError, load_run
from tests.evaluation.fixtures import make_dolma_record, write_run


class LoadRunTest(unittest.TestCase):
    def test_slices_pages_by_span_and_joins_on_source_file(self):
        with tempfile.TemporaryDirectory() as d:
            rec = make_dolma_record("/docs/a.pdf", ["alpha", "beta"])
            ws = write_run(Path(d), [rec])
            pages, metas = load_run("good", ws)
        # spans include the inter-page "\n" separator, faithful to build_dolma_document
        self.assertEqual([(p.page, p.text) for p in pages], [(1, "alpha\n"), (2, "beta")])
        self.assertTrue(all(p.model == "good" and p.doc == "/docs/a.pdf" for p in pages))
        self.assertEqual(metas[0].total_pages, 2)
        self.assertEqual(metas[0].fallback_pages, 0)

    def test_skips_zero_length_blank_pages(self):
        with tempfile.TemporaryDirectory() as d:
            rec = make_dolma_record("/docs/a.pdf", ["alpha", "", "gamma"])
            ws = write_run(Path(d), [rec])
            pages, _ = load_run("m", ws)
        self.assertEqual([p.page for p in pages], [1, 3])  # blank page 2 dropped

    def test_bare_jsonl_file_input(self):
        with tempfile.TemporaryDirectory() as d:
            rec = make_dolma_record("/docs/a.pdf", ["x"])
            f = write_run(Path(d), [rec], as_workspace=False)
            pages, _ = load_run("m", f)
        self.assertEqual(len(pages), 1)

    def test_duplicate_source_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            recs = [make_dolma_record("/docs/a.pdf", ["x"]), make_dolma_record("/docs/a.pdf", ["y"])]
            ws = write_run(Path(d), recs)
            with self.assertRaises(DuplicateSourceFileError):
                load_run("m", ws)


if __name__ == "__main__":
    unittest.main()
