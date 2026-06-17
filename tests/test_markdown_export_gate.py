"""Confirmation test: is per-document Markdown actually exported after a run?

Drives the real ``pipeline.worker`` output path with the inference layer
(``process_pdf``) mocked out, so it runs with no server/GPU. The point is to
observe what lands on disk when ``--markdown`` is and isn't passed.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from paperscale import pipeline
from paperscale.pipeline import PageResult
from paperscale.prompts import PageResponse
from paperscale.work_queue import WorkItem


def _doc(source_file: str):
    page = PageResult(
        source_path=source_file,
        page_num=1,
        response=PageResponse(
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
            natural_text="hello world",
        ),
        input_tokens=3,
        output_tokens=5,
        is_fallback=False,
        is_valid=True,
    )
    return pipeline.build_dolma_document(source_file, [page])


class MarkdownExportGateTests(unittest.IsolatedAsyncioTestCase):
    async def _run_worker(self, workspace: str, markdown: bool):
        source_file = "/data/sub/doc.pdf"
        item = WorkItem(hash="deadbeef", work_paths=[source_file])

        # work_queue: hand out one item, then signal "drain" with None.
        work_queue = mock.Mock()
        work_queue.get_work = mock.AsyncMock(side_effect=[item, None])
        work_queue.mark_done = mock.AsyncMock()

        args = SimpleNamespace(workspace=workspace, markdown=markdown)

        with (
            mock.patch.object(pipeline, "process_pdf", mock.AsyncMock(return_value=_doc(source_file))),
            mock.patch.object(pipeline.tracker, "clear_work", mock.AsyncMock()),
            mock.patch.object(pipeline.metrics, "add_metrics"),
        ):
            await pipeline.worker(args, work_queue, worker_id=0)

        work_queue.mark_done.assert_awaited_once()

    async def test_without_flag_no_markdown_is_written(self):
        with tempfile.TemporaryDirectory() as ws:
            await self._run_worker(ws, markdown=False)
            md_files = list(Path(ws).rglob("*.md"))
            jsonl = list(Path(ws).rglob("*.jsonl"))
            # The dolma JSONL is ALWAYS written...
            self.assertEqual(len(jsonl), 1, f"expected one results jsonl, got {jsonl}")
            # ...but NO markdown at all without the flag. This is the reported symptom.
            self.assertEqual(md_files, [], f"unexpected markdown written: {md_files}")

    async def test_with_flag_markdown_lands_in_nested_mirror(self):
        with tempfile.TemporaryDirectory() as ws:
            await self._run_worker(ws, markdown=True)
            md_files = list(Path(ws).rglob("*.md"))
            self.assertEqual(len(md_files), 1, f"expected one markdown file, got {md_files}")
            # It mirrors the FULL input path under <ws>/markdown/, not a flat dir.
            self.assertEqual(md_files[0], Path(ws) / "markdown" / "data" / "sub" / "doc.md")
            self.assertEqual(md_files[0].read_text(), "hello world")


class ExportMarkdownFromResultsTests(unittest.TestCase):
    """`--export-markdown`: regenerate Markdown from existing results/*.jsonl, no inference."""

    def _write_results(self, ws: str, name: str, docs):
        results_dir = Path(ws) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / name, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

    def test_exports_each_doc_to_its_mirrored_path(self):
        with tempfile.TemporaryDirectory() as ws:
            self._write_results(
                ws,
                "output_aaa.jsonl",
                [_doc("/data/sub/one.pdf"), _doc("/data/other/two.pdf")],
            )
            count = pipeline.export_markdown_from_results(SimpleNamespace(workspace=ws))
            self.assertEqual(count, 2)
            self.assertEqual(
                (Path(ws) / "markdown" / "data" / "sub" / "one.md").read_text(), "hello world"
            )
            self.assertTrue((Path(ws) / "markdown" / "data" / "other" / "two.md").exists())

    def test_skips_blank_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as ws:
            results_dir = Path(ws) / "results"
            results_dir.mkdir(parents=True)
            with open(results_dir / "output_bbb.jsonl", "w") as f:
                f.write("\n")  # blank
                f.write("{not valid json}\n")  # malformed
                f.write(json.dumps(_doc("/data/ok.pdf")) + "\n")  # valid
            count = pipeline.export_markdown_from_results(SimpleNamespace(workspace=ws))
            self.assertEqual(count, 1)
            self.assertTrue((Path(ws) / "markdown" / "data" / "ok.md").exists())

    def test_missing_results_dir_is_noop(self):
        with tempfile.TemporaryDirectory() as ws:
            count = pipeline.export_markdown_from_results(SimpleNamespace(workspace=ws))
            self.assertEqual(count, 0)
            self.assertFalse((Path(ws) / "markdown").exists())


if __name__ == "__main__":
    unittest.main()
