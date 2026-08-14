"""Tests for pure pipeline helpers (no server, no rendering)."""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from paperscale import pipeline
from paperscale.pipeline import PageResult, classify_document
from paperscale.prompts import PageResponse


def _page(page_num: int, text, is_fallback=False) -> PageResult:
    return PageResult(
        source_path="/docs/a.pdf",
        page_num=page_num,
        response=PageResponse(
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
            natural_text=text,
        ),
        input_tokens=3,
        output_tokens=5,
        is_fallback=is_fallback,
        is_valid=True,
    )


class TarballDetectionTests(unittest.TestCase):
    def test_detects_tarballs(self):
        self.assertTrue(pipeline.is_tarball_path("a.tar.gz"))
        self.assertTrue(pipeline.is_tarball_path("a.TGZ"))
        self.assertFalse(pipeline.is_tarball_path("a.pdf"))


class MarkdownPathTests(unittest.TestCase):
    def test_local_path_mirrors_structure(self):
        path = pipeline.get_markdown_path("/ws", "/data/sub/doc.pdf")
        self.assertEqual(path, os.path.join("/ws", "markdown", "data/sub", "doc.md"))

    def test_path_traversal_is_stripped(self):
        path = pipeline.get_markdown_path("/ws", "/../../etc/passwd.pdf")
        self.assertNotIn("..", path)
        self.assertTrue(path.endswith("passwd.md"))

    def test_tarball_member_path(self):
        path = pipeline.get_markdown_path("/ws", "/data/bundle.tar.gz::inner/doc.pdf")
        self.assertEqual(path, os.path.join("/ws", "markdown", "bundle", "inner", "doc.md"))


class BuildDolmaDocumentTests(unittest.TestCase):
    def test_assembles_text_and_spans(self):
        doc = pipeline.build_dolma_document("/docs/a.pdf", [_page(1, "page one"), _page(2, "page two")])
        self.assertEqual(doc["text"], "page one\npage two")
        self.assertEqual(doc["source"], "paperscale")
        self.assertEqual(doc["metadata"]["Source-File"], "/docs/a.pdf")
        self.assertEqual(doc["metadata"]["pdf-total-pages"], 2)
        self.assertEqual(doc["metadata"]["total-output-tokens"], 10)
        self.assertEqual(len(doc["attributes"]["pdf_page_numbers"]), 2)

    def test_counts_fallback_pages(self):
        doc = pipeline.build_dolma_document("/docs/a.pdf", [_page(1, "x"), _page(2, "y", is_fallback=True)])
        self.assertEqual(doc["metadata"]["total-fallback-pages"], 1)

    def test_empty_document_is_none(self):
        self.assertIsNone(pipeline.build_dolma_document("/docs/a.pdf", [_page(1, None)]))


class ExpandPdfInputsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_pdf(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"%PDF-1.4 minimal")
        return path

    def test_globs_pdfs(self):
        self._make_pdf("a.pdf")
        self._make_pdf("b.pdf")
        pdfs, tarballs = pipeline._expand_pdf_inputs([str(self.root / "*.pdf")])
        self.assertEqual(len(pdfs), 2)
        self.assertEqual(tarballs, set())

    def test_txt_list_is_expanded(self):
        listing = self.root / "inputs.txt"
        listing.write_text("/abs/one.pdf\n/abs/bundle.tgz\n\n")
        pdfs, tarballs = pipeline._expand_pdf_inputs([str(listing)])
        self.assertEqual(pdfs, {"/abs/one.pdf"})
        self.assertEqual(tarballs, {"/abs/bundle.tgz"})

    def test_missing_path_raises(self):
        with self.assertRaises(ValueError):
            pipeline._expand_pdf_inputs([str(self.root / "missing.pdf")])


class WipeWorkspaceTests(unittest.TestCase):
    def test_wipe_removes_progress_dirs(self):
        with tempfile.TemporaryDirectory() as ws:
            for sub in ("results", "done_flags", "worker_locks"):
                d = Path(ws) / sub
                d.mkdir()
                (d / "marker").write_text("x")
            pipeline._wipe_workspace_progress(ws)
            for sub in ("results", "done_flags", "worker_locks"):
                self.assertFalse((Path(ws) / sub).exists())


class FdExhaustionBackoffTests(unittest.IsolatedAsyncioTestCase):
    """Out-of-fd (EMFILE/ENFILE) is a soft drop: retried, never a counted failure."""

    async def test_fd_exhaustion_is_dropped_then_succeeds(self):
        good = _page(1, "ok")
        calls = {"n": 0}

        async def fake_try(args, pdf, page, attempt, image, blank):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise OSError(errno.EMFILE, "Too many open files")
            return good

        with (
            mock.patch.object(pipeline, "try_single_page", fake_try),
            mock.patch.object(pipeline.asyncio, "sleep", new=mock.AsyncMock()) as sleep,
        ):
            result = await pipeline.try_single_page_with_backoff(SimpleNamespace(), "doc.pdf", 1, attempt=0, image_base64="x", render_is_blank=False)

        # The page recovers without ever returning a failure (None) to the caller,
        # so process_page's per-page retry budget is untouched.
        self.assertIs(result, good)
        self.assertEqual(calls["n"], 3)  # two soft drops + one success
        self.assertEqual(sleep.await_count, 2)

    async def test_non_fd_connection_error_aborts_after_max_backoff(self):
        async def always_refused(*args, **kwargs):
            raise ConnectionError("connection refused")

        with (
            mock.patch.object(pipeline, "try_single_page", always_refused),
            mock.patch.object(pipeline.asyncio, "sleep", new=mock.AsyncMock()),
        ):
            with self.assertRaises(SystemExit):
                await pipeline.try_single_page_with_backoff(SimpleNamespace(), "doc.pdf", 1, attempt=0, image_base64="x", render_is_blank=False)


class ZeroPageDocumentOutcomeTest(unittest.IsolatedAsyncioTestCase):
    """A document that yielded no pages produced nothing; that's a failure, not "ok"."""

    async def test_zero_page_document_counts_as_discarded_not_ok(self):
        reader = mock.Mock()
        reader.get_num_pages.return_value = 0
        args = SimpleNamespace(apply_filter=False, max_page_error_rate=0.004)

        with (
            mock.patch.object(pipeline, "PdfReader", return_value=reader),
            mock.patch.object(pipeline, "metrics") as metrics,
            mock.patch.object(pipeline.logger, "warning") as warning,
        ):
            result = await pipeline.process_single_pdf(args, 0, "doc.pdf", "/tmp/doc.pdf")

        self.assertIsNone(result)
        metrics.add_metrics.assert_called_once_with(docs_discarded=1)
        warning.assert_called_once()
        message = warning.call_args[0][0]
        self.assertIn("doc.pdf", message)
        self.assertIn("no pages", message.lower())


class DocumentOutcomeSingleCountTest(unittest.IsolatedAsyncioTestCase):
    """Exactly one outcome counter must fire per document: never zero, never two."""

    async def test_build_dolma_document_failure_counts_only_crashed(self):
        reader = mock.Mock()
        reader.get_num_pages.return_value = 2
        args = SimpleNamespace(apply_filter=False, max_page_error_rate=0.004)

        async def fake_process_page(args, worker_id, pdf_orig_path, local_pdf_path, page_num):
            return _page(page_num, "text", is_fallback=False)

        with (
            mock.patch.object(pipeline, "PdfReader", return_value=reader),
            mock.patch.object(pipeline, "process_page", fake_process_page),
            mock.patch.object(pipeline, "build_dolma_document", side_effect=RuntimeError("boom")),
            mock.patch.object(pipeline, "metrics") as metrics,
        ):
            result = await pipeline.process_single_pdf(args, 0, "doc.pdf", "/tmp/doc.pdf")

        self.assertIsNone(result)
        # Not docs_ok *and* docs_crashed -- exactly the one call, for docs_crashed alone.
        metrics.add_metrics.assert_called_once_with(docs_crashed=1)


class ClassifyDocumentTest(unittest.TestCase):
    def test_no_fallback_pages_is_ok(self):
        self.assertEqual(classify_document(10, 0, 0.004), "ok")

    def test_some_fallback_under_threshold_is_partial(self):
        # 1/1000 = 0.001 <= 0.004: ships degraded.
        self.assertEqual(classify_document(1000, 1, 0.004), "partial")

    def test_fallback_over_threshold_is_discarded(self):
        # 9/12 = 0.75 > 0.004: dropped.
        self.assertEqual(classify_document(12, 9, 0.004), "discarded")

    def test_boundary_is_inclusive_of_partial(self):
        # Exactly at the rate is not "exceeding", matching the original
        # `> max_page_error_rate` comparison.
        self.assertEqual(classify_document(1000, 4, 0.004), "partial")

    def test_permissive_rate_keeps_everything(self):
        self.assertEqual(classify_document(10, 10, 1.0), "partial")

    def test_zero_pages_does_not_divide_by_zero(self):
        self.assertEqual(classify_document(0, 0, 0.004), "ok")


if __name__ == "__main__":
    unittest.main()
