"""Tests for pure pipeline helpers (no server, no rendering)."""

import errno
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from paperscale import pipeline
from paperscale.pipeline import PageResult
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
            result = await pipeline.try_single_page_with_backoff(
                SimpleNamespace(), "doc.pdf", 1, attempt=0, image_base64="x", render_is_blank=False
            )

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
                await pipeline.try_single_page_with_backoff(
                    SimpleNamespace(), "doc.pdf", 1, attempt=0, image_base64="x", render_is_blank=False
                )


if __name__ == "__main__":
    unittest.main()
