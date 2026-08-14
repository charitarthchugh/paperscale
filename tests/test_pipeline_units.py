"""Tests for pure pipeline helpers (no server, no rendering)."""

import errno
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from paperscale import pipeline
from paperscale.pipeline import PageResult, _build_arg_parser, _install_tui_logging, _tui_log_path, classify_document, count_documents, count_retries
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


class TuiLoggingTest(unittest.TestCase):
    def setUp(self):
        # _install_tui_logging rewires process-global loggers: it strips
        # console_handler off two of them and attaches its own to three, root
        # included. Left in place, every later test in the run would execute
        # against a logger with no stderr handler -- and the resulting failure
        # would surface somewhere unrelated, since pytest decides the ordering.
        # Snapshot every handler list here and restore them exactly in tearDown.
        self._saved_handlers = [(log, list(log.handlers)) for log in (pipeline.logger, pipeline.server_logger, logging.getLogger())]

    def tearDown(self):
        for log, handlers in self._saved_handlers:
            log.handlers[:] = handlers

    def test_log_path_lives_under_the_workspace(self):
        path = _tui_log_path("/tmp/ws")
        self.assertTrue(path.startswith("/tmp/ws/logs/run-"))
        self.assertTrue(path.endswith(".log"))

    def test_handler_forwards_warnings_to_the_reporter(self):
        seen = []

        class _Rep:
            def log(self, message):
                seen.append(message)

        handler = _install_tui_logging(_Rep(), None)
        record = logging.LogRecord("x", logging.WARNING, "f", 1, "disk is full", None, None)
        handler.emit(record)
        self.assertEqual(len(seen), 1)
        self.assertIn("disk is full", seen[0])

    def test_handler_ignores_info(self):
        seen = []

        class _Rep:
            def log(self, message):
                seen.append(message)

        handler = _install_tui_logging(_Rep(), None)
        handler.emit(logging.LogRecord("x", logging.INFO, "f", 1, "chatter", None, None))
        self.assertEqual(seen, [])

    def test_other_module_loggers_reach_the_pane_not_stderr(self):
        """work_queue, check, filter and vllm_stats own no handlers of their own.

        They propagate to root, where paperscale.filter's import-time
        logging.basicConfig() left a stderr StreamHandler -- which prints straight
        through the live frame. Root has to be taken over as well, and it must
        keep a handler afterwards so logging.lastResort does not step in and write
        to stderr instead.
        """
        seen = []

        class _Rep:
            def log(self, message):
                seen.append(message)

        root = logging.getLogger()
        stderr_handler = logging.StreamHandler()
        root.addHandler(stderr_handler)

        _install_tui_logging(_Rep(), None)
        self.assertNotIn(stderr_handler, root.handlers)
        with mock.patch.object(logging, "lastResort") as last_resort:
            logging.getLogger("paperscale.work_queue").warning("done flag failed")
        self.assertEqual(len(seen), 1)
        self.assertIn("done flag failed", seen[0])
        last_resort.handle.assert_not_called()

    def test_restore_puts_console_logging_back_everywhere(self):
        root = logging.getLogger()
        stderr_handler = logging.StreamHandler()
        root.addHandler(stderr_handler)
        root_before = list(root.handlers)

        handler = _install_tui_logging(mock.Mock(), None)
        pipeline._restore_console_logging(handler)

        for log in (pipeline.logger, pipeline.server_logger):
            self.assertIn(pipeline.console_handler, log.handlers)
            self.assertNotIn(handler, log.handlers)
        self.assertNotIn(handler, root.handlers)
        self.assertEqual(sorted(map(id, root.handlers)), sorted(map(id, root_before)))


class TuiFlagTest(unittest.TestCase):
    def test_tui_defaults_off(self):
        args = _build_arg_parser().parse_args(["/tmp/ws"])
        self.assertFalse(args.tui)

    def test_tui_flag_parses(self):
        args = _build_arg_parser().parse_args(["/tmp/ws", "--tui"])
        self.assertTrue(args.tui)

    def test_poll_interval_default(self):
        args = _build_arg_parser().parse_args(["/tmp/ws"])
        self.assertEqual(args.tui_poll_interval, 5.0)


class CountRetriesTest(unittest.TestCase):
    def test_first_attempt_successes_are_not_retries(self):
        self.assertEqual(count_retries({"finished_on_attempt_0": 500}), 0)

    def test_later_attempts_count(self):
        self.assertEqual(count_retries({"finished_on_attempt_0": 500, "finished_on_attempt_1": 30, "finished_on_attempt_2": 8}), 38)

    def test_parallel_retries_included(self):
        self.assertEqual(count_retries({"finished_on_attempt_1": 2, "finished_on_parallel_retry": 5}), 7)

    def test_unrelated_metrics_ignored(self):
        self.assertEqual(count_retries({"completed_pages": 900, "blank_pages": 3}), 0)

    def test_non_numeric_suffix_ignored(self):
        self.assertEqual(count_retries({"finished_on_attempt_parallel": 4}), 0)


class NoTuiIsInvisibleTest(unittest.IsolatedAsyncioTestCase):
    """Without --tui the run must behave exactly as it did before the dashboard.

    Two things could leak. `NullReporter.phase()` prints a `[name]` header
    straight to `sys.stderr` -- correct for evaluate, which always printed one,
    but output this pipeline never produced. And `_install_tui_logging` strips
    `console_handler`, which would silence stderr logging entirely. Neither may
    happen on the default path, so this drives `main()` end to end with the
    inference parts stubbed and asserts nothing reached `sys.stderr` directly and
    both loggers still hold their original handlers.
    """

    async def test_default_run_writes_nothing_direct_to_stderr(self):
        import contextlib
        import io

        before = [(log, list(log.handlers)) for log in (pipeline.logger, pipeline.server_logger)]
        queue = mock.Mock()
        queue.initialize_queue = mock.AsyncMock(return_value=3)
        queue.size = 3

        with tempfile.TemporaryDirectory() as ws:
            argv = ["paperscale", ws, "--server", "http://example.invalid/v1", "--workers", "2"]
            with (
                mock.patch.object(pipeline.sys, "argv", argv),
                mock.patch.object(pipeline, "check_poppler_version"),
                mock.patch.object(pipeline, "WorkQueue", return_value=queue),
                mock.patch.object(pipeline, "vllm_server_ready", new=mock.AsyncMock()),
                mock.patch.object(pipeline, "worker", new=mock.AsyncMock()) as fake_worker,
                mock.patch.object(pipeline, "metrics_reporter", new=mock.AsyncMock()) as fake_reporter,
            ):
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    await pipeline.main()

        # `[work items]` (or anything else) reaching sys.stderr means the reporter
        # spoke on a path that used to be silent.
        self.assertEqual(buf.getvalue(), "")
        # The old-shaped call: no live reporter, so metrics_reporter keeps logging.
        self.assertEqual(fake_reporter.await_args.args[1:], (None, None, None))
        # worker() must be handed no phase, so it stays byte-identical to before.
        self.assertEqual(fake_worker.await_count, 2)
        for call in fake_worker.await_args_list:
            self.assertIsNone(call.kwargs["phase"])
        # stderr logging survived: console_handler never left either logger.
        for log, handlers in before:
            self.assertEqual(log.handlers, handlers)
            self.assertIn(pipeline.console_handler, log.handlers)
        # No log directory was conjured under the workspace either.
        self.assertFalse(os.path.exists(os.path.join(ws, "logs")))


class CountDocumentsTest(unittest.TestCase):
    def test_sums_every_outcome_counter(self):
        totals = {"docs_ok": 40, "docs_partial": 3, "docs_discarded": 2, "docs_crashed": 1, "docs_missing": 4}
        self.assertEqual(count_documents(totals), 50)

    def test_missing_counters_are_zero(self):
        self.assertEqual(count_documents({"docs_ok": 7}), 7)

    def test_unrelated_metrics_ignored(self):
        self.assertEqual(count_documents({"completed_pages": 900, "server_output_tokens": 12}), 0)

    def test_empty_totals_is_zero(self):
        self.assertEqual(count_documents({}), 0)


class _FakeLiveReporter:
    """A reporter that is not a NullReporter, so main() takes the live path."""

    def __init__(self):
        self.stats = {}
        self.logs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def phase(self, name, total=None):
        return mock.Mock()

    def log(self, message):
        self.logs.append(message)

    def set_stat(self, name, value, *, group="run"):
        self.stats[(group, name)] = value


class DocsStatTest(unittest.IsolatedAsyncioTestCase):
    """The `docs` row must track the outcome counters, not a value fixed at startup."""

    async def _one_tick(self, rep, totals, queue_size=7):
        class _Stop(Exception):
            pass

        with (
            mock.patch.object(pipeline, "metrics") as metrics,
            mock.patch.object(pipeline.asyncio, "sleep", side_effect=_Stop),
        ):
            metrics.get_total_metrics.return_value = totals
            with self.assertRaises(_Stop):
                await pipeline.metrics_reporter(SimpleNamespace(size=queue_size), rep)

    async def test_docs_reflects_the_counters_and_carries_no_denominator(self):
        rep = _FakeLiveReporter()
        await self._one_tick(rep, {"docs_ok": 40, "docs_partial": 3, "docs_discarded": 2, "docs_crashed": 1, "docs_missing": 4})
        self.assertEqual(rep.stats[("run", "docs")], "50")
        # "0/500" was the bug: work items are page groups, not documents.
        self.assertNotIn("/", rep.stats[("run", "docs")])

    async def test_docs_moves_between_ticks(self):
        rep = _FakeLiveReporter()
        await self._one_tick(rep, {"docs_ok": 2})
        self.assertEqual(rep.stats[("run", "docs")], "2")
        await self._one_tick(rep, {"docs_ok": 1200, "docs_crashed": 34})
        self.assertEqual(rep.stats[("run", "docs")], "1,234")


class TuiCleanupTest(unittest.IsolatedAsyncioTestCase):
    """A crash mid-run must not leave the process with no stderr logging."""

    async def test_crash_still_stops_the_poller_and_restores_logging(self):
        before = [(log, list(log.handlers)) for log in (pipeline.logger, pipeline.server_logger, logging.getLogger())]
        self.addCleanup(lambda: [log.handlers.__setitem__(slice(None), h) for log, h in before])

        queue = mock.Mock()
        queue.initialize_queue = mock.AsyncMock(return_value=1)
        queue.size = 1
        poller = mock.Mock()
        poller.available = False
        reporter = _FakeLiveReporter()

        async def boom(*a, **kw):
            raise RuntimeError("worker exploded")

        with tempfile.TemporaryDirectory() as ws:
            argv = ["paperscale", ws, "--server", "http://example.invalid/v1", "--workers", "1", "--tui"]
            with (
                mock.patch.object(pipeline.sys, "argv", argv),
                mock.patch.object(pipeline, "check_poppler_version"),
                mock.patch.object(pipeline, "WorkQueue", return_value=queue),
                mock.patch.object(pipeline, "vllm_server_ready", new=mock.AsyncMock()),
                mock.patch.object(pipeline, "worker", new=boom),
                mock.patch("paperscale.tui.make_reporter", return_value=reporter),
                mock.patch("paperscale.vllm_stats.VLLMStatsPoller", return_value=poller),
            ):
                with self.assertRaises(RuntimeError):
                    await pipeline.main()

            log_dir = os.path.join(ws, "logs")
            self.assertTrue(os.path.isdir(log_dir))
            self.assertTrue(os.listdir(log_dir))

        poller.start.assert_called_once()
        poller.stop.assert_called_once()
        # main() must not seed `docs` with a startup constant. Deterministic either
        # way: absent if no tick ran, otherwise a plain count from the counters.
        self.assertNotIn("/", str(reporter.stats.get(("run", "docs"), "")))
        # Every displaced handler is back, so a crashed run still logs to stderr.
        for log, handlers in before:
            for handler in handlers:
                self.assertIn(handler, log.handlers)


if __name__ == "__main__":
    unittest.main()
