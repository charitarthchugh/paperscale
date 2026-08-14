"""End-to-end test of `paperscale evaluate` (non-pplx path)."""

import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from paperscale.cli import _evaluate_log_path, _parse_runs, build_parser, main
from tests.evaluation.fixtures import make_dolma_record, write_run

CLEAN = ["The quick brown fox jumps over the lazy dog.", "Every good boy deserves fruit today."]
GARBLED = ["Teh qckuz brwn fxo jmps ovr teh lzy dg.", "Evry gd boy dsrvs frt3 tdy xkcdqz."]


class ParseRunsTest(unittest.TestCase):
    def test_rejects_missing_equals(self):
        with self.assertRaises(SystemExit):
            _parse_runs(["nopath"])

    def test_rejects_duplicate_label(self):
        with self.assertRaises(SystemExit):
            _parse_runs(["a=/x", "a=/y"])


class EvaluateE2ETest(unittest.TestCase):
    def test_two_runs_ranks_good_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            bad = write_run(root / "bad", [make_dolma_record("/docs/a.pdf", GARBLED)])
            db_path = root / "eval.sqlite"

            rc = main(["evaluate", "--run", f"good={good}", "--run", f"bad={bad}", "--db", str(db_path)])
            self.assertEqual(rc, 0)

            con = sqlite3.connect(db_path)
            # both runs registered
            self.assertEqual({r[0] for r in con.execute("SELECT model FROM runs")}, {"good", "bad"})
            # good's text needs fewer spell-corrections than bad's (lower is better)
            means = dict(con.execute("SELECT model, AVG(correction_rate) FROM correction_rate GROUP BY model"))
            self.assertLess(means["good"], means["bad"])
            # peer agreement rows exist (2 models, shared doc/pages)
            self.assertGreater(con.execute("SELECT COUNT(*) FROM peer_agreement").fetchone()[0], 0)
            con.close()

    def test_single_run_skips_peer_agreement(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good = write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])
            rc = main(["evaluate", "--run", f"good={good}", "--db", str(root / "eval.sqlite")])
            self.assertEqual(rc, 0)
            con = sqlite3.connect(root / "eval.sqlite")
            self.assertEqual(con.execute("SELECT COUNT(*) FROM peer_agreement").fetchone()[0], 0)
            con.close()


class EvaluateTuiFlagsTest(unittest.TestCase):
    def test_tui_defaults_off(self):
        args = build_parser().parse_args(["evaluate", "--run", "a=/tmp/a.jsonl"])
        self.assertFalse(args.tui)

    def test_poll_interval_default(self):
        args = build_parser().parse_args(["evaluate", "--run", "a=/tmp/a.jsonl"])
        self.assertEqual(args.tui_poll_interval, 5.0)

    def test_disk_logging_defaults_to_none(self):
        args = build_parser().parse_args(["evaluate", "--run", "a=/tmp/a.jsonl"])
        self.assertIsNone(args.disk_logging)

    def test_disk_logging_flag(self):
        args = build_parser().parse_args(["evaluate", "--run", "a=/tmp/a.jsonl", "--disk-logging", "/tmp/e.log"])
        self.assertEqual(args.disk_logging, "/tmp/e.log")

    def test_log_path_defaults_beside_the_db(self):
        path = _evaluate_log_path(Path("/tmp/eval/evaluation.sqlite"))
        self.assertTrue(path.startswith("/tmp/eval/logs/evaluate-"))
        self.assertTrue(path.endswith(".log"))


class _FakeLiveReporter:
    """A reporter that is not a NullReporter, so _handle_evaluate takes the live path."""

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


class _WarningReporter(_FakeLiveReporter):
    """Emits a warning from inside the live block, the way a dependency would.

    vllm_stats' "statistics unavailable" is the real case: a handler-less module
    logger propagating to root while the frame owns the screen.
    """

    def phase(self, name, total=None):
        logging.getLogger("paperscale.vllm_stats").warning("probe from %s", name)
        return super().phase(name, total)


def _one_run(root: Path):
    return write_run(root / "good", [make_dolma_record("/docs/a.pdf", CLEAN)])


class EvaluateNoTuiIsInvisibleTest(unittest.TestCase):
    """Without --tui, evaluate must behave exactly as it did before the dashboard.

    The hand-off strips every handler off the root logger, so if it ran on the
    default path anything logging through root would go silent -- and a logs/
    directory would appear beside a database the user never asked to instrument.
    """

    def test_default_run_leaves_root_logging_alone_and_writes_no_log_dir(self):
        root_logger = logging.getLogger()
        before = list(root_logger.handlers)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            rc = main(["evaluate", "--run", f"good={_one_run(base)}", "--db", str(base / "eval.sqlite")])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(base / "logs"))
        self.assertEqual(root_logger.handlers, before)


class EvaluateTuiLoggingTest(unittest.TestCase):
    """--tui routes evaluate's logging to a file and puts root back afterwards."""

    def setUp(self):
        self._saved = [(log, list(log.handlers)) for log in (logging.getLogger(),)]

    def tearDown(self):
        for log, handlers in self._saved:
            for handler in log.handlers:
                if handler not in handlers and isinstance(handler, logging.FileHandler):
                    handler.close()
            log.handlers[:] = handlers

    def _run(self, base: Path, extra, rep=None):
        rep = rep if rep is not None else _FakeLiveReporter()
        with mock.patch("paperscale.tui.make_reporter", return_value=rep):
            rc = main(["evaluate", "--run", f"good={_one_run(base)}", "--db", str(base / "eval.sqlite"), "--tui", *extra])
        self.assertEqual(rc, 0)
        return rep

    def _file_handlers_on(self, path):
        root = logging.getLogger()
        return {id(h) for h in root.handlers if isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(path)}

    def test_log_lands_beside_the_db_and_root_is_restored(self):
        stderr_handler = logging.StreamHandler()
        logging.getLogger().addHandler(stderr_handler)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            self._run(base, [])
            logs = base / "logs"
            self.assertTrue(logs.is_dir())
            self.assertEqual(len(list(logs.glob("evaluate-*.log"))), 1)
        # The displaced console handler is back and the reporter's is gone.
        self.assertIn(stderr_handler, logging.getLogger().handlers)
        self.assertFalse([h for h in logging.getLogger().handlers if type(h).__name__ == "ReporterLogHandler"])

    def test_explicit_disk_logging_opens_exactly_one_handler(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            log_path = base / "explicit.log"
            self._run(base, ["--disk-logging", str(log_path)])
            # One handler, not two: a file opened twice writes every record twice.
            self.assertEqual(len(self._file_handlers_on(log_path)), 1)
            self.assertFalse(os.path.exists(base / "logs"))

    def test_warnings_reach_the_users_log_file_while_the_frame_is_live(self):
        """A file attached to root before the hand-off would be stripped off by it.

        The hand-off displaces *every* handler on root -- which is exactly where
        evaluate's file has to live -- so the user's --disk-logging has to be
        opened by the hand-off itself, not before it. Otherwise the flag names a
        file that stays empty for the whole run.
        """
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            log_path = base / "explicit.log"
            rep = self._run(base, ["--disk-logging", str(log_path)], rep=_WarningReporter())
            written = log_path.read_text(encoding="utf-8")
        self.assertIn("probe from register runs", written)
        # And the same warning reached the event pane rather than stderr.
        self.assertTrue([line for line in rep.logs if "probe from register runs" in line])

    def test_disk_logging_without_tui_still_writes_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            log_path = base / "plain.log"
            rc = main(["evaluate", "--run", f"good={_one_run(base)}", "--db", str(base / "eval.sqlite"), "--disk-logging", str(log_path)])
            self.assertEqual(rc, 0)
            self.assertTrue(log_path.exists())


class EvaluatePplxPanelTest(unittest.TestCase):
    """--tui --pplx feeds the shared vllm panel and the issues column."""

    def setUp(self):
        self._saved = [(log, list(log.handlers)) for log in (logging.getLogger(),)]

    def tearDown(self):
        for log, handlers in self._saved:
            for handler in log.handlers:
                if handler not in handlers and isinstance(handler, logging.FileHandler):
                    handler.close()
            log.handlers[:] = handlers

    def test_panel_and_issues_are_pushed_as_docs_complete(self):
        def fake_score(pages, *, on_doc=None, progress=None, **kwargs):
            for doc in sorted({p.doc for p in pages}):
                on_doc(doc, [])
                progress(doc)
            return {}

        rep = _FakeLiveReporter()
        poller = mock.Mock(available=False)
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            argv = [
                "evaluate",
                "--run",
                f"good={_one_run(base)}",
                "--db",
                str(base / "eval.sqlite"),
                "--tui",
                "--pplx",
                "--pplx-model",
                "scorer",
                "--pplx-url",
                "http://scorer:8000",
            ]
            with (
                mock.patch("paperscale.tui.make_reporter", return_value=rep),
                mock.patch("paperscale.evaluation.pplx.score_run_pplx", fake_score),
                mock.patch("paperscale.vllm_stats.VLLMStatsPoller", return_value=poller) as poller_cls,
            ):
                self.assertEqual(main(argv), 0)

        self.assertEqual(poller_cls.call_args.args[0], "http://scorer:8000/metrics")
        self.assertEqual(poller_cls.call_args.kwargs["interval"], 5.0)
        poller.start.assert_called_once()
        poller.stop.assert_called_once()
        # An unavailable scraper writes the whole row set, never a stale subset.
        self.assertEqual(rep.stats[("vllm", "status")], "unavailable")
        self.assertEqual(rep.stats[("vllm", "gen")], "-")
        self.assertEqual(rep.stats[("issues", "skipped")], 0)

    def test_no_poller_without_pplx(self):
        rep = _FakeLiveReporter()
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            with (
                mock.patch("paperscale.tui.make_reporter", return_value=rep),
                mock.patch("paperscale.vllm_stats.VLLMStatsPoller") as poller_cls,
            ):
                main(["evaluate", "--run", f"good={_one_run(base)}", "--db", str(base / "eval.sqlite"), "--tui"])
        poller_cls.assert_not_called()
        self.assertNotIn(("vllm", "status"), rep.stats)


if __name__ == "__main__":
    unittest.main()
