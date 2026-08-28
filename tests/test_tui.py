"""Tests for the progress-reporter abstraction and its wiring."""

import io
import logging
import os
import re
import tempfile
import unittest
from unittest import mock

from paperscale.evaluation.runs import DocMeta, PageText
from paperscale.evaluation.textlayer import compute_textlayer_agreement
from paperscale.pipeline import _push_issue_stats
from paperscale.tui import (
    MAX_EVENT_ROWS,
    MAX_LOG_HISTORY,
    NullReporter,
    RenderStyle,
    RichReporter,
    _elapsed,
    _layout_budget,
    install_tui_logging,
    make_reporter,
    restore_console_logging,
    terminal_profile,
)
from paperscale.vllm_stats import _VLLM_ROWS, Snapshot, VLLMStats, push_vllm_stats


class _FakeTTY(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FactoryTest(unittest.TestCase):
    def test_null_when_tui_off(self):
        self.assertIsInstance(make_reporter(False, title="x", stream=_FakeTTY(True)), NullReporter)

    def test_null_when_not_a_tty(self):
        self.assertIsInstance(make_reporter(True, title="x", stream=_FakeTTY(False)), NullReporter)

    def test_rich_when_tui_and_tty(self):
        self.assertIsInstance(make_reporter(True, title="x", stream=_FakeTTY(True)), RichReporter)


class NullReporterTest(unittest.TestCase):
    def test_drive_is_noop(self):
        with NullReporter() as rep:
            rep.set_stat("pages", 3)
            ph = rep.phase("work", total=2)
            ph.advance()
            ph.advance()
            ph.done()
            rep.log("hello")  # should not raise


class RichReporterSmokeTest(unittest.TestCase):
    def test_renders_without_error(self):
        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        rep = RichReporter("paperscale evaluate", console=console)
        with rep:
            rep.set_stat("models", 2)
            ph = rep.phase("scoring", total=2)
            ph.advance()
            ph.done()
            rep.log("an event")
        self.assertIn("paperscale evaluate", buf.getvalue())


class TextlayerProgressTest(unittest.TestCase):
    def test_progress_called_once_per_page(self):
        pages = [PageText("m", "/a.pdf", 1, "x"), PageText("m", "/a.pdf", 2, "y")]
        metas = [DocMeta("m", "/a.pdf", 2, 1, "/a.pdf")]  # fallback_pages>0 -> doc skipped, but progress still fires per page
        calls = []
        rows, _ = compute_textlayer_agreement(pages, metas, progress=lambda note: calls.append(note))
        self.assertEqual(len(calls), 2)
        self.assertEqual(rows, [])  # fallback doc -> no rows


class LayoutBudgetTest(unittest.TestCase):
    def test_frame_exactly_fills_the_terminal(self):
        # The whole point: never one row more, never one row fewer.
        for height in range(1, 61):
            for n_stats in (0, 3, 8):
                for n_phases in (1, 5, 20):
                    b = _layout_budget(height, n_stats, n_phases)
                    self.assertEqual(b.total_rows(), height, f"h={height} stats={n_stats} phases={n_phases}")

    def test_never_negative(self):
        for height in range(1, 61):
            b = _layout_budget(height, 8, 20)
            self.assertGreaterEqual(b.stat_rows, 0)
            self.assertGreaterEqual(b.bar_rows, 0)
            self.assertGreaterEqual(b.event_rows, 0)

    def test_comfortable_terminal_gets_all_three_sections(self):
        b = _layout_budget(40, 4, 3)
        self.assertGreaterEqual(b.stat_rows, 3)
        self.assertGreaterEqual(b.bar_rows, 3)
        self.assertGreaterEqual(b.event_rows, 2)

    def test_events_starve_before_stats(self):
        b = _layout_budget(9, 4, 2)
        self.assertEqual(b.event_rows, 0)
        self.assertGreater(b.stat_rows, 0)

    def test_bars_survive_longest(self):
        b = _layout_budget(3, 4, 2)
        self.assertGreater(b.bar_rows, 0)
        self.assertEqual(b.stat_rows, 0)
        self.assertEqual(b.event_rows, 0)

    def test_surplus_goes_to_bars_before_events(self):
        few = _layout_budget(30, 3, 2)
        many = _layout_budget(30, 3, 12)
        self.assertGreater(many.bar_rows, few.bar_rows)

    def test_bars_never_exceed_phase_count_while_events_can_grow(self):
        b = _layout_budget(40, 3, 2)
        self.assertEqual(b.bar_rows, 2)


class TerminalProfileTest(unittest.TestCase):
    def test_utf8_xterm_gets_rich_glyphs(self):
        style = terminal_profile("utf-8", {"TERM": "xterm-256color"})
        self.assertFalse(style.ascii_only)
        self.assertEqual(style.spinner, "dots")

    def test_non_utf8_falls_back_to_ascii(self):
        style = terminal_profile("ascii", {"TERM": "xterm-256color"})
        self.assertTrue(style.ascii_only)
        self.assertEqual(style.spinner, "line")

    def test_linux_console_keeps_boxes_but_drops_braille(self):
        # The VT console renders box-drawing fine; its font has no braille block.
        style = terminal_profile("utf-8", {"TERM": "linux"})
        self.assertFalse(style.ascii_only)
        self.assertEqual(style.spinner, "line")

    def test_tmux_slows_refresh(self):
        for term in ("screen-256color", "tmux-256color"):
            style = terminal_profile("utf-8", {"TERM": term})
            self.assertEqual(style.refresh_per_second, 2, term)
            self.assertTrue(style.use_screen)

    def test_plain_terminal_refreshes_faster(self):
        self.assertEqual(terminal_profile("utf-8", {"TERM": "xterm-256color"}).refresh_per_second, 4)

    def test_env_override_forces_ascii(self):
        # Font coverage is undetectable, so the escape hatch must be explicit.
        style = terminal_profile("utf-8", {"TERM": "xterm-256color", "PAPERSCALE_TUI_ASCII": "1"})
        self.assertTrue(style.ascii_only)

    def test_env_override_forces_rich_glyphs(self):
        style = terminal_profile("ascii", {"TERM": "xterm-256color", "PAPERSCALE_TUI_ASCII": "0"})
        self.assertFalse(style.ascii_only)

    def test_missing_term_is_safe(self):
        style = terminal_profile("utf-8", {})
        self.assertIsInstance(style, RenderStyle)


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class _EncodedBuffer(io.StringIO):
    """StringIO carrying an `encoding`, which is what Rich reads to decide ASCII mode.

    `encoding` must be re-declared as a property rather than assigned: it is a
    read-only getset descriptor on `_io._TextIOBase`, so `self.encoding = ...`
    raises AttributeError even from a subclass that has a `__dict__`.
    """

    def __init__(self, encoding: str):
        super().__init__()
        self._encoding = encoding

    @property
    def encoding(self) -> str:
        return self._encoding

    def isatty(self) -> bool:
        return True


class _SizedConsole:
    """Build a Console of an exact size writing into a buffer."""

    @staticmethod
    def make(width: int, height: int, encoding: str = "utf-8"):
        from rich.console import Console

        buf = _EncodedBuffer(encoding)
        return Console(file=buf, force_terminal=True, width=width, height=height), buf


def _frame_lines(rep, console) -> list[str]:
    """One frame as visible text.

    `force_terminal=True` means capture() returns styled output, so ANSI has to
    go before any width is measured -- otherwise `len(line)` counts escape bytes
    instead of cells and the width assertion means nothing.
    """
    with console.capture() as cap:
        console.print(rep.__rich__())
    return _ANSI.sub("", cap.get()).rstrip("\n").split("\n")


def _panel_slice(lines: list[str], title: str) -> str:
    """Just the named stat panel's own columns, as text.

    The stat row is three panels side by side, so a whole-frame `assertIn` would
    pass on a `gen` that had been crushed to `ge` as long as some neighbouring
    panel happened to contain the substring. Slicing to the panel's column range
    keeps the assertion about the panel it names.
    """
    for top, line in enumerate(lines):
        start = line.find(f"╭─ {title} ")
        if start == -1:
            continue
        end = line.index("╮", start) + 1
        rows = []
        for row in lines[top:]:
            rows.append(row[start:end])
            if rows[-1].startswith("╰"):
                break
        return "\n".join(rows)
    raise AssertionError(f"no {title!r} panel in frame:\n" + "\n".join(lines))


class _StepClock:
    """Deterministic monotonic clock for VLLMStats. Advance with `tick`."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _live_vllm_stats() -> VLLMStats:
    """A `VLLMStats` whose `rates()` are the size a real vLLM server produces.

    Built from Snapshots and read back through the production code rather than
    hand-written, so nothing here can drift from what `push_vllm_stats` formats.
    Windowed: 6120 gen tok/s, 6210 prompt tok/s, 93% cache hits, 12 running.
    Since-start: 5900 and 5850.
    """
    clock = _StepClock()
    stats = VLLMStats(clock=clock)
    stats.add(Snapshot(generation_tokens=0.0, prompt_tokens=0.0, cache_hits=0.0, cache_queries=0.0, running=0.0, waiting=0.0))
    clock.tick(100.0)
    stats.add(Snapshot(generation_tokens=587_800.0, prompt_tokens=581_400.0, cache_hits=92_000.0, cache_queries=100_000.0, running=12.0, waiting=3.0))
    clock.tick(10.0)
    stats.add(Snapshot(generation_tokens=649_000.0, prompt_tokens=643_500.0, cache_hits=92_930.0, cache_queries=101_000.0, running=12.0, waiting=3.0))
    return stats


# A plausible end-of-run MetricsKeeper total. The `quality_reject_*` keys are the
# real verifier finding kinds -- `truncation_indicator` and friends are long, and
# a fixture with short ones would understate how wide the issues column gets.
_PIPELINE_TOTALS = {
    "docs_ok": 1108,
    "docs_partial": 74,
    "docs_discarded": 19,
    "docs_crashed": 3,
    "docs_missing": 0,
    "blank_pages": 1288,
    "quality_reject_repeated_ngram": 412,
    "quality_reject_truncation_indicator": 96,
    "quality_reject_repeated_character": 51,
    "quality_reject_malformed_frontmatter": 7,
}


class FixedHeightRenderTest(unittest.TestCase):
    def _populate(self, rep):
        """The panel the pipeline actually draws, pushed by the pipeline's own code.

        Every render assertion below is only worth the width of the values it is
        given. The vllm column was previously seeded with `"412 tok/s"` and
        `"93%"` while `push_vllm_stats` shipped 22-character values, so the tests
        exercised a panel about a third of its real width and never saw the key
        column starve. Both group writers are therefore called here rather than
        imitated: a fixture that cannot be written by hand cannot drift.
        """
        rep.set_stat("workspace", "/mnt/data/paperscale/legal-corpus-2026")
        rep.set_stat("docs", "1,204")
        rep.set_stat("queue", 88)
        rep.set_stat("pages", "12,048")
        rep.set_stat("tokens", "9,412,003")
        rep.set_stat("retries", "31")
        push_vllm_stats(rep, _live_vllm_stats(), None)
        _push_issue_stats(rep, _PIPELINE_TOTALS)
        for name in ("register runs", "corrections", "text-layer", "perplexity"):
            rep.phase(name, total=10)
        for i in range(20):
            rep.log(f"event line {i} " + "x" * 200)

    def test_frame_never_exceeds_pane(self):
        # 40x12 vertical split, 80x24 plain pane, 200x60 zoomed pane.
        for width, height in ((40, 12), (80, 24), (200, 60)):
            console, _ = _SizedConsole.make(width, height)
            rep = RichReporter("paperscale", console=console)
            self._populate(rep)
            lines = _frame_lines(rep, console)
            self.assertLessEqual(len(lines), height, f"{width}x{height} too tall")
            for line in lines:
                self.assertLessEqual(len(line), width, f"{width}x{height} line too wide: {line!r}")

    def test_frame_fits_every_pane_across_the_size_range(self):
        # Three sample sizes cannot stand in for the invariant. Panel counts change
        # at width 60, the stat/bar/event sections drop out one at a time as height
        # falls, and the widths where a column starts truncating differ per section
        # -- so the sweep is over both axes, at every height a section changes at.
        for width in (20, 39, 40, 59, 60, 61, 80, 100, 132, 200):
            for height in range(1, 41):
                console, _ = _SizedConsole.make(width, height)
                rep = RichReporter("paperscale", console=console)
                self._populate(rep)
                lines = _frame_lines(rep, console)
                self.assertLessEqual(len(lines), height, f"{width}x{height} too tall")
                self.assertLessEqual(max(len(line) for line in lines), width, f"{width}x{height} too wide")

    def test_vllm_panel_is_legible_at_eighty_columns(self):
        """80 columns is the default pane, and the panel has to be readable there.

        It was not: `push_vllm_stats` emitted 22-character values, which left the
        key column about 3 cells and rendered every label as `st...`, `gen`,
        `pr...`, `kv...`, `ru...`. The feature exists to surface live throughput
        and the prefix-cache hit rate, so the assertion is on the numbers being
        whole, not merely on the frame fitting.
        """
        console, _ = _SizedConsole.make(80, 24)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        panel = _panel_slice(_frame_lines(rep, console), "server")
        for label in _VLLM_ROWS:
            self.assertIn(label, panel, f"vllm label {label!r} truncated at 80 columns")
        for value in ("live", "6.1k tok/s", "6.2k tok/s", "5.9k / 5.8k", "93%", "12  wait 3"):
            self.assertIn(value, panel, f"vllm value {value!r} truncated at 80 columns")
        self.assertNotIn("…", panel, "the vllm panel still truncates at 80 columns")

    def test_header_shows_elapsed_time_on_the_title_row(self):
        console, _ = _SizedConsole.make(80, 24)
        rep = RichReporter("paperscale", console=console)
        rep._start -= 3725.0  # 01:02:05 ago
        self._populate(rep)
        header = _frame_lines(rep, console)[0]
        self.assertTrue(header.startswith("paperscale"), header)
        self.assertTrue(header.rstrip().endswith("elapsed 01:02:05"), header)
        self.assertEqual(len(header), 80)

    def test_header_stays_one_row_when_the_title_outgrows_the_pane(self):
        # HEADER_ROWS is 1 and every other section is budgeted around that, so a
        # title that wrapped would push the frame a row past the pane.
        long_title = "paperscale - " + "allenai/olmOCR-2-7B-1025-FP8-preview " * 4
        for width in (20, 40, 80, 200):
            console, _ = _SizedConsole.make(width, 24)
            rep = RichReporter(long_title, console=console)
            self._populate(rep)
            lines = _frame_lines(rep, console)
            self.assertLessEqual(len(lines[0]), width, f"width {width}: header too wide")
            self.assertLessEqual(len(lines), 24, f"width {width}: header wrapped")

    def test_budget_recomputed_between_renders(self):
        # Regression test for pane zoom and detach/reattach.
        console, _ = _SizedConsole.make(80, 30)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        tall = _frame_lines(rep, console)
        console.height = 14
        short = _frame_lines(rep, console)
        self.assertLessEqual(len(short), 14)
        self.assertLess(len(short), len(tall))

    def test_ascii_mode_emits_no_box_drawing(self):
        # Narrow widths are the whole point: truncation is clean at 80, so a
        # single 80-column case cannot see a column that still ellipsizes with
        # U+2026. The progress counters start truncating at 40 and the bar and
        # elapsed columns follow at 30 and 20.
        for width in (20, 30, 40, 80):
            console, _ = _SizedConsole.make(width, 24, encoding="ascii")
            rep = RichReporter("paperscale", console=console)
            self._populate(rep)
            text = "\n".join(_frame_lines(rep, console))
            for char in text:
                self.assertLess(ord(char), 256, f"non-Latin-1 codepoint {char!r} in ASCII mode at width {width}")

    def test_ascii_stream_renders_without_encoding_error(self):
        # The reporter is pure instrumentation: it must never take down the run
        # it reports on. Against a genuinely ascii-encoded stderr a stray U+2026
        # raises UnicodeEncodeError out of console.print, into the caller.
        # 80x24 is in the sweep on purpose: the escape that reached a user was a
        # plain default-sized pane, where the *key* column -- the one that had no
        # `overflow` -- was the only thing narrow enough to ellipsize.
        from rich.console import Console

        for width in (20, 30, 40, 60, 80, 120):
            stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="")
            console = Console(file=stream, force_terminal=True, width=width, height=24)
            rep = RichReporter("paperscale", console=console)
            self._populate(rep)
            console.print(rep.__rich__())  # must not raise UnicodeEncodeError
            stream.flush()

    def test_grouped_stats_render_under_their_own_heading(self):
        console, _ = _SizedConsole.make(120, 30)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        text = "\n".join(_frame_lines(rep, console))
        self.assertIn("server", text)
        self.assertIn("issues", text)
        self.assertIn("6.1k tok/s", text)

    def test_long_stat_values_keep_the_group_panels_on_one_row(self):
        # rich.columns.Columns re-flows panels onto extra rows as soon as their
        # combined minimum width exceeds the pane (15 rows instead of 5 at width
        # 60 with realistic vLLM values), which is the exact overflow this task
        # exists to kill. The stat region must occupy one row at any width.
        for width in (60, 80, 100, 120):
            console, _ = _SizedConsole.make(width, 24)
            rep = RichReporter("paperscale", console=console)
            rep.set_stat("documents", "1204/2900")
            push_vllm_stats(rep, _live_vllm_stats(), None)
            rep.set_stat("partial", "7 of 500 pages", group="issues")
            rep.phase("scoring", total=10)
            rep.log("an event")
            lines = _frame_lines(rep, console)
            self.assertLessEqual(len(lines), 24, f"width {width} stacked the stat panels")

    def test_tall_pane_fills_events_with_history_not_whitespace(self):
        # _layout_budget treats MAX_EVENT_ROWS as a growth target, not a cap, so
        # a tall pane hands the events panel far more than 8 rows. Retention is
        # therefore MAX_LOG_HISTORY, not MAX_EVENT_ROWS -- otherwise a 60-row
        # pane draws an 8-line panel padded with 24 blank rows.
        console, _ = _SizedConsole.make(120, 60)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        for i in range(250):
            rep.log(f"later event {i}")
        self.assertEqual(len(rep._log), MAX_LOG_HISTORY)
        lines = _frame_lines(rep, console)
        populated = [line for line in lines if "later event" in line]
        self.assertGreater(len(populated), MAX_EVENT_ROWS, "events panel is padded instead of filled")
        self.assertGreaterEqual(len(populated), 40)

    def test_embedded_newlines_do_not_add_rows(self):
        # no_wrap stops wrapping but not an embedded newline, and the bar region
        # has no panel around it to clamp the extra row.
        console, _ = _SizedConsole.make(80, 24)
        rep = RichReporter("paperscale", console=console)
        rep.set_stat("note", "a\nb")
        rep.phase("text-layer\nagreement", total=10)
        rep.log("scored a document\nand then some")
        lines = _frame_lines(rep, console)
        self.assertLessEqual(len(lines), 24)
        self.assertIn("text-layer agreement", "\n".join(lines))

    def test_bars_region_is_exactly_bar_rows_tall(self):
        # A long phase description wraps inside Progress.make_tasks_table, which
        # the width assertion cannot see (rich wraps rather than overflows). Only
        # a height assertion catches it.
        long_name = "perplexity scoring against the reference model with resume"
        for width in (40, 80, 200):
            console, _ = _SizedConsole.make(width, 24)
            rep = RichReporter("paperscale", console=console)
            for i in range(5):
                rep.phase(f"{long_name} {i}", total=10)
            for bar_rows in (1, 2, 3, 5):
                with console.capture() as cap:
                    console.print(rep._bars(bar_rows))
                lines = _ANSI.sub("", cap.get()).rstrip("\n").split("\n")
                self.assertEqual(len(lines), bar_rows, f"w={width} bar_rows={bar_rows}: {lines!r}")


class ElapsedTest(unittest.TestCase):
    def test_formats_hours_minutes_seconds(self):
        self.assertEqual(_elapsed(0), "00:00:00")
        self.assertEqual(_elapsed(59.9), "00:00:59")
        self.assertEqual(_elapsed(3725), "01:02:05")

    def test_hours_keep_counting_past_a_day(self):
        # `time.gmtime` would wrap here and restart an overnight run's clock at
        # 00:00:00, which is inside the range an OCR run reaches.
        self.assertEqual(_elapsed(26 * 3600 + 61), "26:01:01")

    def test_negative_clock_drift_is_clamped(self):
        self.assertEqual(_elapsed(-5), "00:00:00")


class SetStatGroupTest(unittest.TestCase):
    def test_null_reporter_accepts_group(self):
        with NullReporter() as rep:
            rep.set_stat("gen", 1, group="vllm")  # must not raise

    def test_default_group_is_run(self):
        console, _ = _SizedConsole.make(120, 30)
        rep = RichReporter("paperscale", console=console)
        rep.set_stat("docs", 3)
        self.assertIn("docs", rep._stats["run"])


class DumbTerminalTest(unittest.TestCase):
    def test_dumb_term_gets_null_reporter(self):
        from unittest import mock

        with mock.patch.dict("os.environ", {"TERM": "dumb"}):
            self.assertIsInstance(make_reporter(True, title="x", stream=_FakeTTY(True)), NullReporter)


class _Rep:
    def __init__(self):
        self.seen = []

    def log(self, message):
        self.seen.append(message)


class LoggingHandoffTest(unittest.TestCase):
    """The hand-off generalised: the caller's own loggers, and root always.

    The pipeline hands over two loggers that carry a shared console handler.
    evaluate owns none: everything it emits propagates to root, where
    `logging.lastResort` writes to stderr straight through the live frame if
    nothing is attached. So the no-logger call has to be as complete as the
    pipeline's, which is exactly what these tests pin.
    """

    def setUp(self):
        self._saved = [(log, list(log.handlers)) for log in (logging.getLogger(),)]

    def tearDown(self):
        for log, handlers in self._saved:
            for handler in log.handlers:
                if handler not in handlers and isinstance(handler, logging.FileHandler):
                    handler.close()
            log.handlers[:] = handlers

    def test_root_is_taken_over_when_the_caller_owns_no_logger(self):
        root = logging.getLogger()
        stderr_handler = logging.StreamHandler()
        root.addHandler(stderr_handler)
        rep = _Rep()

        install_tui_logging(rep, None)

        self.assertNotIn(stderr_handler, root.handlers)
        with mock.patch.object(logging, "lastResort") as last_resort:
            logging.getLogger("paperscale.vllm_stats").warning("statistics unavailable")
        self.assertEqual(len(rep.seen), 1)
        self.assertIn("statistics unavailable", rep.seen[0])
        last_resort.handle.assert_not_called()

    def test_caller_loggers_are_displaced_and_restored(self):
        console = logging.StreamHandler()
        owned = [logging.getLogger(f"paperscale.test.handoff.{i}") for i in (0, 1)]
        for log in owned:
            log.propagate = False
            log.handlers[:] = [console]
        self.addCleanup(lambda: [log.handlers.clear() for log in owned])

        handler = install_tui_logging(_Rep(), None, owned, console)

        for log in owned:
            self.assertNotIn(console, log.handlers)
            self.assertIn(handler, log.handlers)
        restore_console_logging(handler)
        for log in owned:
            self.assertEqual(log.handlers, [console])
        self.assertNotIn(handler, logging.getLogger().handlers)

    def test_the_log_file_lands_on_every_target(self):
        owned = logging.getLogger("paperscale.test.handoff.file")
        owned.propagate = False
        console = logging.StreamHandler()
        owned.handlers[:] = [console]
        self.addCleanup(owned.handlers.clear)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "logs", "run.log")
            handler = install_tui_logging(_Rep(), path, [owned], console)
            self.addCleanup(restore_console_logging, handler)

            self.assertTrue(os.path.exists(path))
            for target in (owned, logging.getLogger()):
                self.assertTrue(any(isinstance(h, logging.FileHandler) and h.baseFilename == path for h in target.handlers))

    def test_a_failure_opening_the_log_leaves_every_logger_untouched(self):
        """The ordering guarantee, with no caller logger: root must survive too."""
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "not-a-dir")
            with open(blocker, "w") as f:
                f.write("x")
            root = logging.getLogger()
            stderr_handler = logging.StreamHandler()
            root.addHandler(stderr_handler)
            before = list(root.handlers)

            with self.assertRaises(OSError):
                install_tui_logging(_Rep(), os.path.join(blocker, "sub", "run.log"))

            self.assertEqual(root.handlers, before)
            self.assertIn(stderr_handler, root.handlers)


if __name__ == "__main__":
    unittest.main()
