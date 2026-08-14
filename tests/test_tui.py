"""Tests for the progress-reporter abstraction and its wiring."""

import io
import re
import unittest

from paperscale.evaluation.runs import DocMeta, PageText
from paperscale.evaluation.textlayer import compute_textlayer_agreement
from paperscale.tui import (  # noqa: F401
    MAX_EVENT_ROWS,
    MAX_LOG_HISTORY,
    Budget,
    NullReporter,
    RenderStyle,
    RichReporter,
    _layout_budget,
    make_reporter,
    terminal_profile,
)


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


class FixedHeightRenderTest(unittest.TestCase):
    def _populate(self, rep):
        rep.set_stat("docs", "142/500")
        rep.set_stat("pages", "1204/2900")
        rep.set_stat("gen", "412 tok/s", group="vllm")
        rep.set_stat("kv hit", "93%", group="vllm")
        rep.set_stat("partial", 7, group="issues")
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
        from rich.console import Console

        stream = io.TextIOWrapper(io.BytesIO(), encoding="ascii", newline="")
        console = Console(file=stream, force_terminal=True, width=20, height=24)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        console.print(rep.__rich__())  # must not raise UnicodeEncodeError
        stream.flush()

    def test_grouped_stats_render_under_their_own_heading(self):
        console, _ = _SizedConsole.make(120, 30)
        rep = RichReporter("paperscale", console=console)
        self._populate(rep)
        text = "\n".join(_frame_lines(rep, console))
        self.assertIn("vllm", text)
        self.assertIn("issues", text)
        self.assertIn("412 tok/s", text)

    def test_long_stat_values_keep_the_group_panels_on_one_row(self):
        # rich.columns.Columns re-flows panels onto extra rows as soon as their
        # combined minimum width exceeds the pane (15 rows instead of 5 at width
        # 60 with realistic vLLM values), which is the exact overflow this task
        # exists to kill. The stat region must occupy one row at any width.
        for width in (60, 80, 100, 120):
            console, _ = _SizedConsole.make(width, 24)
            rep = RichReporter("paperscale", console=console)
            rep.set_stat("documents", "1204/2900")
            rep.set_stat("gen", "6210 tok/s  (avg 5900)", group="vllm")
            rep.set_stat("prompt", "6210 tok/s  (avg 5900)", group="vllm")
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


if __name__ == "__main__":
    unittest.main()
