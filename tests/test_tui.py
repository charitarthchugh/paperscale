"""Tests for the progress-reporter abstraction and its wiring."""

import io
import unittest

from paperscale.evaluation.runs import DocMeta, PageText
from paperscale.evaluation.textlayer import compute_textlayer_agreement
from paperscale.tui import Budget, NullReporter, RichReporter, _layout_budget, make_reporter  # noqa: F401


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


if __name__ == "__main__":
    unittest.main()
