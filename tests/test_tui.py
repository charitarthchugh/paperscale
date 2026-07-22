"""Tests for the progress-reporter abstraction and its wiring."""

import io
import unittest

from paperscale.evaluation.runs import DocMeta, PageText
from paperscale.evaluation.textlayer import compute_textlayer_agreement
from paperscale.tui import NullReporter, RichReporter, make_reporter


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


if __name__ == "__main__":
    unittest.main()
