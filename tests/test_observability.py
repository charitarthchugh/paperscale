from __future__ import annotations

import hashlib
import logging
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.observability import MetricsKeeper, configure_logging, format_duration, get_logger
from paperscale.providers.async_openai_chat import _usage
from paperscale.providers.base import PageOcrResponse
from paperscale.runner import DocumentOcrRunner, RunnerConfig


class MetricsKeeperTests(unittest.TestCase):
    def test_cumulative_and_window_rates_with_fake_clock(self) -> None:
        now = [1000.0]
        m = MetricsKeeper(window_seconds=10.0, clock=lambda: now[0])
        now[0] = 1001.0
        m.add(pages_completed=2, output_tokens=100)
        now[0] = 1002.0
        m.add(pages_completed=2, output_tokens=100)
        # cumulative: 4 pages over 2s elapsed.
        self.assertAlmostEqual(m.cumulative_rate("pages_completed", now=1002.0), 2.0, places=3)
        # window (10s) not yet full -> span is elapsed (2s): 4/2.
        self.assertAlmostEqual(m.window_rate("pages_completed", now=1002.0), 2.0, places=3)
        self.assertEqual(m.get("output_tokens"), 200)

    def test_window_trims_old_events(self) -> None:
        now = [0.0]
        m = MetricsKeeper(window_seconds=10.0, clock=lambda: now[0])
        now[0] = 1.0
        m.add(pages_completed=8)  # falls outside the window once we advance past t=11
        now[0] = 20.0
        m.add(pages_completed=1)
        # only the t=20 event is inside [10, 20]; span=min(elapsed=20, window=10)=10.
        self.assertAlmostEqual(m.window_rate("pages_completed", now=20.0), 0.1, places=3)
        # cumulative still counts everything: 9 / 20.
        self.assertAlmostEqual(m.cumulative_rate("pages_completed", now=20.0), 0.45, places=3)


class LoggingConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging(quiet=False)  # restore default for other tests

    def test_default_enables_info_quiet_suppresses_it(self) -> None:
        logger = configure_logging(quiet=False)
        self.assertTrue(logger.isEnabledFor(logging.INFO))
        logger = configure_logging(quiet=True)
        self.assertFalse(logger.isEnabledFor(logging.INFO))
        self.assertTrue(logger.isEnabledFor(logging.WARNING))

    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(45), "45s")
        self.assertEqual(format_duration(90), "1m30s")
        self.assertEqual(format_duration(3700), "1h01m")


class UsageParsingTests(unittest.TestCase):
    def test_usage_from_object_dict_and_missing(self) -> None:
        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Resp:
            usage = _Usage()

        self.assertEqual(_usage(_Resp()), (10, 5))
        self.assertEqual(_usage({"usage": {"prompt_tokens": 3, "completion_tokens": 4}}), (3, 4))
        self.assertEqual(_usage(object()), (0, 0))


@dataclass(frozen=True)
class _Rendered:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class _Renderer:
    @property
    def page_count(self) -> int:
        return 1

    def render_page(self, page_number: int) -> _Rendered:
        image = b"page"
        return _Rendered(page_number, image, hashlib.sha256(image).hexdigest())


class _Provider:
    name = "fake"

    def send(self, request: Any) -> PageOcrResponse:
        return PageOcrResponse(
            markdown="# Page\n\nReal body text.",
            provider_request_id=request.fingerprint,
            metadata={"input_tokens": 100, "output_tokens": 20},
        )


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class RunProgressLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure_logging(quiet=False)

    def _run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / ".paperscale"
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=root, base_url="http://fake/v1", model="m", max_in_flight_requests=1),
                provider=_Provider(),
                renderer_factory=lambda _p, _o: _Renderer(),
                sleeper=lambda _s: None,
            )
            status = runner.run(input_path=root / "i.pdf", output_path=root / "o.md", job_id="j")
            self.assertEqual(status.succeeded, 1)

    def test_final_summary_logged_by_default(self) -> None:
        configure_logging(quiet=False)
        handler = _CapturingHandler()
        get_logger().addHandler(handler)
        try:
            self._run()
        finally:
            get_logger().removeHandler(handler)
        self.assertTrue(any("done" in m and "pages succeeded" in m for m in handler.messages))

    def test_quiet_suppresses_progress_logging(self) -> None:
        configure_logging(quiet=True)
        handler = _CapturingHandler()
        get_logger().addHandler(handler)
        try:
            self._run()
        finally:
            get_logger().removeHandler(handler)
        self.assertEqual(handler.messages, [])


if __name__ == "__main__":
    unittest.main()
