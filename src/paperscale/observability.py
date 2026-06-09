"""Logging + throughput metrics for OCR runs (olmOCR-style statistics).

Mirrors the statistics olmOCR surfaces during a pipeline run: cumulative totals and
**sliding-window rates** for pages and input/output tokens, plus queue-remaining,
failure rate, elapsed, and ETA. Logging is on by default (INFO to stderr, so it never
pollutes the stdout result lines) and suppressed by ``--quiet``.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable

LOGGER_NAME = "paperscale"


class MetricsKeeper:
    """Cumulative counters plus per-second rates over a trailing time window.

    ``add(**deltas)`` records named increments (e.g. ``pages_completed=1``,
    ``output_tokens=512``). ``window_rate``/``cumulative_rate`` report per-second
    throughput over the trailing window and over the whole run, the way olmOCR's
    MetricsKeeper does.
    """

    def __init__(self, *, window_seconds: float = 300.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_seconds = window_seconds
        self._clock = clock
        self.start = clock()
        self.total: dict[str, float] = {}
        self._events: list[tuple[float, dict[str, float]]] = []
        self._lock = threading.Lock()

    def add(self, **deltas: float) -> None:
        now = self._clock()
        with self._lock:
            for key, value in deltas.items():
                self.total[key] = self.total.get(key, 0.0) + value
            self._events.append((now, dict(deltas)))
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.pop(0)

    def elapsed(self, now: float | None = None) -> float:
        return (self._clock() if now is None else now) - self.start

    def window_rate(self, key: str, now: float | None = None) -> float:
        now = self._clock() if now is None else now
        with self._lock:
            self._trim(now)
            windowed = sum(deltas.get(key, 0.0) for _, deltas in self._events)
        span = min(self.elapsed(now), self.window_seconds)
        return windowed / span if span > 0 else 0.0

    def cumulative_rate(self, key: str, now: float | None = None) -> float:
        elapsed = self.elapsed(now)
        return (self.total.get(key, 0.0) / elapsed) if elapsed > 0 else 0.0

    def get(self, key: str) -> float:
        return self.total.get(key, 0.0)


def configure_logging(*, quiet: bool = False, level: int | None = None) -> logging.Logger:
    """Configure the ``paperscale`` logger. Idempotent; safe to call once per process.

    Default level is INFO (rich progress on); ``quiet=True`` raises it to WARNING so
    the periodic progress/summary lines are suppressed but warnings/errors still show.
    Output goes to stderr to keep stdout (status lines / JSON) clean.
    """
    logger = logging.getLogger(LOGGER_NAME)
    resolved = level if level is not None else (logging.WARNING if quiet else logging.INFO)
    logger.setLevel(resolved)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(_make_handler(resolved))
    return logger


def _make_handler(level: int) -> logging.Handler:
    try:  # rich console formatting when available
        from rich.console import Console
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            show_path=False,
            rich_tracebacks=False,
            markup=False,
            log_time_format="%H:%M:%S",
        )
    except Exception:  # noqa: BLE001 - plain stderr fallback if rich is unavailable
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    handler.setLevel(level)
    return handler


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
