"""Progress reporting abstraction, decoupled from any specific renderer.

Two implementations back the same tiny interface:

- ``NullReporter`` — the default. Emits phase headers and ``log`` lines to
  stderr as plain text (matching pre-TUI behaviour) and does nothing fancy.
- ``RichReporter`` — an immich-go-style live dashboard (header + stats table +
  per-phase progress bars + a scrolling log tail), rendered with ``rich.Live``.

Use ``make_reporter`` to pick one. Both are context managers and expose the
same ``phase`` / ``log`` / ``set_stat`` surface, so callers (evaluate today, the
OCR pipeline later) stay renderer-agnostic. Reporters are pure instrumentation:
driving them must never change a caller's output.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable


@runtime_checkable
class Phase(Protocol):
    """A unit of work with an optional known total (for a progress bar)."""

    def advance(self, n: int = 1) -> None: ...
    def done(self) -> None: ...


class ProgressReporter(Protocol):
    def __enter__(self) -> "ProgressReporter": ...
    def __exit__(self, *exc) -> None: ...
    def phase(self, name: str, total: int | None = None) -> Phase: ...
    def log(self, message: str) -> None: ...
    def set_stat(self, name: str, value) -> None: ...


# --------------------------------------------------------------------------- #
# Null implementation (default / non-TTY)
# --------------------------------------------------------------------------- #
class _NullPhase:
    def advance(self, n: int = 1) -> None:
        pass

    def done(self) -> None:
        pass


class NullReporter:
    """No live UI. Prints phase starts and log lines to stderr, like before."""

    def __enter__(self) -> "NullReporter":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def phase(self, name: str, total: int | None = None) -> Phase:
        print(f"[{name}]", file=sys.stderr)
        return _NullPhase()

    def log(self, message: str) -> None:
        print(message, file=sys.stderr)

    def set_stat(self, name: str, value) -> None:
        pass


# --------------------------------------------------------------------------- #
# Rich implementation (immich-go style dashboard)
# --------------------------------------------------------------------------- #
class _RichPhase:
    def __init__(self, reporter: "RichReporter", task_id: int, total: int | None) -> None:
        self._reporter = reporter
        self._task_id = task_id
        self._total = total

    def advance(self, n: int = 1) -> None:
        self._reporter._progress.advance(self._task_id, n)
        self._reporter._refresh()

    def done(self) -> None:
        if self._total is None:
            # mark an indeterminate task complete so its bar fills
            self._reporter._progress.update(self._task_id, total=1, completed=1)
        else:
            self._reporter._progress.update(self._task_id, completed=self._total)
        self._reporter._refresh()


class RichReporter:
    """Live dashboard: header + stats table + phase bars + log tail."""

    def __init__(self, title: str, *, console=None, max_log: int = 8) -> None:
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

        self._title = title
        self._console = console or Console(stderr=True)
        self._max_log = max_log
        self._stats: dict[str, object] = {}
        self._log: list[str] = []
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self._console,
        )
        self._live = None

    # -- context management -------------------------------------------------- #
    def __enter__(self) -> "RichReporter":
        from rich.live import Live

        self._live = Live(self._render(), console=self._console, refresh_per_second=12, transient=False)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._refresh()
            self._live.__exit__(*exc)
            self._live = None

    # -- reporter surface ---------------------------------------------------- #
    def phase(self, name: str, total: int | None = None) -> Phase:
        task_id = self._progress.add_task(name, total=total)
        self._refresh()
        return _RichPhase(self, task_id, total)

    def log(self, message: str) -> None:
        self._log.append(message)
        if len(self._log) > self._max_log:
            self._log = self._log[-self._max_log :]
        self._refresh()

    def set_stat(self, name: str, value) -> None:
        self._stats[name] = value
        self._refresh()

    # -- rendering ----------------------------------------------------------- #
    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self):
        from rich.panel import Panel
        from rich.table import Table

        stats = Table.grid(padding=(0, 2))
        stats.add_column(style="cyan", justify="right")
        stats.add_column(style="bold white")
        for k, v in self._stats.items():
            stats.add_row(str(k), str(v))

        log_body = "\n".join(self._log) if self._log else "…"

        outer = Table.grid(expand=True)
        outer.add_row(Panel(stats, title=self._title, title_align="left"))
        outer.add_row(self._progress)
        outer.add_row(Panel(log_body, title="events", title_align="left", height=self._max_log + 2))
        return outer


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_reporter(tui: bool, *, title: str, stream=None) -> ProgressReporter:
    """Pick a reporter. RichReporter only when --tui is on AND output is a TTY.

    Raises a clear error if --tui is requested but ``rich`` isn't installed.
    """
    stream = stream if stream is not None else sys.stderr
    if not tui:
        return NullReporter()
    if not getattr(stream, "isatty", lambda: False)():
        return NullReporter()  # piped/redirected — keep output clean
    try:
        import rich  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit("--tui requires the 'tui' extra: poetry install --extras tui") from exc
    return RichReporter(title)
