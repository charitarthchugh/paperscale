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
from dataclasses import dataclass
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


# --------------------------------------------------------------------------- #
# Height budget
# --------------------------------------------------------------------------- #
# The original renderer never measured the terminal: _render() referenced
# console.size nowhere, so frame height tracked run state instead of the pane it
# was drawn into, and Rich fell back to re-emitting whole frames that scrolled.
# Everything below exists to make height a measured input.
HEADER_ROWS = 1
PANEL_CHROME = 2  # top and bottom border of a bordered panel
MIN_STAT_ROWS, MIN_BAR_ROWS, MIN_EVENT_ROWS = 3, 1, 2
MAX_EVENT_ROWS = 8


@dataclass(frozen=True)
class Budget:
    stat_rows: int
    bar_rows: int
    event_rows: int

    def total_rows(self) -> int:
        return HEADER_ROWS + _cost(self.stat_rows, self.bar_rows, self.event_rows)


def _cost(stat_rows: int, bar_rows: int, event_rows: int) -> int:
    stats = stat_rows + PANEL_CHROME if stat_rows else 0
    events = event_rows + PANEL_CHROME if event_rows else 0
    return stats + bar_rows + events


def _layout_budget(height: int, n_stats: int, n_phases: int) -> Budget:
    """Split `height` rows so the frame is exactly the height of the terminal.

    Recomputed on every render, never cached: tmux panes change size on split,
    zoom, and detach/reattach, and a cached budget would reintroduce the original
    overflow the first time a pane was zoomed.

    Starvation order: events go first, then stats. Bars never go -- a dashboard
    with no progress indicator tells you nothing.
    """
    available = max(int(height), 1) - HEADER_ROWS
    if available < MIN_BAR_ROWS:
        return Budget(0, max(available, 0), 0)

    stat_rows, bar_rows, event_rows = MIN_STAT_ROWS, MIN_BAR_ROWS, MIN_EVENT_ROWS
    if _cost(stat_rows, bar_rows, event_rows) > available:
        event_rows = 0
    if _cost(stat_rows, bar_rows, event_rows) > available:
        stat_rows = 0
    if _cost(stat_rows, bar_rows, event_rows) > available:
        return Budget(0, available, 0)

    surplus = available - _cost(stat_rows, bar_rows, event_rows)
    grown = min(surplus, max(n_phases - bar_rows, 0))
    bar_rows += grown
    surplus -= grown
    if event_rows:
        grown = min(surplus, MAX_EVENT_ROWS - event_rows)
        event_rows += grown
        surplus -= grown
    if stat_rows:
        grown = min(surplus, max(n_stats - stat_rows, 0))
        stat_rows += grown
        surplus -= grown

    # Whatever is left pads a section so the frame fills the pane exactly.
    if event_rows:
        event_rows += surplus
    elif stat_rows:
        stat_rows += surplus
    else:
        bar_rows += surplus
    return Budget(stat_rows, bar_rows, event_rows)


# --------------------------------------------------------------------------- #
# Terminal capability
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RenderStyle:
    """What this terminal can be trusted to draw.

    Geometry is measured per render (see _layout_budget); capability is resolved
    once, because it does not change mid-run.
    """

    ascii_only: bool
    spinner: str
    use_screen: bool
    refresh_per_second: int

    @property
    def box(self):
        from rich import box as rich_box

        return rich_box.ASCII if self.ascii_only else rich_box.ROUNDED


def terminal_profile(encoding: str, env: dict) -> RenderStyle:
    """Resolve render style from terminal capabilities.

    Glyph coverage cannot be detected, and tmux makes that structurally worse:
    TERM describes tmux, not the outer terminal, so the outer terminal's font is
    invisible here and a UTF-8 locale says nothing about whether the rendering
    font has a braille block. Hence PAPERSCALE_TUI_ASCII.
    """
    term = (env.get("TERM") or "").lower()
    override = env.get("PAPERSCALE_TUI_ASCII")
    if override == "1":
        ascii_only = True
    elif override == "0":
        ascii_only = False
    else:
        ascii_only = "utf" not in (encoding or "").lower()

    # The Linux VT console draws boxes but its font has no braille block.
    braille_ok = not ascii_only and not term.startswith("linux")
    multiplexed = term.startswith("screen") or term.startswith("tmux")
    return RenderStyle(
        ascii_only=ascii_only,
        spinner="dots" if braille_ok else "line",
        use_screen=True,
        refresh_per_second=2 if multiplexed else 4,
    )
