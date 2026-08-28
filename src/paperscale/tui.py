"""Progress reporting abstraction, decoupled from any specific renderer.

Two implementations back the same tiny interface:

- ``NullReporter`` — the default. Emits phase headers and ``log`` lines to
  stderr as plain text (matching pre-TUI behaviour) and does nothing fancy.
- ``RichReporter`` — an immich-go-style live dashboard (header + grouped stat
  columns + per-phase progress bars + an event tail), rendered with
  ``rich.Live``. Its frame is sized to the terminal on every tick, so it never
  outgrows the pane and forces ``Live`` to scroll instead of overwrite.

Use ``make_reporter`` to pick one. Both are context managers and expose the
same ``phase`` / ``log`` / ``set_stat`` surface, so callers (evaluate today, the
OCR pipeline later) stay renderer-agnostic. Reporters are pure instrumentation:
driving them must never change a caller's output.

``install_tui_logging`` / ``restore_console_logging`` live here too: both
commands need the identical hand-off of stderr to the event pane, and this is
the one UI module they already share.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


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
    def set_stat(self, name: str, value, *, group: str = "run") -> None: ...


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

    def set_stat(self, name: str, value, *, group: str = "run") -> None:
        pass


# --------------------------------------------------------------------------- #
# Rich implementation (immich-go style dashboard)
# --------------------------------------------------------------------------- #
def _one_line(text) -> str:
    """Flatten a caller-supplied string to one display row.

    ``no_wrap`` stops rich wrapping, but an embedded newline still splits a cell
    into two rows. The stat and event panels are height-clamped so they only lose
    content, but the bar region has no panel around it and would push the frame
    past the pane -- the exact failure this renderer exists to prevent.
    """
    return " ".join(str(text).splitlines())


def _elapsed(seconds: float) -> str:
    """`HH:MM:SS` for the header clock, hand-rolled rather than via ``strftime``.

    ``time.gmtime`` wraps at 24 hours, which is inside the range this reports on:
    an overnight OCR run would restart the clock at 00:00:00 and read as though it
    had just begun. Hours simply keep counting here.
    """
    whole = max(int(seconds), 0)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class _RichPhase:
    def __init__(self, reporter: "RichReporter", task_id: int, total: int | None) -> None:
        self._reporter = reporter
        self._task_id = task_id
        self._total = total

    def advance(self, n: int = 1) -> None:
        # No refresh here. The old code re-rendered on every page, which on a
        # multi-thousand-page run meant thousands of forced frames fighting Live.
        self._reporter._progress.advance(self._task_id, n)

    def done(self) -> None:
        if self._total is None:
            # mark an indeterminate task complete so its bar fills
            self._reporter._progress.update(self._task_id, total=1, completed=1)
        else:
            self._reporter._progress.update(self._task_id, completed=self._total)


class RichReporter:
    """Fixed-height live dashboard: header, stat columns, phase bars, event tail."""

    def __init__(self, title: str, *, console=None, style: "RenderStyle | None" = None) -> None:
        from rich.console import Console
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        from rich.table import Column

        self._title = title
        self._console = console or Console(stderr=True)
        encoding = getattr(self._console.file, "encoding", "utf-8")
        self._style = style or terminal_profile(encoding, dict(os.environ))
        # Rich truncates with U+2026 regardless of console.options.ascii_only, so
        # the fallback has to be chosen here rather than left to rich.
        self._overflow = "crop" if self._style.ascii_only else "ellipsis"
        self._stats: dict[str, dict[str, object]] = {}
        self._log: list[str] = []
        # Wall-clock start for the header's elapsed counter. Monotonic, so a
        # daylight-saving jump or an ntp step cannot make the run look shorter.
        self._start = time.monotonic()
        self._progress = Progress(
            SpinnerColumn(self._style.spinner),
            # TextColumn normally supplies its own Column(no_wrap=True); handing it a
            # table_column to set `overflow` replaces that default wholesale, so
            # no_wrap has to be restated. Omitting it wraps two long phase names into
            # nine rows at width 40 and blows the budget -- and rich wraps rather than
            # overflows, so only a row count catches it, never a width assertion.
            TextColumn("[bold]{task.description}", table_column=Column(no_wrap=True, overflow=self._overflow)),
            # Every column needs the overflow too, not just the description: rich's
            # default is `ellipsis` and it truncates with U+2026 whatever the console
            # encoding says, so on a genuinely ascii-encoded stderr a narrow pane
            # raises UnicodeEncodeError straight out of console.print and into the
            # caller. These three take no table_column of their own -- unlike
            # TextColumn -- so `Column(overflow=...)` drops no default.
            BarColumn(table_column=Column(overflow=self._overflow)),
            MofNCompleteColumn(table_column=Column(overflow=self._overflow)),
            TimeElapsedColumn(table_column=Column(overflow=self._overflow)),
            console=self._console,
        )
        self._live = None

    # -- context management -------------------------------------------------- #
    def __enter__(self) -> "RichReporter":
        from rich.live import Live

        # Passing `self` (not a snapshot) means __rich__ runs on every tick, so the
        # budget is recomputed per frame -- required for tmux pane resizes.
        self._live = Live(
            self,
            console=self._console,
            screen=self._style.use_screen,
            auto_refresh=True,
            refresh_per_second=self._style.refresh_per_second,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # -- reporter surface ---------------------------------------------------- #
    def phase(self, name: str, total: int | None = None) -> Phase:
        return _RichPhase(self, self._progress.add_task(_one_line(name), total=total), total)

    def log(self, message: str) -> None:
        # Retention is deliberately larger than MAX_EVENT_ROWS: that constant is a
        # growth target for the budget, not a cap, so a tall pane asks for far more
        # rows than 8 and would otherwise get blank ones.
        self._log.append(_one_line(message))
        if len(self._log) > MAX_LOG_HISTORY:
            self._log = self._log[-MAX_LOG_HISTORY:]

    def set_stat(self, name: str, value, *, group: str = "run") -> None:
        self._stats.setdefault(group, {})[name] = value

    # -- rendering ----------------------------------------------------------- #
    def __rich__(self):
        return self._render()

    def _render(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        width, height = self._console.size.width, self._console.size.height
        n_stats = max((len(v) for v in self._stats.values()), default=0)
        # The log length is an input to the budget, not just to the slice below:
        # without it the event pane grows into the surplus and pads what it does
        # not have. See `_layout_budget`'s `n_events`.
        budget = _layout_budget(height, n_stats, len(self._progress.tasks), len(self._log))

        rows = [self._header()]

        if budget.stat_rows and self._stats:
            rows.append(self._stat_columns(budget.stat_rows, width))
        if budget.bar_rows:
            rows.append(self._bars(budget.bar_rows))
        if budget.event_rows:
            body = Table.grid()
            body.add_column(no_wrap=True, overflow=self._overflow)
            for line in self._log[-budget.event_rows :]:
                body.add_row(line)
            rows.append(Panel(body, title="events", title_align="left", box=self._style.box, height=budget.event_rows + PANEL_CHROME))
        return Group(*rows)

    def _header(self):
        """Title on the left, elapsed time on the right, in exactly HEADER_ROWS rows.

        A grid rather than one padded ``Text``: the clock has to stay pinned to the
        last column of the pane as it resizes, and `justify="right"` on an expanding
        column does that without measuring anything. Both columns are ``no_wrap``
        with an explicit overflow, so a title longer than the pane truncates instead
        of wrapping the clock onto a second row -- the header budget is one row and
        the whole frame is sized around it.
        """
        from rich.table import Table
        from rich.text import Text

        grid = Table.grid(expand=True)
        grid.add_column(no_wrap=True, overflow=self._overflow)
        grid.add_column(no_wrap=True, overflow=self._overflow, justify="right")
        clock = f"elapsed {_elapsed(time.monotonic() - self._start)}"
        grid.add_row(Text(self._title, style="bold"), Text(clock, style="dim"))
        return grid

    def _stat_columns(self, stat_rows: int, width: int):
        """Render each stat group as its own column.

        Columns are cheap and rows are scarce, which is the whole reason the
        layout is horizontal. Below 60 columns there is no width left to divide,
        so only the first group is drawn.

        The row is a grid rather than ``rich.columns.Columns``: Columns re-flows
        panels onto extra rows once their combined minimum width exceeds the pane
        (15 rows instead of 5 at width 60 with realistic vLLM values), which is
        precisely the overflow this renderer exists to prevent. A grid keeps one
        row and crushes the columns instead.
        """
        from rich.panel import Panel
        from rich.table import Table

        order = [g for g in ("run", "server", "issues") if g in self._stats]
        order += [g for g in self._stats if g not in order]

        panels = []
        for group in order:
            grid = Table.grid(padding=(0, 2))
            # Both columns, not just the value: rich's per-column default is
            # `ellipsis`, which truncates with U+2026 whatever the console encoding
            # says. A key column left on the default raised UnicodeEncodeError out
            # of console.print on an ascii-encoded stderr at 80x24 -- the pipeline's
            # own default stat rows crush `status` to `st...` there. Every
            # `add_column`/`Column(...)` in this module carries `overflow` for that
            # reason; treat one without it as a bug.
            grid.add_column(style="cyan", justify="right", no_wrap=True, overflow=self._overflow)
            grid.add_column(style="bold white", no_wrap=True, overflow=self._overflow)
            for key, value in list(self._stats[group].items())[:stat_rows]:
                grid.add_row(_one_line(key), _one_line(value))
            panels.append(Panel(grid, title=group, title_align="left", box=self._style.box, height=stat_rows + PANEL_CHROME))

        if width < 60:
            panels = panels[:1]
        row = Table.grid(expand=True)
        for _ in panels:
            row.add_column(ratio=1, overflow=self._overflow)
        row.add_row(*panels)
        return row

    def _bars(self, bar_rows: int):
        """Show the most recent `bar_rows` phases, noting anything hidden."""
        from rich.console import Group
        from rich.text import Text

        tasks = self._progress.tasks
        if len(tasks) <= bar_rows:
            return self._progress.make_tasks_table(tasks)
        shown = tasks[-(bar_rows - 1) :] if bar_rows > 1 else []
        more = Text(f" +{len(tasks) - len(shown)} more", style="dim", no_wrap=True, overflow=self._overflow)
        if not shown:
            return more
        return Group(self._progress.make_tasks_table(shown), more)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_reporter(tui: bool, *, title: str, stream=None) -> ProgressReporter:
    """Pick a reporter. RichReporter only when --tui is on AND output is a usable TTY.

    Raises a clear error if --tui is requested but ``rich`` isn't installed.
    """
    stream = stream if stream is not None else sys.stderr
    if not tui:
        return NullReporter()
    if not getattr(stream, "isatty", lambda: False)():
        return NullReporter()  # piped/redirected — keep output clean
    if (os.environ.get("TERM") or "").lower() == "dumb":
        return NullReporter()  # no cursor control to drive a live frame
    try:
        import rich  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise SystemExit("--tui requires the 'tui' extra: poetry install --extras tui") from exc
    return RichReporter(title)


# --------------------------------------------------------------------------- #
# Logging hand-off
# --------------------------------------------------------------------------- #
class ReporterLogHandler(logging.Handler):
    """Forward warnings and errors into the dashboard's event pane."""

    def __init__(self, reporter) -> None:
        super().__init__(level=logging.WARNING)
        self._reporter = reporter
        # The loggers this handler was attached to, and the handlers it displaced
        # off them -- restored verbatim by `restore_console_logging`. They live
        # here because the handler is the one object install and restore share.
        self.displaced: list[tuple[logging.Logger, logging.Handler]] = []
        self.targets: tuple[logging.Logger, ...] = ()

    def emit(self, record: logging.LogRecord) -> None:
        # Logger.callHandlers already screens on self.level, but emit() is the
        # public entry point a handler is judged on and it does no filtering of
        # its own. The event pane holds a handful of rows; one INFO line that
        # slipped in that way costs a warning the user needed to see.
        if record.levelno < self.level:
            return
        try:
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
            self._reporter.log(f"{stamp}  {record.levelname:<5} {record.getMessage()}")
        except Exception:  # pragma: no cover - a logging handler must never raise
            pass


def install_tui_logging(
    reporter,
    log_path: str | None,
    loggers: Sequence[logging.Logger] = (),
    console_handler: logging.Handler | None = None,
) -> ReporterLogHandler:
    """Take stderr away from the loggers and give the pane the important lines.

    Nothing may write to stderr underneath a live frame. Everything still lands in
    the log file, because the alternate screen has no scrollback to recover from.

    `loggers` are the caller's own loggers carrying `console_handler` -- the
    pipeline hands over its module logger and the `vllm` server logger; evaluate
    owns none and passes nothing. The root logger is always taken over on top of
    those, and that is not belt-and-braces. work_queue, check, front_matter,
    filter and vllm_stats each own a handler-less module logger that propagates to
    root, and `paperscale.filter` calls `logging.basicConfig()` at import, which
    puts a stderr StreamHandler there. Left alone it prints straight through the
    frame. (vllm_stats is the sharpest case, and the one that reaches evaluate
    too: its "statistics unavailable" warning would corrupt the very panel it
    feeds -- and with nothing on root at all, `logging.lastResort` prints it to
    stderr in any handler's place.) Root also has to keep at least one handler
    afterwards, for that same lastResort reason.
    """
    # Every fallible step runs first, before a single handler moves. Creating the
    # directory and opening the file can both raise, and this is called *before*
    # the caller enters the try that would call `restore_console_logging` -- so a
    # raise after the displacement loop would leave every logger with no handlers
    # at all and nothing left to put them back. The feature built to protect
    # stderr logging would be the thing that destroyed it.
    file_handler = open_log_file(log_path) if log_path is not None else None

    root = logging.getLogger()
    handler = ReporterLogHandler(reporter)
    # A caller that passes loggers but no console handler has nothing to displace
    # off them; `(logger, None)` in `displaced` would put a None back on restore.
    displaced = [(log, console_handler) for log in loggers if console_handler is not None]
    displaced += [(root, h) for h in list(root.handlers)]
    for target, existing in displaced:
        target.removeHandler(existing)
        handler.displaced.append((target, existing))

    # Callers set propagate=False on the loggers they hand over (the pipeline's
    # two do), so handlers on root cannot double-log their records.
    handler.targets = (*loggers, root)
    for attachment in (file_handler, handler):
        if attachment is not None:
            for target in handler.targets:
                target.addHandler(attachment)
    return handler


def open_log_file(log_path: str) -> logging.FileHandler:
    """Create the log directory and open the file. Fallible on purpose, and isolated.

    `--disk_logging` takes a bare filename as its `const`, so `dirname` is `""` for
    the common `--disk_logging` (no value) form and `os.makedirs("")` raises
    FileNotFoundError. `or "."` resolves that to the working directory, which is
    where a bare filename was always going to land.
    """
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    return file_handler


def restore_console_logging(tui_handler: ReporterLogHandler) -> None:
    """Undo install_tui_logging's rewiring. Must run even when the run crashed.

    The reporter handler comes off every logger it was added to, root included:
    past the `with rep` block the reporter is dead, and anything still routed into
    it is lost rather than printed. The file handler stays, matching how
    --disk_logging behaves for the life of the process.
    """
    for target in tui_handler.targets:
        target.removeHandler(tui_handler)
    for target, displaced in tui_handler.displaced:
        target.addHandler(displaced)


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
# `_layout_budget` treats MAX_EVENT_ROWS as a growth target, not a cap: events
# absorb the leftover surplus, so a 60-row pane hands the panel ~48 rows.
# Retention has to be decoupled from that target, or a tall pane would draw a
# few log lines padded out with blank rows. Retention is now also what bounds
# the panel outright, since `_layout_budget`'s `n_events` stops it at the lines
# actually held: this constant is the most rows any pane can ever draw.
MAX_LOG_HISTORY = 200


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


def _layout_budget(height: int, n_stats: int, n_phases: int, n_events: int | None = None) -> Budget:
    """Split `height` rows between the sections so the frame fits the terminal.

    Exactly fills it, save for the one case `n_events` describes below.

    Recomputed on every render, never cached: tmux panes change size on split,
    zoom, and detach/reattach, and a cached budget would reintroduce the original
    overflow the first time a pane was zoomed.

    Starvation order: events go first, then stats. Bars never go -- a dashboard
    with no progress indicator tells you nothing.

    `n_events` is how many log lines the caller actually holds. Given it, the
    event pane stops at that many rows and the frame is allowed to come up
    *short* of the pane. Without it events absorb the whole surplus, which on a
    60-row pane drew three log lines above forty blank ones -- invisible to the
    OCR pipeline, which is chatty enough to fill any pane, and the first thing an
    embedding Invocation shows, because it is silent until something fails. The
    parameter is optional so that a caller which genuinely does not know the log
    length keeps the original meaning rather than being made to lie about it.
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

    # An empty log keeps MIN_EVENT_ROWS rather than collapsing the panel: the
    # pane exists before the first line does, and letting it appear on that line
    # would shuffle the stats and bars above it exactly when something has just
    # gone wrong and the operator is reading them.
    event_cap = None if n_events is None else max(n_events, MIN_EVENT_ROWS)

    surplus = available - _cost(stat_rows, bar_rows, event_rows)
    grown = min(surplus, max(n_phases - bar_rows, 0))
    bar_rows += grown
    surplus -= grown
    if event_rows:
        target = MAX_EVENT_ROWS if event_cap is None else min(MAX_EVENT_ROWS, event_cap)
        grown = min(surplus, max(target - event_rows, 0))
        event_rows += grown
        surplus -= grown
    if stat_rows:
        grown = min(surplus, max(n_stats - stat_rows, 0))
        stat_rows += grown
        surplus -= grown

    # Whatever is left pads a section so the frame fills the pane exactly. With a
    # known log length there is nothing left to pad *with*: every section has
    # already grown to what it can fill (bars to `n_phases`, stats to `n_stats`,
    # events to the lines they hold), so the remainder is dropped and the frame
    # shrinks to what it has to say. Padding stats instead would move the blank
    # rows one panel to the left, not remove them.
    if event_cap is not None:
        # Never on a starved pane: `event_rows == 0` means the panel was dropped
        # whole, and handing it rows back would re-add its two border rows to a
        # budget that had no room for them.
        if event_rows:
            event_rows += min(surplus, max(event_cap - event_rows, 0))
    elif event_rows:
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
