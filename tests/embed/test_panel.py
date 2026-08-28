"""Tests for the embed stat panel and the sustained-queue advisory.

The render tests drive a real `RichReporter` rather than inspecting the values
dict: two of the panel's decisions -- three rows instead of four, and `model` in
the header instead of a row -- are about what survives a crushed pane, and only
a rendered frame can show that.
"""

from __future__ import annotations

import io
import re
import unittest

from paperscale.embed.panel import ADVISORY_WINDOW_S, EMBED_ROWS, QUEUE_ADVISORY, push_embed_stats, queue_advisory
from paperscale.tui import MIN_STAT_ROWS, RenderStyle, RichReporter, _layout_budget
from paperscale.vllm_stats import Snapshot, VLLMStats

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# ASCII box drawing, so every assertion in this file can be written in ASCII and
# still name the characters it is matching against.
_ASCII_STYLE = RenderStyle(ascii_only=True, spinner="line", use_screen=True, refresh_per_second=4)


class _Rep:
    """Reporter stand-in: keeps the last value pushed per (group, row)."""

    def __init__(self) -> None:
        self.stats: dict[tuple[str, str], object] = {}

    def set_stat(self, name: str, value, *, group: str = "run") -> None:
        self.stats[(group, name)] = value

    def server(self) -> dict[str, object]:
        return {name: value for (group, name), value in self.stats.items() if group == "server"}


class _Poller:
    def __init__(self, available: bool) -> None:
        self.available = available


class _StepClock:
    """Deterministic monotonic clock for VLLMStats. Advance with `tick`."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _live_stats(waiting: float | None = 3.0) -> VLLMStats:
    """A `VLLMStats` whose windowed rates are the size a real vLLM produces.

    Built from Snapshots and read back through the production code, so a number
    here cannot drift from what `Rates` computes. Over the last 10 s: 6120
    generation tok/s and 6210 prompt tok/s -- deliberately close, so a panel
    reading the wrong one still looks plausible and only an exact assertion
    catches it.
    """
    clock = _StepClock()
    stats = VLLMStats(clock=clock)
    stats.add(Snapshot(generation_tokens=0.0, prompt_tokens=0.0, running=0.0, waiting=0.0))
    clock.tick(100.0)
    stats.add(Snapshot(generation_tokens=587_800.0, prompt_tokens=581_400.0, running=12.0, waiting=waiting))
    clock.tick(10.0)
    stats.add(Snapshot(generation_tokens=649_000.0, prompt_tokens=643_500.0, running=12.0, waiting=waiting))
    return stats


def _queue(*waiting: float | None, step: float = 5.0) -> list[tuple[float, float | None]]:
    """`(timestamp, waiting)` pairs one 5 s stats tick apart, oldest first."""
    return [(1000.0 + i * step, value) for i, value in enumerate(waiting)]


# 5 s ticks, so 13 samples span exactly ADVISORY_WINDOW_S and 12 fall one tick short.
_TICKS_TO_WINDOW = int(ADVISORY_WINDOW_S / 5.0) + 1


class PushEmbedStatsTest(unittest.TestCase):
    def _push(self, rep, stats, poller, *, outstanding: int = 16) -> None:
        push_embed_stats(rep, stats, poller, stored_dim=768, native_dim=4096, outstanding=outstanding)

    def test_every_row_is_written_in_both_branches(self):
        # The fixed-row-set discipline: `set_stat` can add and overwrite but never
        # remove, so a branch writing a subset leaves the other branch's values on
        # screen. Both branches are checked against the same tuple.
        for stats, poller in ((_live_stats(), _Poller(True)), (_live_stats(), _Poller(False)), (None, None)):
            with self.subTest(poller=poller):
                rep = _Rep()
                self._push(rep, stats, poller)
                self.assertEqual(set(rep.server()), set(EMBED_ROWS))

    def test_the_unavailable_branch_leaves_no_stale_numbers(self):
        # The failure this inherits from `_VLLM_ROWS`: live numbers that stay on
        # screen after the scraper dies, reading as though they were current.
        rep = _Rep()
        self._push(rep, _live_stats(), _Poller(True))
        self.assertEqual(rep.server()["tok/s"], "6.2k")
        self._push(rep, _live_stats(), _Poller(False))
        self.assertEqual(rep.server()["tok/s"], "-")
        self.assertEqual(rep.server()["in-flight"], "16/-")
        # dim is the flag and the Adapter, not the scraper -- it stays true.
        self.assertEqual(rep.server()["dim"], "768/4096")

    def test_the_panel_reads_prompt_tps_never_gen_tps(self):
        # Embeddings are prefill-only, so `vllm:generation_tokens_total` never
        # moves on a real Invocation and reading it would show an idle server
        # rather than a bug. The fixture keeps both counters moving so the wrong
        # field renders a plausible number instead of an obvious zero.
        rep = _Rep()
        self._push(rep, _live_stats(), _Poller(True))
        self.assertEqual(rep.server()["tok/s"], "6.2k")
        self.assertNotEqual(rep.server()["tok/s"], "6.1k")

    def test_dim_is_stored_over_native(self):
        rep = _Rep()
        push_embed_stats(rep, None, None, stored_dim=768, native_dim=4096, outstanding=0)
        self.assertEqual(rep.server()["dim"], "768/4096")

    def test_in_flight_is_the_client_count_over_the_server_queue(self):
        rep = _Rep()
        self._push(rep, _live_stats(waiting=3.0), _Poller(True))
        self.assertEqual(rep.server()["in-flight"], "16/3")

    def test_an_absent_queue_renders_a_dash_not_a_zero(self):
        # `format_rate`'s rule: absence is not a measurement. The OCR panel's
        # `running` row prints `rates.waiting or 0` and so reports a dead scraper
        # as an empty queue; this one must not.
        rep = _Rep()
        self._push(rep, _live_stats(waiting=None), _Poller(True))
        self.assertEqual(rep.server()["in-flight"], "16/-")

    def test_an_empty_queue_is_a_measurement(self):
        rep = _Rep()
        self._push(rep, _live_stats(waiting=0.0), _Poller(True))
        self.assertEqual(rep.server()["in-flight"], "16/0")

    def test_a_deep_queue_is_not_abbreviated_like_a_token_rate(self):
        # `format_rate` would render this as `1.2k`, which is right for tokens
        # per second and wrong for a count of queued requests.
        rep = _Rep()
        self._push(rep, _live_stats(waiting=1200.0), _Poller(True))
        self.assertEqual(rep.server()["in-flight"], "16/1200")

    def test_no_server_statistics_still_reports_what_the_client_knows(self):
        # Unlike `push_vllm_stats`, this does not return early: two of the three
        # rows are sourced from this process and stay true with no scraper at all.
        rep = _Rep()
        self._push(rep, None, None, outstanding=7)
        self.assertEqual(rep.server(), {"dim": "768/4096", "tok/s": "-", "in-flight": "7/-"})

    def test_the_rows_land_in_the_server_group(self):
        rep = _Rep()
        self._push(rep, _live_stats(), _Poller(True))
        self.assertEqual({group for group, _ in rep.stats}, {"server"})


class QueueAdvisoryTest(unittest.TestCase):
    def test_a_short_history_is_silent(self):
        self.assertIsNone(queue_advisory([]))
        self.assertIsNone(queue_advisory(_queue(3.0)))

    def test_silent_before_the_window_is_spanned(self):
        history = _queue(*[3.0] * (_TICKS_TO_WINDOW - 1))
        self.assertLess(history[-1][0] - history[0][0], ADVISORY_WINDOW_S)
        self.assertIsNone(queue_advisory(history))

    def test_fires_once_the_window_is_sustained_and_names_the_flag(self):
        message = queue_advisory(_queue(*[3.0] * _TICKS_TO_WINDOW))
        self.assertEqual(message, QUEUE_ADVISORY)
        self.assertIn("--concurrency", str(message))

    def test_it_does_not_repeat_on_every_later_tick(self):
        # Edge-triggered on purpose: the event pane holds eight rows, and one
        # sentence repeated every five seconds would push everything else out.
        for extra in range(1, 6):
            with self.subTest(extra=extra):
                self.assertIsNone(queue_advisory(_queue(*[3.0] * (_TICKS_TO_WINDOW + extra))))

    def test_a_drained_queue_restarts_the_window(self):
        # The history spans well over a minute, but the queue emptied inside it,
        # so the run is five ticks old and says nothing.
        history = _queue(*([3.0] * _TICKS_TO_WINDOW + [0.0] + [3.0] * 5))
        self.assertGreater(history[-1][0] - history[0][0], ADVISORY_WINDOW_S)
        self.assertIsNone(queue_advisory(history))

    def test_a_second_episode_fires_again(self):
        history = _queue(*([3.0] * _TICKS_TO_WINDOW + [0.0] + [3.0] * _TICKS_TO_WINDOW))
        self.assertEqual(queue_advisory(history), QUEUE_ADVISORY)

    def test_a_failed_scrape_breaks_the_run(self):
        # A missing measurement is not evidence of a queue; treating None as
        # "still busy" would turn a dead poller into an accusation against a flag.
        samples: list[float | None] = [3.0] * _TICKS_TO_WINDOW
        samples[6] = None
        self.assertIsNone(queue_advisory(_queue(*samples)))

    def test_it_changes_nothing(self):
        # Advisory only. It touches neither its input nor the panel beside it --
        # the whole reason the record kept a fixed --concurrency instead of a
        # control loop is that nothing here is allowed to move on its own.
        history = _queue(*[3.0] * _TICKS_TO_WINDOW)
        before = list(history)
        rep = _Rep()
        push_embed_stats(rep, _live_stats(), _Poller(True), stored_dim=768, native_dim=4096, outstanding=16)
        panel = dict(rep.server())
        self.assertEqual(queue_advisory(history), queue_advisory(history))
        self.assertEqual(history, before)
        self.assertEqual(rep.server(), panel)


def _render(rep, console) -> list[str]:
    """One frame as visible text.

    `force_terminal=True` means capture() returns styled output, so the escapes
    have to go before any column index is taken -- otherwise a position counts
    escape bytes rather than cells.
    """
    with console.capture() as cap:
        console.print(rep.__rich__())
    return _ANSI.sub("", cap.get()).rstrip("\n").split("\n")


class _Buffer(io.StringIO):
    @property
    def encoding(self) -> str:
        return "utf-8"


def _reporter(width: int, height: int):
    from rich.console import Console

    console = Console(file=_Buffer(), force_terminal=True, width=width, height=height)
    return RichReporter("paperscale . Qwen/Qwen3-Embedding-8B", console=console, style=_ASCII_STYLE), console


def _populate(rep, *, log_lines: int = 20) -> None:
    """The panel an embed Invocation actually draws, pushed by its own writers."""
    for name, value in (("documents", "4,309"), ("chunks", "61,204"), ("skipped", "3,902"), ("empty", "12")):
        rep.set_stat(name, value)
    push_embed_stats(rep, _live_stats(), _Poller(True), stored_dim=768, native_dim=4096, outstanding=16)
    for name, value in (("failed", "0"), ("retrying", "2"), ("oversize", "0")):
        rep.set_stat(name, value, group="issues")
    rep.phase("embedding", total=8578)
    for i in range(log_lines):
        rep.log(f"event line {i}")


class PanelRenderTest(unittest.TestCase):
    def test_the_server_group_renders_ahead_of_issues(self):
        # `_stat_columns` orders on ("run", "server", "issues"), which is why the
        # `vllm` -> `server` rename discharged the ordering prerequisite: named
        # `vllm` the group sorted last and put `issues` above the throughput.
        rep, console = _reporter(120, 30)
        _populate(rep)
        titles = next(line for line in _render(rep, console) if "- run " in line)
        self.assertLess(titles.index("run"), titles.index("server"))
        self.assertLess(titles.index("server"), titles.index("issues"))

    def test_the_server_panel_is_legible_at_eighty_columns(self):
        # 80 columns is the default pane and the width the row set was sized for:
        # three panels of 22 cells, `in-flight` (9) plus 2 of padding taking the
        # key column, 11 left for the value. A truncated label crops rather than
        # ellipsises under `ascii_only`, so a short match would hide it.
        rep, console = _reporter(80, 24)
        _populate(rep)
        frame = "\n".join(_render(rep, console))
        for row in EMBED_ROWS:
            self.assertIn(row, frame)
        self.assertIn("768/4096", frame)
        self.assertIn("16/3", frame)

    def test_three_rows_survive_the_height_cliff(self):
        # `_layout_budget` grows sections in the order bars, events, stats, so
        # stats reach a fourth row only once events have grown; height 17 hands
        # out three stat rows and height 18 four. `_stat_columns` truncates from
        # the tail, so a four-row `server` group would lose `in-flight` -- the
        # saturation signal -- on any pane shorter than 18 rows.
        self.assertEqual(_layout_budget(17, 4, 1).stat_rows, MIN_STAT_ROWS)
        self.assertEqual(_layout_budget(18, 4, 1).stat_rows, 4)
        self.assertLessEqual(len(EMBED_ROWS), MIN_STAT_ROWS)

        rep, console = _reporter(80, 17)
        _populate(rep)
        frame = "\n".join(_render(rep, console))
        for row in EMBED_ROWS:
            self.assertIn(row, frame)
        # The four-row `run` group loses its tail at this height, which is what
        # `server` would have done with a fourth row.
        self.assertNotIn("empty", frame)

    def test_the_model_id_rides_in_the_header(self):
        # It cannot be a row: the two pinned ids are 23 and 31 characters against
        # a value column of 11 at 80 columns, so both would read `Qwen/Qwen...`
        # for the whole Invocation.
        rep, console = _reporter(80, 24)
        _populate(rep)
        lines = _render(rep, console)
        self.assertIn("Qwen/Qwen3-Embedding-8B", lines[0])
        self.assertNotIn("model", "\n".join(lines[1:]))
