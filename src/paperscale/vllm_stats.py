"""Scrape and summarise vLLM server statistics.

vLLM's ``/health`` endpoint returns a zero-byte 200 and carries no numbers; it is
a liveness probe only. Everything useful lives at ``/metrics`` in Prometheus text
format. vLLM V1 dropped the V0 ``vllm:avg_*_throughput_toks_per_s`` gauges and
exposes only cumulative counters, so every rate here is derived client-side by
differencing consecutive scrapes.

This module never raises at the caller. A statistics panel must not be able to
end a twelve-hour OCR run.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, fields

logger = logging.getLogger(__name__)

# name{label="value",...} 123.0   -- the label block is optional
_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[^\s]+)\s*$"
)
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>[^"]*)"')


@dataclass(frozen=True)
class Sample:
    labels: dict[str, str]
    value: float


def parse_metrics(text: str) -> dict[str, list[Sample]]:
    """Parse Prometheus text exposition into ``{metric_name: [Sample, ...]}``.

    Comments, blank lines, unparseable lines, and non-finite values are skipped.
    ``_created`` siblings keep their own metric names, so exact-name lookups of a
    counter never pick them up.
    """
    out: dict[str, list[Sample]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        labels = {m.group("key"): m.group("value") for m in _LABEL.finditer(match.group("labels") or "")}
        out.setdefault(match.group("name"), []).append(Sample(labels, value))
    return out


# Each logical metric resolves against an ordered candidate list, first name wins.
# This is what lets one scraper survive vLLM version drift.
_CANDIDATES: dict[str, tuple[str, ...]] = {
    "generation_tokens": ("vllm:generation_tokens_total",),
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "cache_hits": ("vllm:prefix_cache_hits_total", "vllm:prompt_tokens_cached_total"),
    "cache_queries": ("vllm:prefix_cache_queries_total", "vllm:prompt_tokens_total"),
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
# kv usage is already a fraction; summing it across engines would read as a full cache.
_AVERAGED = frozenset({"kv_usage"})


@dataclass(frozen=True)
class Snapshot:
    generation_tokens: float | None = None
    prompt_tokens: float | None = None
    cache_hits: float | None = None
    cache_queries: float | None = None
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None


def snapshot_from(parsed: dict[str, list[Sample]]) -> Snapshot:
    """Aggregate a parsed scrape into one Snapshot, summing across engines.

    With ``--data-parallel-size > 1`` there is one series per engine. Counters and
    request gauges add up; fractions are averaged. An unresolvable metric stays
    ``None`` -- zero is a measurement, absence is not.
    """
    values: dict[str, float | None] = {}
    for field, names in _CANDIDATES.items():
        samples = next((parsed[n] for n in names if n in parsed), None)
        if not samples:
            values[field] = None
        elif field in _AVERAGED:
            values[field] = sum(s.value for s in samples) / len(samples)
        else:
            values[field] = sum(s.value for s in samples)
    return Snapshot(**values)


def metrics_url(server: str) -> str:
    """Derive the /metrics URL from an OpenAI-compatible base URL."""
    base = server.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/metrics"


# Counters that must only ever increase. A decrease means the server restarted.
_MONOTONIC = ("generation_tokens", "prompt_tokens", "cache_hits", "cache_queries")


@dataclass(frozen=True)
class Rates:
    gen_tps: float | None = None
    gen_tps_avg: float | None = None
    prompt_tps: float | None = None
    prompt_tps_avg: float | None = None
    kv_hit: float | None = None
    kv_hit_avg: float | None = None
    running: float | None = None
    waiting: float | None = None
    kv_usage: float | None = None


def _rate(newer: Snapshot, older: Snapshot, field: str, seconds: float) -> float | None:
    """Tokens per second between two samples, or None if underivable."""
    new_val, old_val = getattr(newer, field), getattr(older, field)
    if new_val is None or old_val is None or seconds <= 0:
        return None
    return (new_val - old_val) / seconds


def _ratio(newer: Snapshot, older: Snapshot, num: str, den: str) -> float | None:
    """Hit ratio between two samples, or None if the denominator did not move."""
    d_num = _delta(newer, older, num)
    d_den = _delta(newer, older, den)
    if d_num is None or d_den is None or d_den <= 0:
        return None
    return d_num / d_den


def _delta(newer: Snapshot, older: Snapshot, field: str) -> float | None:
    new_val, old_val = getattr(newer, field), getattr(older, field)
    if new_val is None or old_val is None:
        return None
    return new_val - old_val


class VLLMStats:
    """Sliding window of scrapes, exposing windowed and since-start figures.

    ``*_avg`` means since the first scrape of this run, not since server boot:
    ``/metrics`` exposes no dependable uptime, and reporting a server-lifetime
    ratio beside a run-scoped rate in the same panel would mislead.
    """

    def __init__(self, window: float = 60.0, clock=time.monotonic) -> None:
        self._window = window
        self._clock = clock
        self._samples: deque[tuple[float, Snapshot]] = deque()
        self._first: tuple[float, Snapshot] | None = None
        # Highest value ever observed per monotonic field, since the last reset.
        # Tracked independently of `_samples`/`_first` because both can retain
        # (or evict) a different subset of history: `rates()` diffs against the
        # window's oldest sample *and* the since-first sample, either of which
        # may sit behind an intervening scrape that reported `None` for a
        # field. Comparing only against the latest sample would let that
        # `None` mask a real regression across either reference point.
        self._high_water: dict[str, float] = {}

    def add(self, snap: Snapshot) -> None:
        now = self._clock()
        if self._is_reset(snap):
            # Server restarted: every prior sample is from a different counter
            # lineage and differencing across the boundary is meaningless.
            self._samples.clear()
            self._first = None
            self._high_water.clear()
        self._samples.append((now, snap))
        if self._first is None:
            self._first = (now, snap)
        for field in _MONOTONIC:
            value = getattr(snap, field)
            if value is not None:
                self._high_water[field] = max(self._high_water.get(field, value), value)
        while len(self._samples) > 2 and self._samples[0][0] < now - self._window:
            self._samples.popleft()

    def _is_reset(self, snap: Snapshot) -> bool:
        # Compare against the high-water mark, not just the most recent
        # sample: a decrease anywhere relative to everything retained so far
        # is a genuine counter regression, even if it is invisible sample to
        # sample because of an intervening `None`.
        return any(getattr(snap, f) is not None and f in self._high_water and getattr(snap, f) < self._high_water[f] for f in _MONOTONIC)

    def rates(self) -> Rates:
        if not self._samples:
            return Rates()
        newest_t, newest = self._samples[-1]
        oldest_t, oldest = self._samples[0]
        window_s = newest_t - oldest_t

        first_t, first = self._first if self._first is not None else (newest_t, newest)
        avg_s = newest_t - first_t

        return Rates(
            gen_tps=_rate(newest, oldest, "generation_tokens", window_s),
            gen_tps_avg=_rate(newest, first, "generation_tokens", avg_s),
            prompt_tps=_rate(newest, oldest, "prompt_tokens", window_s),
            prompt_tps_avg=_rate(newest, first, "prompt_tokens", avg_s),
            kv_hit=_ratio(newest, oldest, "cache_hits", "cache_queries"),
            kv_hit_avg=_ratio(newest, first, "cache_hits", "cache_queries"),
            running=newest.running,
            waiting=newest.waiting,
            kv_usage=newest.kv_usage,
        )


class VLLMStatsPoller:
    """Poll ``/metrics`` on a daemon thread and feed a VLLMStats.

    A thread rather than an asyncio task on purpose: the pipeline saturates its
    event loop with concurrent workers, and an async poller would go stale exactly
    when throughput matters most. It also lets evaluate's largely synchronous flow
    reuse this unchanged.

    Every failure is silent by contract. ``available`` goes False and the panel
    renders as unavailable; nothing propagates to the caller.
    """

    def __init__(self, url: str, stats: VLLMStats, interval: float = 5.0, fetch=None) -> None:
        self._url = url
        self._stats = stats
        self._interval = interval
        self._fetch = fetch or _http_fetch
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False
        self.available = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vllm-stats", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self._interval)

    def _tick(self) -> bool:
        """One scrape. Returns True on success. Never raises."""
        try:
            parsed = parse_metrics(self._fetch(self._url))
            snap = snapshot_from(parsed)
            if all(getattr(snap, f.name) is None for f in fields(snap)):
                raise ValueError("no vllm metrics in response")
            self._stats.add(snap)
        except Exception as exc:
            if not self._warned:
                logger.warning("vLLM statistics unavailable at %s: %s", self._url, exc)
                self._warned = True
            else:
                logger.debug("vLLM statistics still unavailable at %s: %s", self._url, exc)
            self.available = False
            return False
        self._warned = False
        self.available = True
        return True


def _http_fetch(url: str) -> str:
    import httpx

    response = httpx.get(url, timeout=5.0)
    response.raise_for_status()
    return response.text


def format_rate(value: float | None) -> str:
    """Render a token rate. Absent is `-`, never `0` -- zero is a measurement."""
    if value is None:
        return "-"
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


# Every push writes this exact set of rows, in this order. A reporter's `set_stat`
# can only add or overwrite -- there is no way to take a row back off the panel --
# so a branch that writes a subset leaves the rest of the previous branch behind.
# That is what put a permanent `status: unavailable` next to live token rates: the
# pipeline's first tick fires before the poller's first scrape completes, and
# nothing ever overwrote that row again. A fixed row set makes the panel
# self-consistent in both directions by construction rather than by care.
#
# Values are also kept short on purpose, and the reason is a width budget rather
# than taste. At 80 columns -- the commonest pane, and the default -- the three
# stat groups split the pane three ways, leaving each panel 22 cells of content.
# The key column takes the widest label (`running`, 7) plus 2 of padding, so a
# value has 13 cells. The original single-row form, `6.1k tok/s  (avg 5.9k)`, is
# 22 and starved the key column down to 3: every label rendered as `st...`,
# `gen`, `pr...`, `kv...`, `ru...`, which surfaced neither the throughput nor the
# hit rate the panel exists for. Splitting the averages onto one shared row costs
# a single extra row of height and keeps every value inside 13 cells.
_VLLM_ROWS = ("status", "gen", "prompt", "avg g/p", "kv hit", "running")


def push_vllm_stats(rep, stats: "VLLMStats | None", poller: "VLLMStatsPoller | None") -> None:
    """Fill the reporter's `vllm` column. Shared by the pipeline and evaluate.

    Both branches must supply every row in `_VLLM_ROWS`; the write loop raises a
    KeyError if one forgets, which is why the values are assembled first rather
    than pushed inline. Unavailable renders the rates as `-` -- the same "no
    measurement" marker `format_rate` uses -- so a dead scraper cannot leave its
    last numbers on screen reading as though they were current.
    """
    if stats is None:
        return
    if poller is not None and not poller.available:
        values = dict.fromkeys(_VLLM_ROWS, "-") | {"status": "unavailable"}
    else:
        rates = stats.rates()
        values = {
            "status": "live",
            "gen": f"{format_rate(rates.gen_tps)} tok/s",
            "prompt": f"{format_rate(rates.prompt_tps)} tok/s",
            # gen / prompt, in the order of the two rows immediately above.
            "avg g/p": f"{format_rate(rates.gen_tps_avg)} / {format_rate(rates.prompt_tps_avg)}",
            "kv hit": "-" if rates.kv_hit is None else f"{rates.kv_hit:.0%}",
            "running": f"{rates.running or 0:.0f}  wait {rates.waiting or 0:.0f}",
        }
    for row in _VLLM_ROWS:
        rep.set_stat(row, values[row], group="server")
