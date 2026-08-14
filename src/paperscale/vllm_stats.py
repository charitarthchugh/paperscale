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

import math
import re
from dataclasses import dataclass

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
