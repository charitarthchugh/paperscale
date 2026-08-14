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
