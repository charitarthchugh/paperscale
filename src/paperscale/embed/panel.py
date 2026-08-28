"""The `embed` stat panel, and the sustained-queue advisory.

`push_vllm_stats` is deliberately not parameterised into serving both panels.
The mismatch is in the *inputs*, not the row names: it can reach exactly
`stats.rates()` and `poller.available`, and of embed's three `server` rows only
`tok/s` is reachable from that pair. `dim` comes from `--embed-dim` and the
Adapter, and the client half of `in-flight` is a counter that lives on
`EmbedClient` and did not exist anywhere before this feature. Threading a model
id, two dimensions and a live request counter through a function whose entire
premise is that it needs none of them costs more than a second writer of three
rows, so `push_vllm_stats` is left alone.

What *is* shared is `vllm_stats` itself: `format_rate`'s "absent renders `-`,
never `0`" rule, `Rates` and the poller (the scraping side is engine-specific
and already correct), and the fixed-row-set discipline recorded in
`_VLLM_ROWS`'s comment -- inherited here as a rule, not as a row tuple.
"""

from __future__ import annotations

from collections.abc import Sequence

from paperscale.vllm_stats import Rates, VLLMStats, VLLMStatsPoller, format_rate

# Every push writes this exact set of rows, for the same reason `_VLLM_ROWS`
# does: `set_stat` can add and overwrite but there is no way to take a row back
# off the panel, so a branch that writes a subset leaves the other branch's
# values on screen reading as though they were current. That is what put a
# permanent `status: unavailable` next to live token rates in the OCR panel.
#
# Three rows, not four, and that removes a height cliff rather than merely
# fitting. `_layout_budget` grows sections in the order bars, events, stats, and
# `_stat_columns` truncates each group from the tail; run the budget directly and
# `height=17` yields `stat_rows=3` while `height=18` yields 4. So below 18
# terminal rows a four-row `server` group silently drops its last row, which
# would be `in-flight` -- the saturation signal, i.e. the one row the group
# exists for. At three rows the group is complete wherever stats render at all,
# because MIN_STAT_ROWS is 3. The freed fourth slot stays empty; a three-row
# group in a four-row panel draws one blank line and costs nothing.
#
# `model` is the row that is missing, and it cannot be a row at all. At 80
# columns -- the default pane -- each of the three panels gets 22 cells, and the
# key column takes the widest label plus 2 of padding; embed's widest label is
# `in-flight` (9), which leaves 11 for a value. The two pinned model ids are 23
# and 31 characters, and basename-only does not fix it (`Nemotron-3-Embed-1B-BF16`
# is 24). It goes in the header instead, which is full pane width and where a
# value constant for the whole Invocation belongs; `pipeline.py` already titles
# the OCR reporter `paperscale . <model>` the same way.
EMBED_ROWS = ("dim", "tok/s", "in-flight")


def push_embed_stats(
    rep,
    stats: VLLMStats | None,
    poller: VLLMStatsPoller | None,
    *,
    stored_dim: int,
    native_dim: int,
    outstanding: int,
) -> None:
    """Fill the reporter's `server` column for an embedding Invocation.

    Unlike `push_vllm_stats` this does not return early when there are no server
    statistics. `dim` comes from the flag and the Adapter and the client half of
    `in-flight` is a counter this process owns, so both stay true with the
    scraper dead; only the two figures sourced from `/metrics` go absent. Their
    going absent is also the whole liveness signal here -- embed has no `status`
    row and, at three rows, cannot afford one.

    The unavailable branch is an empty `Rates` rather than a second dict of `-`
    literals. Every field of it is `None`, which is precisely what "no
    measurement" means, and the two branches then cannot drift apart the way two
    hand-written dicts eventually do.
    """
    if stats is None or (poller is not None and not poller.available):
        rates = Rates()
    else:
        rates = stats.rates()
    values = {
        "dim": f"{stored_dim}/{native_dim}",
        # `prompt_tps`, never `gen_tps`. Embeddings are prefill-only, so
        # `vllm:generation_tokens_total` never moves: the wrong field here reads
        # as an idle server rather than as a bug, and nothing at runtime can
        # distinguish the two. This is the one row that reads a *different* field
        # from the OCR panel rather than merely labelling one differently, which
        # is why it is called out here and not left to the row name.
        "tok/s": format_rate(rates.prompt_tps),
        # Two numbers in one slot: what `--concurrency` controls, over what says
        # the flag is set too high. `waiting`, not the OCR panel's `running` --
        # queue depth is the signal and admitted requests are not. The client
        # figure counts `/v1/embeddings` only: `/tokenize` is served CPU-side
        # in the API server process and never enters the engine scheduler, so it
        # cannot produce the queue it would be compared against, and folding it
        # in would inflate the left number with traffic that cannot cause the
        # thing the row measures.
        "in-flight": f"{_requests(outstanding)}/{_requests(rates.waiting)}",
    }
    for row in EMBED_ROWS:
        rep.set_stat(row, values[row], group="server")


def _requests(value: float | None) -> str:
    """Render a request count. Absent is `-`, never `0` -- zero is a measurement.

    Not `format_rate` itself: that renders 1200 as `1.2k`, which is right for a
    token rate and wrong for a queue of 1200 requests. Only the absent rule is
    shared, and it is the half that matters -- the OCR panel's `running` row
    prints `rates.waiting or 0`, which reports a dead scraper as an empty queue.
    """
    if value is None:
        return "-"
    return f"{value:.0f}"


# ~60 s of unbroken queue. Long enough that a burst of arrivals, or one Document
# whose Chunks all land at once, cannot trip it; short enough that an operator
# who mis-set the flag hears about it inside the first minute rather than at the
# end of a twelve-hour Invocation.
ADVISORY_WINDOW_S = 60.0

# The flag is named in the text on purpose, so the advice is actionable without
# reading the design. The value is the default, not the operator's setting: this
# function is given a queue history and nothing else, and the alternative --
# threading `args.concurrency` in -- buys a number the operator already knows
# they typed.
QUEUE_ADVISORY = "queue depth sustained; --concurrency 64 may be too high for this server"


def queue_advisory(rates_waiting_history: Sequence[tuple[float, float | None]]) -> str | None:
    """One event-pane line when the server's queue never empties. Advisory only.

    `rates_waiting_history` is `(monotonic timestamp, Rates.waiting)` pairs,
    oldest first, one per stats tick. `Rates` carries no timestamp of its own and
    `VLLMStats` keeps its samples private, so the caller pairs each observed
    value with the clock it already reads -- "no new plumbing" means no new
    scrape, not no bookkeeping.

    **It changes nothing.** The alternative was an adaptive throttler (Vespa's
    `DynamicThrottler` optimising `throughput / inflight^0.3`), which measures
    the server instead of asking it anything. Two things kept the simple thing:
    that throttler feeds a distributed store of genuinely unknown aggregate
    capacity, whereas this feeds one vLLM server whose queue depth is directly
    observable; and a control loop here would be a new failure mode --
    oscillation, and a throughput number that moves for reasons the operator
    cannot see -- bought against a signal already published.

    **Edge-triggered**: the string comes back on the one tick the window is first
    crossed and `None` on every tick after, so the caller needs no "already said
    this" flag and one repeated sentence cannot fill an eight-row event pane. A
    queue that drains and then builds again for another full window is a fresh
    observation and fires again.

    That does mean the caller must retain the whole episode. A `deque` with a
    `maxlen` shorter than the window evicts the run's start on every tick, which
    holds the measured span constant and can re-arm the edge every tick; keep the
    history unbounded, or bounded well above `ADVISORY_WINDOW_S` of samples.

    A missing measurement breaks the run exactly as an empty queue does: a failed
    scrape is not evidence of a queue, and treating `None` as "still busy" would
    turn a dead poller into an accusation against the flag.
    """
    history = list(rates_waiting_history)
    if len(history) < 2:
        return None
    started_at = None
    for stamp, waiting in reversed(history):
        if waiting is None or waiting <= 0:
            break
        started_at = stamp
    if started_at is None or history[-1][0] - started_at < ADVISORY_WINDOW_S:
        return None
    # The previous tick already spanned the window, so it already said this. The
    # run is at least two samples long by here, so `history[-2]` is inside it.
    if history[-2][0] - started_at >= ADVISORY_WINDOW_S:
        return None
    return QUEUE_ADVISORY
