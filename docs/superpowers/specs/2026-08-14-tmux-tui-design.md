# Terminal TUI with vLLM statistics

Date: 2026-08-14
Status: awaiting review

## Problem

Three gaps, one feature.

**The existing dashboard breaks under tmux.** The renderer never measures the
terminal. `_render()` (`tui.py:148-164`) does not reference `console.size` at any
point; the only height expressed anywhere in the module is the hardcoded
`height=self._max_log + 2` on the events panel (`tui.py:163`). The frame's height is
therefore a function of run state — one row per `set_stat` key, one row per phase,
plus panel chrome — and is completely uncorrelated with the height of the terminal
it is being drawn into.

Rich's `Live` can only overwrite in place when the frame fits. Once run state pushes
it past whatever the terminal happens to be, every refresh re-emits the whole frame
and the terminal scrolls. The observed result is dozens of stacked copies of the
stats panel, each clipped at a different row, with only the bottom few rows showing
live progress. Two further faults make it worse:

- No alternate screen, so an unbounded frame scrolls rather than being clipped.
- An eager redraw storm: `advance()`, `log()`, and `set_stat()` each force a full
  re-render (`tui.py:80,137,141`), so a run emits thousands of frames instead of
  letting `Live` refresh on its own clock.

The fix is therefore not to make the frame smaller. It is to make height a measured
input to rendering at all, which it currently is not.

Glyphs and colour are **not** the problem. The reported pane renders rounded
box-drawing, 256-colour, and bold correctly. Capability fallbacks are insurance for
genuinely limited terminals, not the fix for this bug.

**The OCR pipeline has no TUI at all.** `--tui` exists only on `evaluate`
(`cli.py:39,170`). The pipeline logs to stderr and dumps `MetricsKeeper` and
`WorkerTracker` tables every 10 seconds (`pipeline.py:929-934`), which is the
longest-running command and the one that most needs a dashboard.

**Server throughput and failure counts are invisible.** No vLLM statistics are
surfaced anywhere. Document outcomes are classified at `pipeline.py:551-563` but
only logged, never counted, so there is no live view of how many documents shipped
degraded versus were discarded.

## Goals

- A full-screen dashboard for the OCR pipeline, reusing it for `evaluate`.
- Correct rendering under tmux and in other limited terminals: no scrolling
  artefacts, no flicker, and correct behaviour across pane splits, zoom, and
  detach/reattach.
- Live vLLM statistics: token throughput and prefix-cache hit rate.
- Live processing counters, including partial versus full document failures.

## Non-goals

Keybindings, mouse, and scrollback (the choice was a read-only dashboard, not an
interactive app). Textual. Sparklines or plots. Per-worker tables in the UI — the
`WorkerTracker` table stays in the log file. Config files. Changing pipeline
behaviour: the TUI is instrumentation, and a run must produce identical output
with and without it.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Pipeline first, then `evaluate` | Both drive a vLLM server; `evaluate` already speaks `ProgressReporter` |
| Renderer | Rich `Live`, alternate screen | `rich` is already the `tui` extra; keeps `tui.py`'s protocol intact; no restructuring of the pipeline's asyncio main |
| Logs | File plus tail in pane | Alt-screen has no scrollback, so logs must survive the run |
| Averaging | Sliding window beside lifetime | Mirrors `MetricsKeeper`; lifetime prefix-cache ratio is near-frozen after 596k queries and no longer reflects current behaviour |
| Enablement | Opt-in `--tui` | No behaviour change for existing invocations |

## Architecture

```
src/paperscale/vllm_stats.py   NEW   parse_metrics / VLLMStats / VLLMStatsPoller
src/paperscale/tui.py          EDIT  RenderStyle, Budget, grouped stats, anti-jank
src/paperscale/pipeline.py     EDIT  outcome counters, --tui, log routing, wiring
src/paperscale/cli.py          EDIT  pass the pplx server URL into the vllm panel
```

`vllm_stats.py` has no dependency on `tui.py`, and `tui.py` has no dependency on
`vllm_stats.py`. Callers own the poller and push values in through `set_stat`, so
the reporter keeps the "pure instrumentation" contract its module docstring
promises.

## vLLM statistics

### Source

`/health` returns a zero-byte 200 and carries no statistics; it is a liveness probe
only. All numbers come from `/metrics` in Prometheus text format. Verified against a
live server (415 `vllm:` series). The URL is derived from the existing server
argument by stripping a trailing `/v1` and appending `/metrics`, so
`http://localhost:8000/v1` becomes `http://localhost:8000/metrics`.

### Metric names

vLLM V1 does **not** expose `vllm:avg_generation_throughput_toks_per_s` or
`vllm:avg_prompt_throughput_toks_per_s`; those were V0 gauges and are absent
(confirmed: zero matches). Only cumulative counters remain, so all rates are derived
client-side.

Each logical metric resolves against an ordered candidate list, using the first name
present, so the scraper survives version drift:

| Logical | Candidates (in order) |
|---|---|
| generation tokens | `vllm:generation_tokens_total` |
| prompt tokens | `vllm:prompt_tokens_total` |
| cache hits | `vllm:prefix_cache_hits_total`, `vllm:prompt_tokens_cached_total` |
| cache queries | `vllm:prefix_cache_queries_total`, `vllm:prompt_tokens_total` |
| running | `vllm:num_requests_running` |
| waiting | `vllm:num_requests_waiting` |
| kv usage | `vllm:kv_cache_usage_perc`, `vllm:gpu_cache_usage_perc` |

An unresolvable logical metric renders as `-`, not as zero. Zero is a measurement;
a dash is an absence, and conflating them would misreport an idle server as broken.

### Label handling

Series carry `engine` and `model_name` labels. With `--data-parallel-size > 1` there
is one series per engine. Counters and request gauges are **summed** across engines;
`kv_cache_usage_perc` is **averaged**, since it is already a fraction.

### Derivation

`VLLMStats` keeps a deque of `(monotonic_timestamp, snapshot)` trimmed to a 60-second
window, and exposes both a windowed and a lifetime figure for each rate:

| Displayed | Window | Lifetime |
|---|---|---|
| gen tok/s | Δ`generation_tokens` / Δt | total / uptime |
| prompt tok/s | Δ`prompt_tokens` / Δt | total / uptime |
| kv hit | Δhits / Δqueries | hits / queries |
| running, waiting | instantaneous | — |

**Counter resets.** A vLLM restart returns counters to zero. Any negative delta means
a reset: the window is cleared and rates report `-` until it refills, rather than
emitting a negative or absurd rate. Lifetime figures restart from the new baseline.

**Zero-denominator.** Δt at or below zero, or zero cache queries in the window,
yields `-`.

### Poller

`VLLMStatsPoller` runs a daemon thread with a synchronous `httpx.Client`, polling
every 5 seconds (`--tui-poll-interval`). A thread rather than an asyncio task,
deliberately: the pipeline saturates its event loop with concurrent workers, and an
async poller would report stale numbers exactly when throughput matters most. It also
lets `evaluate`'s largely synchronous flow use identical code.

Shutdown is a `threading.Event` set by the reporter's `__exit__`, joined with a
2-second timeout.

**Failure is always silent.** Connection refused, timeout, 404 (backends such as
`qianfan` and `surya` have no `/metrics`), non-200, or an unparseable body all mark
the panel unavailable. The first failure logs at WARNING, subsequent ones at DEBUG.
The panel shows `vllm  unavailable` and the run continues. A statistics panel must
never be able to end a twelve-hour OCR job.

## Processing counters

### Document outcomes

The classification already exists at `pipeline.py:551-567`; only counters are added,
at these exact branch points.

| Counter | Site | Meaning |
|---|---|---|
| `docs_ok` | `:564` | no fallback pages |
| `docs_partial` | `:559` | some fallback pages, at or under `--max_page_error_rate`, document **shipped** |
| `docs_discarded` | `:553` | above `--max_page_error_rate`, document **dropped** |
| `docs_crashed` | `:565` | exception in `process_single_pdf` |
| `docs_missing` | `:572` | input file not found |

`docs_partial` and `docs_discarded` are the pair that shows whether
`--max_page_error_rate` suits a corpus, live rather than reconstructed from logs.

### Page statistics

Already recorded by `MetricsKeeper`, currently visible only in the 10-second log dump:
`completed_pages`, `failed_pages`, `blank_pages`, `fd_exhaustion_drops`,
`quality_reject_{kind}`, `finished_on_attempt_{i}`, `finished_on_parallel_retry`,
`server_input_tokens`, `server_output_tokens`.

`retries` in the panel is derived as `sum(count for attempt i > 0)` from the
`finished_on_attempt_{i}` family, collapsing an unbounded key family into one row.

**`quality_reject_{kind}` is dynamically keyed** — one counter per verifier finding
kind, so the set is unbounded at runtime. It must never drive panel height. The panel
shows the top three by count plus `+N more`; the full breakdown goes to the log file
and the final summary.

### The issues column in `evaluate`

The document-outcome counters above are pipeline-only. `evaluate` populates the same
column from its own failure path: `_handle_doc_failure` (`evaluation/pplx.py:282`)
fires when a document exhausts its retry budget, giving a `docs_failed` count, beside
a `skipped` count of documents resume already had. The column has the same purpose in
both commands — what did not come out clean — with command-specific rows.

## Layout

Fixed height by construction: the frame is always exactly `console.size.height` rows.
The top region is a three-column grid, trading abundant width for scarce height.

```
paperscale · olmocr                                     elapsed 01:23:45
┌─ run ──────────────┬─ vllm  localhost:8000 ──┬─ issues ─────────────┐
│ docs      142/500  │ gen       412 tok/s     │ partial          7   │
│ pages   1204/2900  │ prompt    6.1k tok/s    │ discarded        2   │
│ tokens      4.1M   │ kv hit       93%        │ crashed          1   │
│ retries       38   │ running 8    wait 2     │ blank           14   │
└────────────────────┴─────────────────────────┴──────────────────────┘
 work items  ━━━━━━━━━━━━━━━╺────────────  142/500        1:23:45
┌─ events ───────────────────────────────────────────────────────────┐
│ 12:04:11  WARN  b91… 3/412 fallback pages, shipping degraded       │
│ 12:04:02  ERR   a3f… 9/12 fallback pages > 0.004, discarded        │
└────────────────────────────────────────────────────────────────────┘
```

- Header is one row, not a panel.
- Each bordered panel costs two rows of chrome.
- Progress bars are capped at `bar_rows`; a `+N more` line appears if there are more
  phases than rows.
- Log lines use `no_wrap=True, overflow="ellipsis"` so a long path cannot reflow the
  layout.
- Below 60 columns the grid stacks to one panel per row; `issues` is the last column
  dropped, because a failure you cannot see is worse than a throughput figure you
  cannot see. A vertical split halves pane width, so this is a routine path rather
  than an edge case.
- `Live(screen=True, auto_refresh=True, refresh_per_second=4)`. Alt-screen clips an
  over-tall frame instead of scrolling it, but it is **not** load-bearing: tmux
  ignores the request when `alternate-screen` is off, silently falling back to inline
  rendering. The height budget is the guarantee; alt-screen is a second layer that may
  or may not be present.
- Every eager `_refresh()` is removed from `advance()`, `log()`, and `set_stat()`.
  `_refresh()` survives only for `__exit__`.

### Height budget

```python
@dataclass(frozen=True)
class Budget:
    stat_rows: int
    bar_rows: int
    event_rows: int

def _layout_budget(height: int, n_stats: int, n_phases: int) -> Budget:
    """Split `height` rows so the frame never exceeds the terminal.

    Invariant: 1 + (stat_rows + 2) + bar_rows + (event_rows + 2) == height
    """
```

Floors are 3 stat rows, 1 bar row, 2 event rows; surplus goes to bars first, then
events. Below the floor total the events panel is dropped entirely, then the issues
column, then the vllm column. **This function's starvation policy is the one open
decision** (see Open questions).

### Running under tmux

**The budget is recomputed on every render, never cached at `__enter__`.** This is
the load-bearing constraint for tmux. Pane geometry changes constantly and abruptly —
splitting, zooming with `prefix + z`, and detaching then reattaching from a client of
a different size. Rich re-reads `console.size` on each refresh, so recomputing the
budget inside `_render()` means the layout self-corrects within one tick (250 ms at
4 fps) with no `SIGWINCH` handler. Caching the budget would reintroduce the original
bug the first time a pane was zoomed.

`console.size` reports the **pane**, not the window, and tmux's status line has
already been subtracted. No adjustment is needed.

**Alt-screen may not happen.** When `alternate-screen` is off, tmux discards the
request and the program renders inline without knowing it. This is why the height
budget, not alt-screen, is the guarantee.

**Alt-screen costs scrollback.** Content drawn to the alternate screen does not enter
tmux's copy-mode history and `capture-pane` will not retrieve it. Accepted
deliberately: the dashboard's scrollback consists of superseded frames and is worth
nothing, while the log file is the durable record. This is why file logging is
mandatory rather than optional when `--tui` is on.

`TERM` inside tmux is `screen-256color` or `tmux-256color`, describing tmux rather
than the outer terminal. Where the `tmux-256color` terminfo entry is missing, Rich
degrades on its own. Truecolour requires an `RGB`/`Tc` terminal override; without one
Rich uses 256 colours. Neither needs handling.

### Terminal capability

`terminal_profile(console, env) -> RenderStyle` resolves once at `__enter__`, holding
`box`, `spinner`, `bar_chars`, and `use_screen`. Unlike the budget, these describe the
terminal's capabilities rather than its geometry and do not change mid-run.

| Condition | Detectable | Fallback |
|---|---|---|
| `TERM=dumb` | yes | `NullReporter`, no `Live` |
| non-UTF-8 encoding | yes | `box.ASCII`, `line` spinner, ASCII bars |
| `TERM=linux` | yes | box-drawing is fine, braille is not: `line` spinner |
| outer terminal lacks glyphs | **no** | `PAPERSCALE_TUI_ASCII=1` |

Glyph coverage cannot be detected, and tmux makes this structurally worse: `TERM`
describes tmux, so the outer terminal's font and Unicode support are invisible to the
program, and a UTF-8 locale says nothing about whether the rendering font has a
braille block. The escape hatch has to be explicit. `PAPERSCALE_TUI_ASCII=0` forces
rich glyphs when detection is over-cautious. Both are documented in the README.

## Log routing

With `--tui` active:

- Remove `console_handler` from `logger` and `server_logger` (`pipeline.py:74-79`).
  Nothing may write to stderr underneath `Live`.
- Default `--disk_logging` to `<workspace>/logs/run-<pid>.log` when unset, reusing the
  existing mechanism at `pipeline.py:1107-1112`. An explicit `--disk_logging` wins.
- Attach a `ReporterLogHandler` forwarding WARNING and above into the events pane,
  formatted as `HH:MM:SS  LEVEL  message` and truncated to the pane width.
- `metrics_reporter()` pushes into `set_stat` instead of logging tables when a live
  reporter is present. With `NullReporter` it behaves exactly as today.

After `Live.__exit__` restores the normal screen, print the final metrics summary and
the log file path to stderr. Alt-screen erases the dashboard on exit, so without this
a completed run leaves a bare prompt and no numbers.

`evaluate` has no `--workspace` and no `--disk_logging`, so it defaults its log file
to `<dirname of --db>/logs/evaluate-<pid>.log` and gains `--disk-logging` for parity.
Both commands gain `--tui-poll-interval` (default 5 seconds) for the vLLM scrape.

## Reporter interface change

`set_stat` gains a keyword-only group:

```python
def set_stat(self, name: str, value, *, group: str = "run") -> None: ...
```

Backwards compatible — existing calls land in `run`. `NullReporter` continues to
ignore it. Groups map to grid columns: `run`, `vllm`, `issues`.

## Testing

`tests/test_vllm_stats.py`

- `parse_metrics` against a fixture captured from a real server, including comment
  lines, `_created` series, and histogram buckets.
- Multi-engine summing for counters and gauges; averaging for `kv_cache_usage_perc`.
- Candidate-name resolution, including absence rendering as `-` rather than `0`.
- Rate arithmetic with an injected clock.
- Counter reset produces `-`, never a negative rate.
- Zero Δt and zero cache queries produce `-`.
- Poller: 404, connection error, and malformed body each mark unavailable without
  raising, and log at WARNING exactly once.

`tests/test_tui.py` (extending the existing `_FakeTTY` style)

- `_layout_budget` satisfies its invariant for heights 8 through 60 across stat and
  phase counts, and never returns a negative row count. The range starts below the
  11-row floor total so the degradation path (drop events, then issues, then vllm) is
  covered, not just the comfortable case.
- Rendered frames at 40×12, 80×24, and 200×60 — a vertical split, a plain pane, and a
  zoomed pane — contain no line wider than the console and no more lines than its
  height. This is the regression test for the reported bug.
- Resizing the console **between** two `_render()` calls yields a frame matching the
  new size, proving the budget is recomputed rather than cached. This is the
  regression test for pane zoom and detach/reattach.
- ASCII mode emits zero codepoints outside Latin-1.
- `terminal_profile` parametrized over `TERM` (`xterm-256color`, `screen-256color`,
  `tmux-256color`, `linux`, `dumb`) × encoding × the env override, including
  `TERM=linux` selecting a braille-free spinner and `TERM=dumb` yielding
  `NullReporter`.
- Grouped stats render under the correct column.
- `quality_reject_*` with ten distinct kinds renders three rows plus `+7 more`.

`tests/test_pipeline_units.py`

- Each document outcome counter increments on its branch: clean, partial, discarded
  (over `max_page_error_rate`), crashed, missing.

Runner is `poetry run pytest -q`.

## Open questions

**`_layout_budget` starvation policy.** When rows run short, what gives way first?
Guaranteeing bars and stats suits watching a long run for throughput; guaranteeing
events suits checking in because something looks wrong. A third option is hard floors
on all three with the surplus distributed proportionally. The default written above
is floors of 3/1/2 with surplus to bars then events, to be confirmed or replaced.

## Out of scope

`cpu_vs_wall` in `metrics.py:178` is dead code. Removing it is unrelated to this work.
