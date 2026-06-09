# Recovery and Retry Redesign

Status: **implemented**. Supersedes the *default* recovery behavior described in
`ledger-recovery.md` (ambiguous-by-default). The ledger state machine and
`ambiguous`/`reconcile` machinery are retained but become opt-in for metered
endpoints (`--metered`). See `runner.py` (`_recover_expired_attempts`,
`_process_one_page`, `_demote_terminal`), `profiles/base.py:ModelOcrProfile.remediation_for`,
and tests in `tests/test_recovery_retry_redesign.py`.

Implementation extensions (to make every document complete, validated at 233/233
pages over 30 scanned law PDFs against a real DeepSeek-OCR-2 server):
- **Rendering uses PDFium** (`pypdfium2`), which rasterizes scanned ImageMask/CCITT
  pages that the old `pdf_oxide` backend skipped (blank image → empty OCR).
- **`empty_output` joins the render-remediation set** — an empty completion on a
  content page re-renders at higher DPI rather than re-rolling identically.
- **Blank-page acceptance** — a near-blank render (`png_dark_fraction` below
  `_BLANK_INK_THRESHOLD`) whose OCR yields no readable content (empty, or a VLM
  repetition loop on noise) is committed as a *successful empty page*, so a blank
  separator page never strands a job at `failed_terminal`. The ink gate ensures real
  content pages still remediate normally.

## Motivation

The v1 recovery contract escalates every crashed `in_flight` page to
`ambiguous`, blocking on an operator to avoid a duplicate provider call. That
contract assumes provider calls carry a billable or non-idempotent side effect.
Paperscale's primary target is **self-hosted** inference (a local vLLM endpoint,
`api_key="paperscale-local"`), where a duplicate call costs only GPU time and
latency. Page output is idempotent: artifacts are keyed by page number and the
content is deterministic for a given image + profile, so a re-call cannot
corrupt the document. The redesign optimizes for "never block on a human" and
treats duplicate-call avoidance as the exceptional, metered-only case.

## Decisions

1. **Auto-requeue by default.** A crashed `in_flight` page with an expired lease
   recovers to `pending` and is retried on the next `run`/`resume`. The
   `ambiguous` state is no longer the default outcome.

2. **Duplicate calls are idempotent.** Re-calling the provider for an
   already-attempted page cannot harm correctness — artifacts are page-keyed and
   content-deterministic. The only cost is GPU/latency. This is the premise that
   makes auto-requeue safe.

3. **`metered` is a transient switch.** A `metered: bool = False` flag on
   `RunnerConfig`, set by a `--metered` CLI flag, restores the ambiguous-on-crash
   behavior for billable/non-idempotent endpoints. It is **not** persisted in the
   manifest: `resume` reads the live flag, so metered-safety depends on operator
   discipline at every invocation. This is an accepted tradeoff for a
   self-hosted-first tool; it is a choice, not an oversight.

4. **Adopt-then-requeue on recovery.** The provider call is the last slow step
   before the artifact is committed, so the most likely crash window is *after*
   the artifact is durably written but *before* the index records `succeeded`.
   Recovery of an `in_flight` page therefore:
   - if the artifact file exists **and** its `fingerprint` matches the attempt's
     fingerprint → adopt it, mark `succeeded`, trigger assembly (zero provider
     calls);
   - else → `pending` (or `ambiguous` when `metered`).

   The fingerprint match proves the artifact came from *this* attempt's exact
   request, not a stale/superseded epoch. This mirrors the existing
   `repair_index` adoption logic.

5. **Adoption is safe because writes are atomic.** A present artifact is always
   complete: `write_json_atomic` writes a temp file, `fsync`s it, then
   `os.replace`s into place — a reader sees either nothing or the fully-written
   file, never a torn one. `ensure_known_schema` fail-closes on read. The
   parent-directory `fsync` exists to make the *rename durable* (not lose a
   commit on power loss), not to prevent torn reads.

6. **Hard retry cap of 8.** `max_attempts: int = 8` on `RunnerConfig`. The
   per-page `epoch` becomes load-bearing: once attempts exceed the cap, a
   `failed_retryable` page is demoted to `failed_terminal` with a diagnostic.
   Without this, a deterministically-bad page (e.g. an unreadable scan producing
   mojibake) is `retryable` forever and re-calls the provider on every `resume`.
   Refusals remain terminal on first sight (re-rolling identical input is
   provably futile).

7. **Automatic diagnostic-driven remediation.** The 8 attempts are not identical
   re-rolls. Each failure's `diagnostic` (already persisted on the page entry)
   drives an escalating remediation on the next attempt, via a
   diagnostic → overrides table **owned by the profile** (only the profile
   understands its own decoding/render semantics):

   | diagnostic | remediation |
   |---|---|
   | `truncation_indicator`, `length_anomaly` (too long) | increase decoding token budget |
   | `mojibake`, `control_characters` | re-render at higher DPI |
   | `repeated_ngram`, `repeated_character` | raise `repetition_penalty` / nudge temperature |
   | unknown | identity (plain re-roll) |

   Each remediation changes the request fingerprint — intentional: the ledger
   records exactly which remediation produced the eventual success, and
   adopt-then-requeue (Decision 4) still matches per attempt.

8. **Remediations accumulate.** Overrides layer additively across attempts rather
   than being rebuilt from base on each failure. Reacting only to the *last*
   diagnostic would discard prior fixes (e.g. a token bump for `truncation` would
   reset a DPI bump from an earlier `mojibake`) and could oscillate without
   converging. This requires persisting the effective `decoding`/`render_options`
   (or an ordered remediation list) on the page entry, so a crash mid-ladder
   resumes with corrections intact. Each dimension has a ceiling (max DPI, max
   token budget) so monotonic escalation cannot exceed provider/image limits; the
   8-attempt cap is the outer backstop. No per-dimension early-terminal logic for
   now (YAGNI).

9. **No schema bump.** All additions (accumulated overrides, carried per-page
   state) are additive and optional, read with `.get(key, default)`. The
   fail-closed guard rejects only `schema_version > current`; unfamiliar fields
   are tolerated. Old jobs lack remediation history and correctly default to the
   base profile. Stays `schema_version = 1`. Discipline: additive+optional → same
   version; required-or-breaking → bump. Never read a new field with `payload[key]`.

## Code touchpoints

- `runner.py:_recover_expired_attempts` — branch on `self.config.metered`; add the
  adopt path (check `_artifact_rel` existence + fingerprint match before requeue).
- `runner.py:_process_pages` — read `epoch` + accumulated overrides at the top;
  demote to `failed_terminal` at the cap; build the request from
  `profile.with_overrides(...)`.
- `profiles/base.py:ModelOcrProfile` — new `remediation_for(diagnostic) -> overrides`
  table; per-dimension ceilings.
- `runner.py:_write_indexes` — carry `overrides` (or remediation list) on the page
  entry alongside `epoch`/`diagnostic`.
- `runner.py:RunnerConfig` — add `metered: bool = False`, `max_attempts: int = 8`.
- `cli.py` — `--metered` flag on `run`/`resume`; `--max-attempts` if exposed.

## Completion and settling

10. **Terminal pages keep hard-failing (for now).** A `failed_terminal` page
    (refusal, or 8-exhaustion) blocks `JobStatus.complete`; the job assembles
    output only when the operator passes `--allow-partial`. No auto-degrade to
    partial-complete. Deferred, not rejected — revisit if the no-human-block
    philosophy wins out.

11. **Minimal settled-reporting.** Even under hard-fail, `status`/`resume` must
    distinguish **settled-with-failures** (no `pending`/`failed_retryable`/
    `in_flight`/`reserved` pages remain — only `succeeded` + terminal, so resume
    cannot progress) from **has-retryable-work** (resume will make progress).
    This is a *reporting-only* change: no partial output is written without
    `--allow-partial`, exit stays non-zero, but the message becomes actionable —
    e.g. `job X: settled with 1 terminal failure (page 137); resume will not
    progress — pass --allow-partial to assemble 199/200, or re-source page 137`.
    Without it, a settled job's `resume` is an infinite silent no-op
    indistinguishable from a job that is still progressing.

### Additional code touchpoint

- `runner.py:JobStatus` / `_write_indexes` — derive a `settled` boolean (no
  pending/retryable/in_flight/reserved pages) from the counts already computed;
  surface it in `status`/`resume` messaging and exit-code selection.
