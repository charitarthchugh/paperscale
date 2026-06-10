# Lazy multi-process shared-queue job processing

> **Status: APPROVED for execution** (team mode, 2026-06-09). Ralplan consensus: Architect REVISE→incorporated; Codex Critic REJECT→REJECT→ITERATE→ITERATE→APPROVE over 5 iterations. Deliberate mode.
>
> **For agentic workers:** implement test-first, task-by-task. Steps use checkbox (`- [ ]`) tracking. Each production slice starts from a failing test.

## Problem

Batch `run --input-list`/`enqueue` registers one job per document *before* any OCR begins. For ~5000 PDFs this enqueue phase was an invisible ~5-minute, fsync-bound stall (measured 14:05:42→14:10:37 on a real run) before the first page. Committed fix #1 (skip index fsync at enqueue) + #4 (log the phase) took it to ~2.2 min, but the first page still waits for the whole batch.

## Goal (scoped)

For a **new `plan`+`work` lazy workflow**:
1. The first OCR page's provider call starts **before the whole batch is materialized** (no full-batch wait).
2. Multiple `paperscale work` processes **drain one workload** concurrently.
3. Job ids and output filenames stay human-readable.
4. Eager `run`/`enqueue` paths unchanged.

**Concurrency guarantee (accepted scope):** lazy provides **exactly the eager `work()` guarantee — not exactly-once.** Pre-existing claim-fencing weaknesses (claim-epoch is not the page-attempt fencing epoch; `--metered` recovers only expired *indexed* attempts, not two concurrent live finishers) are **inherited unchanged, out of scope**, documented as residual risk; `--metered` remains the billable-duplicate mitigation under the pre-existing lease-expiry race.

## Design (Option A — candidate manifest + lazy materialization)

Naming note: distinct from the existing page-level `JobScheduler.build_lazy_queue`. New surface uses `candidates` / `materialize_job`.

1. **`plan` command.** Read input-list → dedup inputs by **absolute path** (drop+log exact dupes) → global in-memory **stem dedup** (`doc`, `doc-1`, …). Write **one immutable** `candidates/<workload_id>.jsonl` (`workload_id = <epoch_seconds>-<shortuuid>`) via **one atomic durable commit** (temp + fsync + rename + parent-dir fsync). Each record carries the **complete, deterministic job intent stamped at plan time**: `{job_id, input_path(abs), output_path(abs), profile, model, base_url, capacity, created_at}`. No `pdfinfo`, no per-job dirs at plan time.

2. **`materialize_job(record)`.** Tolerates the claim-created dir. If a manifest already exists → compare against the **complete** record (every field); full match = no-op; **any** mismatch = skip + error log (cross-workload job-id collision with divergent intent). Else: run `pdfinfo`; build a **deterministic** manifest using `record.created_at`; write manifest (fsync — the completion barrier) then indexes (fsync=False).

3. **Missing-index recovery `_pages_from_manifest(job_id, manifest)`.** Extracted from `repair_index`'s per-page loop, seeded `old_pages={}` (can't reuse `repair_index` directly — it first reads the missing status index). Adopts on-disk page artifacts as succeeded, marks the rest pending. Handler: manifest present → `_read_pages_from_status`; on `CompactIndexError` → `_pages_from_manifest`.

4. **Heartbeat across materialization.** On winning `try_claim`, the handler starts **one** `threading.Thread`: `while not stop_event.wait(interval): claim = store.heartbeat(claim)`, `interval = config.claim_heartbeat_seconds`. Covers sync `pdfinfo` **and** async page processing; subsumes the in-pool heartbeat for the lazy path. **Shutdown is strictly ordered in `finally`: signal (`stop_event.set()`) → join (thread) → release (claim)** — so no tick can recreate `claim.json` after release. Claim shared via a holder so heartbeat/release agree on epoch (mirrors `_drive_claimed`).

5. **Durable terminal failure.** New fsync'd `failed.json` (mirrors `done.json`), written when materialize fails (corrupt/unreadable PDF). Discovery checks **`is_done` before `is_failed`** (`done` precedence; markers may transiently coexist under lease races — not asserted exclusive). A corrupt PDF is **terminal** (not retried forever, not falsely done). **Retry is explicit:** `work --retry-failed` unlinks `failed.json` for the candidates it processes, then claims+materializes normally (stable job identity, no id reallocation).

6. **`work` discovery.** Snapshot/**single-pass** (parity with existing `work()`). Drain every `candidates/*.jsonl` (skip live-owned peers), **then** legacy `jobs/` scan for eager workloads; dedup by job_id (claim + is_done/is_failed are idempotent; seen-set is perf only). A peer crashed mid-materialize is recovered on the **next** `work` invocation after lease expiry — same re-run contract as resume; explicitly not revisited within one invocation.

7. **Inject document-claim lease/heartbeat.** Add `RunnerConfig.claim_lease_seconds=60.0` and `claim_heartbeat_seconds=20.0` (defaults preserve current behavior; named distinctly from the page-attempt `lease_seconds=300`). `_claim_store()` passes them through instead of the hardcoded `_CLAIM_LEASE_SECONDS`/`_CLAIM_HEARTBEAT_SECONDS`. Enables a provable heartbeat-renewal test.

## Pre-mortem (detection / containment / recovery / residual)

1. **Crash post-manifest-fsync / pre-index-write:** index missing → handler never trusts index presence → `_pages_from_manifest` adopts artifacts + marks rest pending → no residual beyond a re-read.
2. **Concurrent expired-lease reclaim (two epoch N+1):** non-CAS by design (OUT OF SCOPE) → page-keyed deterministic artifacts → epoch reconcile drops superseded attempts → residual = duplicate provider call (`--metered` → ambiguous). Same as eager.
3. **Slow `pdfinfo` > lease:** heartbeat age → background heartbeat thread from claim → peer reclaim → redundant deterministic materialize (identical bytes) → residual = rare double `pdfinfo`, no semantic harm.
4. **Cross-workload job-id reuse, divergent intent:** manifest-vs-complete-record mismatch → immutable per-workload files → skip + error log → residual = that doc not processed by the colliding workload (operator-visible).
5. **Malformed candidate line:** JSON/schema error → per-line try/except → skip+log+continue → residual = that doc skipped.
6. **Unreadable/corrupt PDF at work time:** `pdfinfo` nonzero/FileNotFound → per-candidate try/except → `mark_failed` + log + continue → residual = doc terminal until `--retry-failed`.
7. **Heartbeat resurrection after release:** post-join spy tick → signal→join→release ordering → prevented → no residual.
8. **done/failed coexistence:** both files present → done-precedence check order → treat as done → residual = spurious `failed.json` lingers (harmless).

## Test plan (TDD)

- **Unit:** abspath-dedup drops exact dupes; stem dedup; record carries complete intent incl `created_at`; candidates file = one atomic durable commit (torn temp never visible; parent-dir fsync'd); `materialize_job` no-op on full match / skip+log on any field mismatch; `_pages_from_manifest` adopts artifacts + marks rest pending; deterministic manifest (two materializations → byte-identical); `mark_failed` writes fsync'd `failed.json`, discoverable.
- **Integration** (fake renderer/provider/clock/crash-hook): `plan`→`work` end-to-end; resume partially-materialized; corrupt PDF → `failed.json`, drain continues, skipped on plain re-run, retried under `--retry-failed`; crash post-manifest/pre-index → rebuild adopts artifacts.
- **Ordering (deterministic perf):** instrumented provider+renderer record call order; assert **first provider invocation precedes the last candidate's materialization**. Wall-clock is a *separately labeled* benchmark (hw/fs/cache/reps/percentiles), not pass/fail.
- **Heartbeat renewal (provable):** two runners share a state root, `claim_lease_seconds=0.05`, `claim_heartbeat_seconds=0.01`. Worker A wins, materialize stub blocks on a barrier ~0.3s (real time, ≫ lease). During block: heartbeat spy records ≥1 tick AND Worker B's `try_claim` returns `None` (renewal proven: takeover attempted after the 0.05s original deadline). Then signal→join→release leaves `claim.json` absent, no post-join tick. **Control:** heartbeat disabled → B's `try_claim` succeeds after >0.05s (takeover observable → positive case meaningful).
- **Concurrency (real OS subprocesses):** two `work` over one workload → deterministic page artifacts, done-precedence, no crash on claim races; barrier-controlled expired-lease reclaim; worker SIGKILL at each boundary (pre-claim, post-claim/pre-manifest, post-manifest/pre-index, mid-page) → next invocation recovers; output-path collision attempt → dedup/skip, no different-content overwrite.

## Acceptance criteria

- `plan` writes exactly one immutable `candidates/<id>.jsonl` (atomic durable commit); no per-job dirs.
- Instrumented ordering test passes: first provider call precedes last materialize.
- **No new double-execution window beyond eager mode** (concurrency tests); duplicate calls only via the pre-existing lease-race path, `--metered`-governed. **No exactly-once claim.**
- Killing `work` mid-run leaves no orphan job dirs for un-reached docs; un-materialized entries live only in the immutable candidate file; next `work` finishes them.
- One corrupt/duplicate/malformed candidate never aborts the worker or blocks peers; corrupt PDF is durably terminal (`failed.json`), retried only via `--retry-failed`.
- Manifest materialization is deterministic (`created_at` from record); heartbeat keeps the lease live across a materialize longer than the lease; shutdown leaves no resurrected claim.
- All 167 existing tests pass; eager paths unchanged.

## Task breakdown (implementation order, each test-first)

- [ ] **T1** `RunnerConfig.claim_lease_seconds`/`claim_heartbeat_seconds` + `_claim_store()` plumbing (enables later tests). Default-preserving.
- [ ] **T2** `ClaimStore.mark_failed`/`is_failed` + fsync'd `failed.json`; done-precedence discovery helper.
- [ ] **T3** Candidate record dataclass + `plan_candidates(inputs, output_dir, config) -> Path`: abspath-dedup, stem-dedup, deterministic `created_at`, one atomic durable commit to `candidates/<workload_id>.jsonl`.
- [ ] **T4** `materialize_job(record)`: claim-tolerant, complete-intent match/skip, deterministic manifest, manifest-then-index ordering. Refactor manifest core out of `_create_job` (eager path stays).
- [ ] **T5** `_pages_from_manifest` (extract from `repair_index`, seed `old_pages={}`, adopt artifacts).
- [ ] **T6** Threaded heartbeat with injectable interval + strict signal→join→release; wire into the candidate handler.
- [ ] **T7** Candidate handler: skip done/failed → claim → heartbeat → (read-or-rebuild | materialize) → process(recover) → mark_done/mark_failed → shutdown. Per-candidate failure isolation.
- [ ] **T8** `work` drains `candidates/*.jsonl` then legacy `jobs/` scan; `--retry-failed`.
- [ ] **T9** CLI: `paperscale plan …`; `work --retry-failed`. Docs (`docs/cli.md`, `docs/state-layout.md`, `docs/concurrency-and-queuing.md`).
- [ ] **T10** Concurrency (real subprocess) + ordering + heartbeat-renewal tests; full suite green.

## ADR

- **Decision:** Add a lazy `plan`+`work` workflow: a one-write immutable candidate manifest plus on-demand job materialization during `work`, scoped to the existing eager `work()` concurrency guarantee.
- **Drivers:** (1) kill the multi-minute pre-OCR startup / first-page-fast; (2) preserve single-machine multi-process drain; (3) human-readable ids/outputs.
- **Alternatives considered:** **B** hash(abspath) ids over the raw input-list — rejected (unreadable ids/outputs, moved-file orphans, input-list becomes a load-bearing interface). **C** thread-pool the eager enqueue — rejected as *primary* (first page still waits for the whole batch; fsync storm only made concurrent) but retained as the cheaper fallback; committed fix #1 already reached ~2.2 min and C could plausibly reach ~30–40 s.
- **Why chosen:** A is the only option satisfying all three drivers; it's justified by first-page-fast UX + multi-process drain, **not** by raw startup seconds (which C could also improve).
- **Consequences:** new on-disk format (`candidates/<id>.jsonl`) and `plan` command; `work` gains a second discovery source; a new "claimed dir, no manifest" recovery window (contained by manifest-as-barrier + lease reclaim); **does not** strengthen the pre-existing claim-epoch fencing (explicitly out of scope) — lazy is no safer and no less safe than eager `work` today.
- **Follow-ups (out of scope, tracked):** claim-epoch fencing of page writes (would make exactly-once provable); transient-vs-terminal classification of materialize errors; optional `run --lazy` to route the eager front door through `plan`+`work`.
