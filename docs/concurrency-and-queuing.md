# Concurrency and Queuing Design

Status: **implemented**. Describes the multi-worker request pipeline that replaced
the strictly-sequential `DocumentOcrRunner` loop. Builds on the durability model in
`recovery-retry-redesign.md` and `ledger-recovery.md`. Implementation:
`runner.py` (`_run_pool`, `_worker`, `_attempt_page`, `_IndexWriter`),
`async_pool.py:AdaptiveLimiter`, `providers/async_openai_chat.py`,
`providers/self_hosted.py:ProviderOverloadController` (AIMD), and
`state/claim_store.py` (O_EXCL claim tier). The single-machine multi-process tier is
driven by `paperscale enqueue` + `paperscale work`. Tests:
`tests/test_concurrency_pool.py`, `tests/providers/test_aimd.py`,
`tests/state/test_claim_store.py`.

Operational note (server-side hardening): on a 24 GB GPU, DeepSeek-OCR-2 OOMs the
engine ("death, not degradation") under the doc's defaults — launch vLLM with
`--gpu-memory-utilization ≈ 0.70` and a right-sized `--max-num-seqs`/
`--max-num-batched-tokens` to leave headroom for vision-tower activations. The
client layers (AIMD, circuit breaker) protect against over-send but cannot revive a
dead engine.

## Goal

Keep a local vLLM server's continuous-batching engine saturated — always offer
the server more work than it can run at once — without exhausting file
descriptors, sockets, or memory, and without weakening the per-page durability
contract.

## Baseline reality

The live runner is **strictly sequential**: `_process_pages` loops pages one at a
time and `_process_page` calls `provider.send()` synchronously. The
`ResourceGovernor` (`resources.py`), `Scheduler`/`LazyPageQueue` (`scheduler.py`),
and `ProviderOverloadController` (`providers/self_hosted.py`) are **unattached
scaffolding** — exercised by unit tests but never instantiated on the live
`run`/`resume` path. The capacity-profile limits are inert except in `doctor`.

## Concurrency model (olmOCR-shaped)

- **Single-process `asyncio` pool of full-lifecycle workers.** Each worker owns a
  page end to end: render → submit → await response → post-process → retry or
  finalize. There is **no** submit-thread / poll-thread split — that gives only
  one in-flight request and contends on a shared mutable queue. Full-lifecycle
  workers get automatic backpressure: a worker doesn't pull its next page until
  its current one resolves.
- **`asyncio`, not OS threads.** The work is I/O-bound (HTTP + fsync); an event
  loop holds many in-flight requests cheaply. `/v1/chat/completions` is
  synchronous request/response — workers `await` their own responses; there is no
  server to "poll".
- **Saturation is governed by outstanding-request count**, not local queue depth.
  vLLM does continuous batching server-side; our job is to keep enough requests
  outstanding. Surplus beyond `max_num_seqs` sits in vLLM's waiting queue (adds
  latency, not throughput).
- **Durability edge retained.** Per-page ledger + artifacts + adopt-then-requeue
  (see `recovery-retry-redesign.md`) is *finer* than olmOCR's work-item
  completion markers. Keep it. True multi-*node* execution is deferred to a coarse
  work-item claim tier later, mapping onto the existing
  `worker_id`/`epoch`/`lease_expires_at`/`heartbeat_at` ledger fields.

## Backpressure knobs (three, not one)

"Too many pages" decomposes into three independent limits. One overloaded knob
cannot express them: sized for the GPU it under-bounds memory; sized for memory it
starves the GPU.

1. **`max_in_flight_requests`** — process-wide `asyncio.Semaphore`, acquired
   *before* a request leaves for the server, released on response. This is the
   global "pages sent to the server" cap. **Default `4 × server max_num_seqs`,
   configurable.** Client concurrency governs vLLM's *waiting queue depth*, not
   its active batch — surplus beyond `max_num_seqs` queues cheaply in **host RAM**
   (raw image bytes), keeping the scheduler from starving between steps. The
   HTTP connection pool MUST be sized to match this number. The limit is **global
   across all document workers**, not per-document (per-document would let K
   documents open K×limit sockets and reopen the FD problem). The 4× default
   **assumes a dedicated server** (only vLLM running) with a correctly-sized
   `max_num_seqs` and `gpu_memory_utilization ≈ 0.85`; on a shared GPU or
   undersized headroom, lower it. See "Protecting the server" below.
2. **`max_open_documents`** — bounds concurrently open PDF handles under **lazy**
   rendering (one decoded page at a time per active document). An open PDF costs
   ~1 FD, so this is cheap. **Default 128, configurable.**
3. **`render_ahead`** — bounded prefetch queue of decoded-but-unsubmitted page
   images, in **pages**. This is the real memory bound (decoded bitmaps are
   ~15-20 MB each at the small profile). Cap ≈ `max_in_flight_requests`,
   independent of document count.

### FD and memory accounting

- FD pressure is dominated by **sockets** (= `max_in_flight_requests`, e.g. 512),
  not open PDFs (128 FDs is noise). The 512 default exceeds the common
  `ulimit -n 1024` once sockets + PDF handles + state writes + stdio are summed —
  **docs must instruct raising the FD limit** and note the `max-num-seqs`
  relationship.
- Memory is bounded by `render_ahead` (decoded pages), **not** by document count.
  Rendering stays **lazy** (file open, `render_page(n)` on demand, one image at a
  time) — never eager-decode-whole-document-then-close, which trades 1 FD of
  savings for multi-GB of RAM and scales with the largest document seen.

## Protecting the server (OOM avoidance)

Over-sending to a self-hosted VLM server can OOM it (olmOCR's "too many asyncio
workers" failure). The cliff is **death, not degradation** — once the engine
OOMs, no client retry policy recovers it; it needs a restart. So we stay under
the memory ceiling by construction and probe gently.

**Mechanism.** Two populations consume memory differently:
- *Active batch* (vLLM is prefilling/decoding) — bounded by `max_num_seqs` /
  `max_num_batched_tokens`; holds vision-tower activations in **VRAM**. This is
  the OOM cliff, owned **server-side**.
- *Waiting queue* (admitted, not scheduled) — holds raw image bytes in **host
  RAM**, not GPU activations. Cheap; this is what client concurrency controls.

So client concurrency cannot cause a GPU OOM *by itself once `max_num_seqs` is
sized to fit VRAM* — it only sets queue depth. OOM means the server config is
wrong, a shared GPU, or eager multimodal preprocessing in some vLLM versions.

**Four layers of protection:**

1. **Conservative-but-fed default** — `max_in_flight_requests = 4 × max_num_seqs`
   (above), so the GPU never starves while surplus waits cheaply in host RAM.
2. **Adaptive concurrency (AIMD)** — drive the semaphore size dynamically:
   additive-increase on sustained success/low latency up to the 4× ceiling,
   **multiplicative-decrease toward a floor of ~`max_num_seqs`** on any true
   overload signal (429/503/timeout/connection-reset). Wire the existing
   `ProviderOverloadController` to resize the semaphore, not just gate retry
   timing. AIMD floats *below* the default; its job is snap-back, not ramp-up.
   **Signal hygiene:** only genuine overload shrinks concurrency — a content
   `400` or parse/verify rejection must NEVER throttle (else a mojibake-heavy run
   self-throttles against a healthy server).
3. **Circuit breaker + timeouts** — `ProviderOverloadController.circuit_open`
   trips after N consecutive overloads: stop sending, let the server drain, then
   probe. Per-request timeouts feed the AIMD decrease.
4. **Server-side hardening (the real cliff-guard)** — operator config, surfaced
   by `doctor` where observable and documented next to the `max-num-seqs` note:
   `--gpu-memory-utilization ≈ 0.85` (headroom for multimodal activations),
   right-sized `--max-num-seqs`, bounded `--max-model-len` and request
   `max_tokens` (caps KV per sequence), `--limit-mm-per-prompt`,
   `--max-num-batched-tokens` (bounds prefill spikes). The client default
   *assumes* these are set; layers 1–3 survive the cases where they aren't.

## Durability under concurrency

- **Results are write-through.** Each succeeded page is written to its durable
  artifact on disk first (unchanged from today); an in-RAM copy is at most a
  *cache* to skip the re-read in `_assemble_if_ready`. **Never RAM-primary** — a
  RAM-only result is lost on crash and defeats the ledger/resume contract.
- **Single serialized state-writer for the compact indexes (chosen: Option A).**
  `_write_indexes` rewrites the entire status index with an `fsync` on every
  transition; under concurrency that both loses updates (workers clobbering each
  other's whole-file rewrites) and creates an O(N²) fsync storm. Resolution:
  - Workers **never write the indexes**. They push tiny events
    `(page_number, new_state, fingerprint, diagnostic?)` onto an `asyncio.Queue`.
  - **One writer task** owns the authoritative in-memory `pages`, folds each event
    in (O(1)), and regenerates **all three** indexes (`status`/`resume`/`reconcile`)
    together so they stay mutually consistent.
  - **Coalesced flush cadence:** flush on `max(64 events, 250 ms)`, whichever
    first, plus an **immediate flush on notable events** (terminal failure, settle,
    job completion). Bounds index staleness to ≤250 ms; collapses the fsync storm
    by ~100×.
  - **Index writes are atomic but NOT fsync'd.** The durable truth (artifact +
    per-attempt ledger) is fsync'd by workers per page; the indexes are a *derived,
    rebuildable rollup*, so they need atomicity (`os.replace`, no torn reads) but
    not durability. Losing the last index write on power-loss is harmless — resume
    rebuilds it. Sharp rule: **fsync the truth (artifact, ledger); atomic-but-
    unsynced the derived cache (indexes).**
  - `run()`/`resume()` do a **final synchronous flush** before returning, and the
    returned `JobStatus` comes from the writer's in-memory snapshot (never a stale
    read).
- **The compact index is now formally eventually-consistent.** This leans into the
  codebase's existing intent: the index is already called *compact*, already has
  `repair_index` to rebuild it from artifacts, and `fsck` to triangulate it. The
  per-page **artifact + per-attempt ledger** are the source of truth and are
  already per-file concurrency-safe.
- **Resume reconciles before processing.** Because the index may lag the truth
  after a crash, `resume` first runs a reconcile pass (`repair_index` semantics
  fused with adopt-then-requeue: artifact present + fingerprint match →
  `succeeded`) so processing sees real state — turning index staleness into a cheap
  reconcile, never lost or duplicated durable work.
- **Out-of-order completion is fine.** `_assemble_if_ready` sorts by page number
  and only assembles when complete (or `allow_partial`); pages may finish in any
  order.

## Resolved

- **State-writer mechanism** — Option A (single coalescing writer), see
  "Durability under concurrency".
- **Backpressure / retry-storm under load** — see "Protecting the server".

## Multi-process claim tier (single machine)

Scope: **always a single machine**, multiple `paperscale` processes pulling work.
No cross-machine / NFS / object-store backends — explicitly YAGNI.

- **Unit of claim: the document/job**, not the page. Each job is single-owner, so
  the entire intra-process design (single index writer, async pool, backpressure)
  runs unchanged within it. Horizontal scale = many documents across processes.
  A single gigantic document is handled by **sharding into sub-jobs at ingest**
  (page ranges, assembled at the end), never by multiple processes on one job.
- **Claim substrate: `O_EXCL` filesystem ClaimStore.** Atomic on a local fs, so
  claims are genuinely exclusive between processes. A document-level claim record
  `jobs/<job_id>/claim.json` carries `worker_id`, `epoch`, `lease_expires_at`,
  `heartbeat_at`. (A `ClaimStore` seam may exist, but only the `O_EXCL`
  implementation is needed — no pluggable multi-machine backends.)
- **Lease 60 s, heartbeat 20 s** (≈ lease/3). Owner renews via heartbeat; a silent
  owner past `lease_expires_at` is reclaimed by another process with a **higher
  epoch**.
- **Correctness does NOT rest on the lock.** A lease/epoch on plain storage gives
  efficient, mostly-exclusive claims + crash reclaim — **not** hard mutual
  exclusion: a paused (not dead) owner can wake after takeover and still write,
  and a dumb filesystem cannot fence that write (`os.replace` lands last-write-
  wins). Safety rests on two enforceable things instead:
  1. **D2 idempotency** — page-keyed, deterministic artifacts mean a zombie write
     is *redundant, not corrupting*. This is what makes the unenforceable fence
     tolerable, and it lets leases be short (false takeover = wasted work, never
     wrong work) without padding for worst-case GC pauses.
  2. **Epoch-aware reconcile (read-side fencing)** — every page attempt stamps the
     **owner's claim epoch**; the index writer / recovery ignores attempts from a
     superseded epoch. The fence moves from "prevent the write" (impossible) to
     "reject the stale write on read" (enforceable by the single higher-epoch
     owner). Composes with the eventually-consistent derived index: a brief
     two-owner overlap just yields a stale index the higher-epoch owner overwrites
     and resume reconciles.
- **Completion markers.** On true completion (all pages succeeded + assembled),
  write a **durable, fsync'd `done` marker** — truth, unlike the eventually-
  consistent index. The claim path checks the marker before attempting a claim, so
  finished documents are skipped without re-claiming. The marker joins
  artifact + ledger on the fsync-the-truth side of the split.
