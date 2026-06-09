# Ledger and Recovery v1 Notes

This worker slice implements the in-process ledger state machine used by the
local-first OCR runner. Filesystem persistence remains behind the `LedgerStore`
protocol owned by the state-store slice; this module preserves the ledger safety
contract independently of the backend.

## Safety properties covered

- A page attempt is reserved and persisted before provider I/O.
- Provider calls are marked `in_flight` before the request can be committed.
- `reserved` attempts with no provider start requeue to `pending` after lease
  expiry.
- `in_flight` attempts without a committed provider response become
  `ambiguous` after lease expiry.
- Ambiguous attempts are not retried unless the caller supplies an explicit
  duplicate-call policy.
- Stale worker epochs cannot heartbeat or commit after a higher-epoch takeover.

## Interface boundary

`paperscale.ledger.LedgerStore` is the storage boundary. Normal recovery reads
`active_records()` only, so production stores can satisfy status/resume/recovery
from compact indexes instead of scanning artifacts. The included
`InMemoryLedgerStore` is for tests and fake local flows; durable filesystem
writes belong in the state-store implementation.

## Operator behavior

Ambiguous work should be surfaced with page IDs and duplicate-call risk. The
ledger intentionally raises `AmbiguousAttemptError` by default to prevent silent
provider replays against non-idempotent endpoints. Operators resolve ambiguous
pages with `reconcile --supersede PAGE` (discard the uncertain attempt, requeue)
or `reconcile --accept PAGE` (adopt the page's already-written artifact); both
record the prior attempt as `superseded`.

## Resource governance and capacity

The real `DocumentOcrRunner` processes pages sequentially under a
`ResourceGovernor`, which enforces the global acquisition order
`scheduler -> render -> provider -> page_lease -> state_store`. Persistence is
governed at `state_store_lock` granularity; the state store owns its own
transient file descriptors.

Provider pressure is controlled by the capacity profile resolved from
`manifest.capacity` (`builtin_capacity_profile`). Transient provider transport
errors are retried with exponential backoff (`backoff_initial_seconds` doubling
up to `backoff_max_seconds`). After `circuit_breaker_failure_threshold`
consecutive failures the circuit opens: the run stops and any pages not yet
started stay `pending`, so a later `paperscale resume` continues the work.
Content failures (parse/verify rejections) are classified by the profile and are
not retried within a single run.
