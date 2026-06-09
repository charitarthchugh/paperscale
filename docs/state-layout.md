# State layout: `.paperscale/jobs/<job_id>`

Every job writes its durable state under the `--state-root` directory (default
`.paperscale`). The assembled Markdown goes to the `--output` path you choose and
lives outside this tree. All internal files are written atomically (temp file,
fsync, rename) and carry a `schema_version`; the runner refuses to read a version
it does not understand.

```
.paperscale/
└── jobs/
    └── <job_id>/
        ├── manifest.json            # immutable job description
        ├── indexes/
        │   ├── status.json          # per-page state + summary counts
        │   ├── resume.json          # pending / ambiguous / in-flight page lists
        │   └── reconcile.json       # ambiguous attempts + recommended actions
        ├── ledger/
        │   └── <attempt_id>.json    # one file per page attempt (latest state)
        └── artifacts/
            └── pages/
                └── <page_number>.json   # committed Markdown for a succeeded page
```

## `manifest.json`

Written once when the job is created and never mutated. Records the input and
output paths, `document_id`, `page_count`, and the `profile`, `base_url`,
`model`, and `capacity` the job runs with. `resume` reads these so it continues
with the original settings.

## `indexes/`

Compact indexes let `status`, `resume`, and `reconcile` answer without scanning
the artifact tree.

- **`status.json`** — the authority for per-page state. Holds a `pages` map
  (`"<n>" -> {state, epoch, attempt_id, fingerprint, ...}`) plus summary counts
  (`succeeded`, `pending`, `failed_retryable`, `failed_terminal`, `ambiguous`,
  `in_flight`). Page states: `pending`, `reserved`, `in_flight`, `succeeded`,
  `failed_retryable`, `failed_terminal`, `ambiguous`, `superseded`.
- **`resume.json`** — pages grouped into `pending_pages`, `ambiguous_pages`, and
  `in_flight_pages` for fast resume planning.
- **`reconcile.json`** — `ambiguous_attempts`, each with its `page_number`,
  `attempt_id`, `duplicate_call_risk`, and `recommended_actions`
  (`["supersede", "accept"]`).

## `ledger/`

One file per page attempt, named by `attempt_id`. The runner reserves and
persists an attempt **before** any provider I/O, marks it `in_flight` before the
request can commit, and writes the terminal state afterward. This ordering is the
crash-safety contract: a `reserved` attempt with no provider start requeues to
`pending` after its lease expires, while an `in_flight` attempt with no committed
result becomes `ambiguous`. Each attempt carries an `epoch`; a higher-epoch
takeover invalidates stale workers. See [ledger-recovery.md](ledger-recovery.md).

## `artifacts/pages/`

One file per succeeded page, named by page number. Holds the validated
`markdown`, the request `fingerprint` and `image_hash`, provider metadata, and
verifier metadata. A page counts as `succeeded` only when its artifact exists;
`repair-index` and `fsck` rely on this to triangulate index against disk.
