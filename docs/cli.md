# paperscale CLI reference

All commands run as `paperscale <command>` (use `poetry run paperscale ...` in a
Poetry checkout). State lives under `--state-root` (default `.paperscale`); see
[state-layout.md](state-layout.md).

## Common options

| Option | Commands | Meaning |
| --- | --- | --- |
| `--state-root PATH` | all stateful commands | state directory (default `.paperscale`) |
| `--base-url URL` | `run`, `doctor` | OpenAI-compatible base URL; `/v1` is appended when omitted |
| `--model ID` | `run`, `doctor` | served model id (`run` defaults to the profile's model) |
| `--profile NAME` | `run`, `doctor` | OCR profile (default `generic_vlm_markdown`) |
| `--capacity NAME` | `run`, `doctor` | capacity profile (default `local-vllm-small`) |

## `run` — OCR a document

Renders each page, OCRs it through the provider, and assembles the Markdown.

```bash
paperscale run --input doc.pdf --output doc.md --base-url http://127.0.0.1:8009 --model mock-vlm --job-id doc1
```

| Option | Required | Meaning |
| --- | --- | --- |
| `--input PATH` | yes | PDF input |
| `--output PATH` | yes | Markdown output |
| `--base-url URL` | yes | provider base URL |
| `--job-id ID` | no | workload id (generated from the filename when omitted) |
| `--model`, `--profile`, `--capacity`, `--state-root` | no | see common options |
| `--allow-partial` | no | assemble succeeded pages even if some pages fail |

The runner processes pages sequentially under managed resource ordering and a
capacity-driven retry/circuit-breaker. A transient provider error is retried with
exponential backoff; after the capacity's failure threshold the circuit opens,
the run stops, and unfinished pages stay `pending` for `resume`.

**Exit code:** `0` when every page succeeds (or `--allow-partial` and at least
one page succeeds), `1` otherwise.

## `resume` — continue a stopped job

```bash
paperscale resume doc1
paperscale resume doc1 --retry-ambiguous --allow-partial
```

Reuses the original job's input, provider URL, model, and capacity from its
manifest, so no `--base-url` is needed. Recovers leases whose worker died, then
processes the remaining pages.

| Option | Meaning |
| --- | --- |
| `--retry-ambiguous` | retry `ambiguous` in-flight attempts (risks duplicate provider calls) |
| `--allow-partial` | assemble succeeded pages even if some fail |
| `--state-root` | see common options |

**Exit code:** same rule as `run`.

## `status` — show progress

```bash
paperscale status doc1
paperscale status doc1 --json
```

Reads the compact status index only (no artifact tree scan) and prints
succeeded / pending / failed / ambiguous counts. `--json` emits a machine-readable
summary. **Exit code:** `0`.

## `reconcile` — resolve ambiguous pages

An `ambiguous` page is one whose provider call was in flight when its lease
expired: the call may or may not have completed, so paperscale never retries it
silently. List them, or resolve one:

```bash
paperscale reconcile doc1                 # list ambiguous attempts + guidance
paperscale reconcile doc1 --json
paperscale reconcile doc1 --supersede 4   # discard the attempt, requeue page 4 as pending
paperscale reconcile doc1 --accept 4      # accept page 4's existing artifact as succeeded
```

`--supersede` and `--accept` are mutually exclusive. Both mark the prior attempt
`superseded` in the ledger; `--accept` additionally adopts the page's
already-written artifact and re-assembles the document if it is now complete.
**Exit code:** `0`.

## `fsck` — consistency check (read-only)

```bash
paperscale fsck doc1
```

Cross-checks the ledger, indexes, and artifacts without writing anything. Emits a
JSON report with an `issues` list. Issue codes: `missing_artifact`,
`orphan_artifact`, `missing_page`, `count_mismatch`, `ledger_mismatch`,
`fingerprint_mismatch`, `stale_lease`. **Exit code:** `0` when clean, `1` when any
issue is found.

## `repair-index` — rebuild compact indexes

```bash
paperscale repair-index doc1
```

Rebuilds the status/resume/reconcile indexes from the artifacts actually present
on disk (a page is `succeeded` only if its artifact exists). Use after manual
recovery or if an index is lost. **Exit code:** `0`.

## `doctor provider` — validate a provider

```bash
paperscale doctor provider --base-url http://127.0.0.1:8000 --model your-model
```

Checks that the server is reachable and lists the requested model in `/v1/models`,
and reports the resolved OCR and capacity profiles. Starts no OCR work.
**Exit code:** `0` when compatible, `1` otherwise.

## `assemble` — combine pre-OCR'd pages

Assembles a JSONL file of completed page artifacts into one document, independent
of the runner.

```bash
paperscale assemble --input pages.jsonl --output doc.md --title "My Document" --enforce-quality
```

Each JSONL line needs `document_id`, `page_number`, and `markdown`.
`--enforce-quality` rejects empty, mojibake, or repeated fragments;
`--allow-partial` marks the output partial; `--title` prepends an H1.
**Exit code:** `0`.
