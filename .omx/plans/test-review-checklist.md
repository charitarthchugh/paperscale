# Paperscale VLM OCR v1 Test Review Checklist

Status: worker-1 initial critic disposition record  
Scope: tests/harness plus critic checklist for the approved VLM OCR v1 plan.

## Critic disposition summary

Disposition: approved-with-follow-up for the initial red-test backlog.

Rationale:
- Tests map directly to the non-negotiable invariants and acceptance tests in `docs/vlm-ocr-consensus-plan.md`.
- Tests use deterministic fakes and stdlib `unittest`; no real provider or network calls are required.
- Tests prefer observable ordering, state, resource, and fingerprint effects over CLI-only proxy success.
- Follow-up needed after other workers implement production slices: tighten any helper APIs that drift from implementation names while preserving the same invariant coverage.

## Review gates for every production slice

For each production PR/slice, record:

- Requirement trace: acceptance test IDs or invariant names.
- Red evidence: exact failing command and expected failure.
- Critic disposition: approved, approved-with-follow-up, or rejected-and-rewritten.
- Mock/proxy audit: why the test observes the invariant directly enough.
- Safety audit: no real provider calls, no hidden file/socket opens, no full-tree scan in normal paths.
- Green evidence: exact passing command after implementation.

## File coverage matrix

| Test file | Acceptance coverage | Current critic disposition | Follow-up focus |
| --- | --- | --- | --- |
| `tests/test_contracts.py` | 15, 16, 26 | approved-with-follow-up | Align record loaders and fingerprint API names if production chooses equivalent names. |
| `tests/state/test_fs_store_atomicity.py` | 1, 2, 12, 14, 15 | approved-with-follow-up | Add disk-full simulation once fs write hooks exist. |
| `tests/test_ledger.py` | 4, 5, 6, 7, 8, 21 | approved-with-follow-up | Add no-lock-during-provider-sleep timing probe after ledger runner exists. |
| `tests/test_resources.py` | 9, 11 | approved-with-follow-up | Add socket/subprocess factory probes with implementation. |
| `tests/test_scheduler.py` | 12, 13, 20 | approved-with-follow-up | Add retry budget/backoff integration with provider worker. |
| `tests/providers/test_openai_chat.py` | 19 | approved-with-follow-up | Keep fake client only; no SDK/network calls in tests. |
| `tests/providers/test_self_hosted.py` | 20, 22, 23, 24 | approved-with-follow-up | Expand overload taxonomy as provider code lands. |
| `tests/profiles/test_builtin_profiles.py` | 25, 26, 27, 28, 29, 30, 31 | approved-with-follow-up | Replace representative prompt assertions with fixture snapshots if desired. |
| `tests/quality/test_verifier.py` | 32, 33, 34 | approved-with-follow-up | Add truncation and length-anomaly cases when policy constants land. |
| `tests/test_assembly.py` | 17, 18 | approved-with-follow-up | Add manifest checksum checks after artifact schema exists. |
| `tests/test_cli.py` | 12, 14, 21, CLI exit criteria | approved-with-follow-up | Add subprocess-level smoke after console script is wired. |
| `tests/test_e2e_fake_job.py` | integration gate, 4, 9, 10, 17 | approved-with-follow-up | Update fake runner hook names if final orchestrator surface differs. |

## Non-negotiable rejection criteria

Reject or rewrite tests that:

- pass through real OpenAI, vLLM, HTTP, sockets, or filesystem scans not explicitly under test;
- verify only that a CLI exits zero without observing ledger/resource/state invariants;
- allow provider I/O before a durable reservation;
- hold `StateStore` mutation locks during provider sleep/I/O;
- make `status`, `resume`, `reconcile`, scheduler discovery, or progress reporting scan artifact trees;
- accept unknown future schema versions with best-effort downgrade or mutation;
- expose v1 public modes other than document-to-Markdown;
- make ambiguous in-flight attempts auto-retry by default.
