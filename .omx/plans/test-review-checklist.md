# Test Review Checklist

## Worker 3: Ledger/recovery

- Test files: `tests/test_ledger.py`, `tests/test_recovery.py`
- Requirements: acceptance tests 4, 5, 6, 7, 8 from `docs/vlm-ocr-consensus-plan.md`
- Red evidence: `PYTHONPATH=src python -m unittest tests.test_ledger tests.test_recovery` failed before production code with `ModuleNotFoundError: No module named 'paperscale'`.
- Critic disposition: approved-with-follow-up.
- Critic rationale: tests directly observe ledger transition order, lease-expiry recovery, ambiguous duplicate-call prevention, and stale epoch rejection. They use fake clocks and an in-memory compact store instead of real providers, so they prove the ledger invariant without network I/O.
- Follow-up: once the filesystem `StateStore` lands, add integration tests proving the same transitions persist through atomic writes and compact indexes on disk.
