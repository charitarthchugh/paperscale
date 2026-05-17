# Context Snapshot: Paperscale VLM OCR Implementation

## Task statement
Implement the approved Paperscale v1 local-first VLM OCR library + CLI plan using OMX team mode.

## Desired outcome
A greenfield Python package with tested contracts, atomic filesystem state store, ledger/recovery semantics, resource governor/scheduler, OpenAI-compatible/self-hosted provider profiles, model OCR profiles for LightOnOCR-2-1B/DeepSeek-OCR-2/GLM-OCR, deterministic quality verification, Markdown assembly, and CLI commands.

## Known facts/evidence
- Current repo is greenfield: pyproject.toml and poetry.lock exist; no src/ or tests/ package exists yet.
- Existing approved plan is `.omx/plans/vlm-ocr-consensus-plan.md` and the user pasted the current implementation handoff with non-negotiable invariants.
- Git status is an unborn main branch with no commits and untracked `.idea/`, `.omc/`, `poetry.lock`, `pyproject.toml`.
- Prior memory notes say this repo had Codemap setup verification only; no product implementation exists.

## Constraints
- Must use OMX team runtime for implementation coordination.
- Must proceed test-first: red tests before production slices and critic dispositions recorded.
- Must keep normal status/resume/reconcile paths index-only and repair/fsck as explicit scan paths.
- Must not add dependencies unless explicitly necessary; use existing OpenAI dependency and stdlib where possible.
- v1 public surface is document-to-Markdown OCR only.
- No real provider/network calls in tests; use fake clients/providers.

## Unknowns/open questions
- Exact public API ergonomics can be minimal as long as plan invariants and CLI commands are covered.
- External model docs may refine prompts later; v1 implementation should encode conservative profile-specific prompts/metadata and fingerprint sensitivity.

## Likely codebase touchpoints
- `src/paperscale/contracts.py`
- `src/paperscale/resources.py`
- `src/paperscale/state/fs_store.py`
- `src/paperscale/ledger.py`
- `src/paperscale/scheduler.py`
- `src/paperscale/providers/base.py`
- `src/paperscale/providers/openai_chat.py`
- `src/paperscale/providers/self_hosted.py`
- `src/paperscale/profiles/base.py`
- `src/paperscale/profiles/builtin.py`
- `src/paperscale/quality/verifier.py`
- `src/paperscale/assembly.py`
- `src/paperscale/cli.py`
- `tests/**`
- `.omx/plans/test-review-checklist.md`
