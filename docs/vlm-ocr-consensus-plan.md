# Paperscale VLM OCR Library + CLI Consensus Plan

Status: Approved by Ralplan consensus  
Date: 2026-05-17  
Scope: Architecture and implementation handoff plan for a local-first VLM OCR workload manager at large PDF scale.

## ADR: Local-first replayable OCR runner with enforced safety invariants

### Decision

Build Paperscale v1 as a local-first Python library and CLI for large PDF/VLM OCR workloads using:

- a `StateStore` boundary with a filesystem journal as the first backend;
- compact ledger/index files for reservation, status, resume, and reconcile;
- a bounded lazy local queue;
- a `ResourceGovernor` that owns file, socket, provider, render, temp, and memory budgets;
- first-class support for vLLM and other OpenAI-compatible self-hosted inference servers;
- provider capacity profiles for resource-constrained inference hardware;
- model-specific OCR profiles for prompts, decoding settings, parsers, validators, and quality policy;
- first-class v1 OCR profiles for LightOnOCR-2-1B, DeepSeek-OCR-2, and GLM-OCR, scoped to document-to-Markdown conversion only;
- OpenAI-compatible chat as the first provider adapter behind a provider-neutral interface;
- separate page OCR and document Markdown assembly phases;
- a future quality-verification track for incoherent-output detection using heuristics and optional verifier SLMs.

### Drivers

1. Process very large PDF/page workloads without upfront 10M+ request materialization.
2. Avoid Linux fd/socket exhaustion and retry storms.
3. Make crash recovery, resume, and duplicate-call risks testable.
4. Treat self-hosted vLLM/OpenAI-compatible servers as first-class, especially when running on constrained GPUs.
5. Support multiple VLMs without pretending one prompt/settings/parser works for all models.
6. Make LightOnOCR-2-1B, DeepSeek-OCR-2, and GLM-OCR the first supported model targets.
7. Limit v1 behavior to document-to-Markdown conversion; free OCR, QA, extraction, and other task modes are out of scope.
8. Keep v1 deployable locally before distributed stores or queues.

### Consequences

- v1 is single-host and local-first, not distributed-safe.
- v1 supports self-hosted inference endpoints directly but does not manage GPU server lifecycle.
- Strong filesystem durability discipline is required.
- Full-tree scans are forbidden in normal status/resume/reconcile paths.
- Recovery complexity is accepted because provider calls may be expensive and non-idempotent.

## Non-negotiable invariants

### Atomic and replayable writes

Every durable state transition must be replayable.

Required write protocol:

1. Write to a temp file in the same directory/filesystem as the target.
2. Flush and `fsync` the temp file.
3. Publish with atomic `os.replace`.
4. `fsync` the parent directory where supported.
5. Never treat temp files as committed state.
6. On weak filesystems, require `--allow-weak-fs` or a stronger backend.

### Ledger before provider I/O

A worker must reserve a page attempt in the ledger before provider/network I/O.

Allowed sequence:

1. Acquire resources in fixed order.
2. Reserve ledger attempt.
3. Persist reservation.
4. Release `StateStore` lock.
5. Perform provider call.
6. Reacquire state lock only for commit/failure transition.

Forbidden:

- provider call before durable reservation;
- holding `StateStore` locks during provider I/O;
- writing output without a matching committed ledger transition.

### Fixed resource acquisition order

All workers acquire resources in one global order:

1. cancellation token;
2. scheduler slot;
3. PDF/render slot;
4. file descriptor token;
5. provider concurrency token;
6. page/document lease;
7. `StateStore` mutation lock.

Release in reverse order. No files, sockets, subprocesses, HTTP clients, or provider streams may be opened outside `ResourceGovernor`-managed paths.

### Compact indexes only for normal operations

These paths must use compact indexes/ledger state only:

- `paperscale status`
- `paperscale resume`
- `paperscale reconcile`
- scheduler discovery
- progress reporting

Full-tree scans are repair-only, for commands such as:

- `paperscale repair-index`
- `paperscale fsck`
- `paperscale rebuild-ledger-index`

### Unknown schemas fail closed

Every persisted object carries a schema version. Unknown future versions fail closed with:

- no mutation;
- no best-effort downgrade;
- no partial resume;
- actionable diagnostics.

### Provider capacity is a scheduler input

Self-hosted inference servers must be represented by explicit capacity profiles, not hidden CLI flags. The scheduler and `ResourceGovernor` must use provider limits when deciding queue fill, in-flight calls, timeouts, retry backoff, and image-render size.

A vLLM or other self-hosted profile must include:

- endpoint and health-check URL;
- served model name;
- max provider sockets and in-flight requests;
- provider-side queue tolerance;
- timeout and backoff policy;
- image-size/rendering hints for constrained VRAM;
- circuit-breaker thresholds for overload or slow responses.

### Model OCR behavior is profile-driven

Each model may require a different prompt, output contract, render size, parser, decoding settings, and retry/quality policy. The core must not hard-code prompts in providers or assume all VLMs follow the same instructions.

For v1, every `ModelOcrProfile` must implement the same product task: **convert one document page image into a Markdown fragment suitable for ordered document-level Markdown assembly**. Profiles may vary prompts and settings, but they must not expose alternate user-facing modes such as free OCR, visual QA, key-value extraction, layout-only extraction, or arbitrary prompting.

A `ModelOcrProfile` must define:

- prompt template and prompt/profile version;
- expected output format, such as structured Markdown, JSON, or YAML frontmatter;
- parser and normalizer;
- validation policy and retry classification hooks;
- decoding defaults, including temperature, top-p, max tokens, stop sequences, and guided decoding where supported;
- image/render preferences, including target longest side and format constraints;
- quality policy, including deterministic incoherence heuristics and optional verifier SLM behavior.

V1 first-class model profiles:

- `lighton_ocr_2_1b`: targets `lightonai/LightOnOCR-2-1B`; optimize for clean naturally ordered text/Markdown, fast throughput, and LightOn-specific rendering/preprocessing guidance.
- `deepseek_ocr_2`: targets `deepseek-ai/DeepSeek-OCR-2`; support its document-to-Markdown prompt style, dynamic-resolution/crop-mode assumptions, and no-repeat/repetition-oriented quality checks. Free-OCR mode is explicitly out of scope for v1.
- `glm_ocr`: targets `zai-org/GLM-OCR`; support GLM-OCR's document parsing behavior, vLLM/SGLang/Ollama deployment paths, and Markdown plus JSON/layout output where available.

These profiles are separate from provider/server profiles. For example, `glm_ocr`  a vLLM endpoint or SGLang, while preserving the same OCR behavior contract where possible.

Runtime flow:

```text
PageTask -> Renderer -> ModelOcrProfile.build_request() -> Provider.send() -> ModelOcrProfile.parse_and_validate() -> Quality verifier -> Ledger/result commit
```

Request fingerprints must include the `ModelOcrProfile` name, version, prompt hash, parser/schema version, decoding options, render options, provider, model, and image hash. Changing a prompt/settings/parser must create a distinct request key so stale results are not reused.

## Ledger and crash recovery model

### Ledger states

Each page attempt has one durable state:

- `pending`
- `reserved`
- `in_flight`
- `succeeded`
- `failed_retryable`
- `failed_terminal`
- `ambiguous`
- `superseded`

### Lease fields

Each reservation/in-flight attempt records:

- `attempt_id`
- `page_id`
- `provider_request_fingerprint`
- `worker_id`
- `epoch`
- `lease_expires_at`
- `heartbeat_at`
- `provider_call_started_at`
- `provider_response_committed_at`
- `result_pointer`

### Crash handling

On restart:

- `reserved` without provider start may requeue after lease expiry;
- `in_flight` with no committed result becomes `ambiguous` after lease expiry;
- `ambiguous` is not automatically retried unless policy allows duplicate provider calls;
- CLI surfaces ambiguous work with count, page sample, recommended action, and duplicate-call risk.

### Duplicate-call policy

Default: prevent silent duplicate provider calls.

Retrying ambiguous work requires one of:

- provider idempotency key support and matching request fingerprint;
- explicit `--retry-ambiguous`;
- manual reconciliation marking the prior attempt failed or superseded.

Provider adapters should pass idempotency keys when supported, but the core must not assume provider idempotency.

## Architecture

### Core modules

- `contracts.py`: schemas, IDs, versioned records, status enums.
- `resources.py`: `ResourceGovernor`, fd/socket/provider/render tokens, acquisition-order enforcement.
- `state/fs_store.py`: filesystem journal, atomic writes, compact indexes, schema checks.
- `ledger.py`: reservation, lease, heartbeat, recovery, ambiguity handling.
- `scheduler.py`: bounded lazy queue and status/resume from compact indexes.
- `providers/base.py`: provider-neutral OCR request/response interface.
- `providers/openai_chat.py`: first provider adapter using OpenAI-compatible chat.
- `providers/self_hosted.py`: vLLM/self-hosted OpenAI-compatible provider profiles, health checks, and constrained-hardware defaults.
- `profiles/base.py`: `ModelOcrProfile` interface for model-specific prompts, decoding settings, parser, validator, and quality policy.
- `profiles/builtin.py`: built-in OCR profiles such as `lighton_ocr_2_1b`, `deepseek_ocr_2`, `glm_ocr`, `generic_vlm_markdown`, and `strict_json_ocr`.
- `rendering.py`: PDF page rendering separate from provider calls.
- `quality/verifier.py`: planned incoherence detection using deterministic heuristics in v1 and optional verifier SLMs later.
- `assembly.py`: document/page Markdown assembly independent from page OCR.
- `cli.py`: `run`, `status`, `resume`, `reconcile`, `fsck`, `repair-index`.

### Separation rule

Page OCR produces durable page artifacts. Markdown/document assembly consumes completed artifacts later. Assembly failure must not invalidate successful page OCR.


## Provider, capacity, and model profile configuration

Add first-class provider configuration records:

- `InferenceServerProfile`: endpoint, model, request format, health-check URL, auth mode, timeout, retry policy, and provider-specific overload signals.
- `ProviderCapacityProfile`: max in-flight requests, max provider-side queued requests, latency budget, image-size/render hints, and constrained-hardware defaults.
- `ModelOcrProfile`: model-specific prompt, output format, parser, normalizer, validation policy, decoding defaults, render preferences, and quality/verifier policy.
- `SelfHostedOpenAICompatibleProvider`: first concrete self-hosted adapter for vLLM and compatible servers.

Recommended built-in provider/capacity profiles:

- `local-vllm-small`: conservative concurrency and image size for constrained GPUs.
- `local-vllm-large`: higher concurrency for dedicated local inference hardware.
- `remote-openai-compatible`: network-aware defaults for hosted compatible endpoints.

Recommended built-in model OCR profiles:

- `lighton_ocr_2_1b`: first-class LightOnOCR-2-1B profile for high-throughput clean text/Markdown extraction.
- `deepseek_ocr_2`: first-class DeepSeek-OCR-2 profile for document-to-Markdown prompts, dynamic-resolution/crop assumptions, and repetition-sensitive validation.
- `glm_ocr`: first-class GLM-OCR profile for GLM's OCR pipeline, self-hosted vLLM/SGLang/Ollama-compatible deployments, and Markdown plus layout/JSON metadata where exposed.
- `generic_vlm_markdown`: lowest-common-denominator document-to-Markdown OCR prompt for unknown VLMs.
- `strict_json_ocr`: strict JSON output contract for providers/models with reliable structured output support.

Example config shape:

```toml
[provider]
kind = "self-hosted-openai-compatible"
endpoint = "http://localhost:8000/v1"
model = "Qwen2.5-VL-7B-Instruct"

[capacity]
profile = "local-vllm-small"
max_concurrency = 2
max_provider_queue = 4

[ocr_profile]
name = "qwen2_5_vl_markdown"
prompt_version = "v1"
output_format = "structured_markdown"
temperature = 0.0
max_tokens = 4096
render_target_longest_side = 1280
```

`paperscale doctor provider` must validate endpoint reachability, model availability, auth, observed latency, configured timeout, and effective concurrency before a long run. It should also report the selected OCR profile and whether that profile is compatible with the provider request format.



## V1 product scope: document to Markdown only

Paperscale v1 supports exactly one user-facing OCR task: converting PDFs/document page images into ordered Markdown documents.

Supported v1 behavior:

- one page image in, Markdown page fragment out;
- document-level assembly of page fragments into final Markdown;
- model-specific prompts/settings only when they serve document-to-Markdown conversion;
- optional metadata needed to validate, retry, or assemble Markdown.

Out of scope for v1:

- free-form OCR modes that return unstructured text without document-to-Markdown assembly semantics;
- visual question answering;
- key-value extraction as a primary API;
- layout-only extraction as a primary API;
- arbitrary user prompts;
- multiple output formats beyond page artifacts and final Markdown, except metadata needed for validation/debugging.

Future versions may add task profiles beyond document-to-Markdown, but v1 `ModelOcrProfile` implementations must not expose those modes through the public CLI/library surface.

## V1 model target requirements

Paperscale v1 treats these as first-class model targets, not examples:

### LightOnOCR-2-1B

- Built-in profile name: `lighton_ocr_2_1b`.
- Default model ID: `lightonai/LightOnOCR-2-1B`.
- Support Transformers and vLLM/OpenAI-compatible server routes where available.
- Tune for high throughput, naturally ordered text, tables/forms/math, and LightOn rendering/preprocessing guidance.
- Persist LightOn profile version and prompt hash in request fingerprints.

### DeepSeek-OCR-2

- Built-in profile name: `deepseek_ocr_2`.
- Default model ID: `deepseek-ai/DeepSeek-OCR-2`.
- Support only its document-to-Markdown prompt style for v1.
- Track dynamic-resolution/crop-mode/render settings in request fingerprints.
- Add repetition and malformed-output heuristics suited to DeepSeek's recommended no-repeat/quality controls.

### GLM-OCR

- Built-in profile name: `glm_ocr`.
- Default model ID: `zai-org/GLM-OCR`.
- Support hosted API, SDK server, vLLM, SGLang, and Ollama-compatible deployment routes as provider variants when practical.
- Preserve Markdown plus JSON/layout details where available.
- Include GLM deployment/capacity settings, such as context length, memory fraction, speculative decoding, or server queue constraints, in profile/capacity fingerprints.

Each target must have profile-specific prompt fixtures, fake provider responses, parser/validator tests, and documented capacity defaults before being considered supported.

## Scale risks and mitigations

### Millions of small files / directory fanout

Mitigate with hash sharding, files-per-directory caps, compact hot-path indexes, and repair-only scans.

### Index corruption

Mitigate with checksummed index segments, fail-closed behavior, and explicit repair commands that rebuild from journal/artifacts and report confidence.

### Retry storms

Mitigate with global retry budgets, exponential backoff with jitter, provider circuit breakers, retryable/terminal error taxonomy, and no automatic retry for ambiguous in-flight attempts.

### Disk-full/temp exhaustion

Mitigate with free-space preflight, colocated temp files, old-state-preserving atomic writes, orphan temp cleanup, and local retryable classification only after operator intervention.

### Partial Markdown

Mitigate with assembly manifests, explicit partial markers, and a default requirement that all required pages succeed unless `--allow-partial` is set.

### Resume latency

Mitigate with compact scheduler/ledger indexes, no default tree scans, periodic checkpoints, and bounded recovery batches.

## Alternatives considered

### Filesystem journal vs SQLite/Postgres/object store

Filesystem journal is chosen for v1 because it is local-first, transparent, dependency-light, and inspectable. SQLite remains a serious fallback if ledger/index complexity grows. Postgres/object-store backends are deferred until distributed coordination is required.

### Local bounded queue vs distributed queue

Local bounded lazy queue is chosen for v1 to avoid materializing huge workloads and to control fd/provider pressure. Distributed queues require stronger cross-worker leasing and backend consistency guarantees and are deferred.

### OpenAI-compatible chat first vs provider lock-in

OpenAI-compatible chat is first because many VLM endpoints expose that shape, including vLLM and many self-hosted gateways. Core provider contracts must not expose OpenAI-specific types. Model-specific prompts and settings belong in `ModelOcrProfile`, not in the provider adapter. Future adapters can support local vLLM-specific controls, Anthropic-style messages, batch APIs, or custom HTTP.

### Self-hosted vLLM support vs cloud-only provider support

Self-hosted vLLM and OpenAI-compatible servers are first-class because the target workload often depends on local or dedicated inference hardware. Cloud-only support is rejected for v1 because it would hide the most important operational constraints: constrained VRAM, server-side queue depth, slow responses, health checks, and overload recovery.

### Model OCR profiles vs one universal prompt

`ModelOcrProfile` is chosen because each VLM may need different prompts, decoding parameters, image sizing, structured-output strategy, and validators for optimal OCR. A universal prompt is rejected for v1 because it would either underperform on capable models or over-constrain weaker/open self-hosted models. Provider adapters should transport requests; profiles should define model behavior.

### Assembly separate from page OCR

Separate assembly preserves expensive page OCR results and allows reassembly without provider calls. A monolithic “OCR whole document to Markdown” pipeline is rejected for v1 because it couples retry, formatting, and expensive inference.


### Document-to-Markdown only vs multi-task OCR toolkit

Document-to-Markdown is the only v1 product mode because it keeps prompts, validators, retries, page artifacts, and assembly coherent across models. Free OCR and other model-specific modes are deferred so v1 does not become an arbitrary VLM task runner.

## Test-driven execution contract

Implementation must proceed test-first. Each production slice starts with a failing test file or failing test case that traces directly to one or more numbered acceptance tests or non-negotiable invariants in this plan. Production code may be written only after that red state is captured.

Required TDD loop for every phase:

1. **Trace**: map the target requirement to a concrete acceptance-test number, invariant, or phase exit criterion.
2. **Red**: add or update the smallest test file that proves the missing behavior and verify it fails for the expected reason.
3. **Critic review**: a critic agent reviews the test file before implementation, checking direct requirement alignment, missing edge cases, negative paths, crash/recovery coverage, and whether the test could pass through a proxy or mock that does not prove the invariant.
4. **Green**: implement the minimal production code needed to pass the reviewed test without weakening earlier invariants.
5. **Refactor**: simplify only after the targeted test and the relevant existing suite pass.
6. **Evidence**: record the test command, expected red failure, green result, and critic disposition in the phase notes or PR/commit body.

Critic review is a gate, not a courtesy. If the critic finds that a test is indirect, under-specified, over-mocked, or not clearly tied to the plan, rewrite the test before production implementation. Test files that encode safety invariants must prefer fake crash hooks, fake providers, fake clocks, and observable resource-governor probes over broad end-to-end assertions alone.

### Test ownership and review matrix

| Test area | Expected test files | Requirement coverage | Critic review focus |
| --- | --- | --- | --- |
| Contracts/schema | `tests/test_contracts.py`, schema fixture tests | schema versions, IDs, state enums, fail-closed behavior | unknown-version mutation prevention and explicit migration behavior |
| Atomic filesystem store | `tests/state/test_fs_store_atomicity.py` | acceptance tests 1-3, 12-16 | temp-file crash points, parent fsync behavior, repair-only scan boundaries |
| Ledger/recovery | `tests/test_ledger.py`, `tests/test_recovery.py` | acceptance tests 4-8, 21 | reservation-before-provider proof, lease expiry, stale epoch rejection, ambiguous duplicate-call risk |
| Resource governor/scheduler | `tests/test_resources.py`, `tests/test_scheduler.py` | acceptance tests 9-14, 20, 24 | acquisition-order enforcement, no unauthorized files/sockets, compact-index-only status/resume |
| Providers/self-hosted capacity | `tests/providers/test_openai_chat.py`, `tests/providers/test_self_hosted.py` | acceptance tests 19, 20, 22-24 | provider-neutral contract, vLLM `/models` health checks, overload/backoff and bounded queues |
| Model OCR profiles | `tests/profiles/test_builtin_profiles.py`, prompt fixtures | acceptance tests 25-31 | profile-specific prompts/parsers/validators, no free-OCR public mode, fingerprint sensitivity |
| Quality/assembly/CLI | `tests/quality/test_verifier.py`, `tests/test_assembly.py`, `tests/test_cli.py` | acceptance tests 17-18, 32-34 and CLI exit criteria | incoherence classification before commit, partial markers, assembly independence, operator diagnostics |
| End-to-end fake job | `tests/test_e2e_fake_job.py` | phase integration and verification gate | proves full red-green path without real provider calls while preserving safety ordering |

## Phased execution plan

### Phase 0: Contracts and safety test harness

Deliver:

- versioned schema records;
- state enums;
- `ModelOcrProfile` protocol and fake profile;
- fake provider;
- fake filesystem/crash hooks;
- `ResourceGovernor` test doubles.

Exit criteria: invariant tests are written first, reviewed by the critic for direct traceability to this plan, and can fail before production store/provider code exists.

### Phase 1: Atomic filesystem `StateStore`

Deliver:

- same-filesystem temp-write protocol;
- `fsync` + `os.replace`;
- directory fsync where supported;
- schema-version fail-closed behavior;
- repair-only scan command skeleton.

Exit criteria: atomic write and schema tests are added red first, critic-approved, then pass without relaxing fail-closed behavior.

### Phase 2: Ledger, leases, recovery

Deliver:

- reservation-before-provider API;
- heartbeat/lease/epoch model;
- stale commit rejection;
- ambiguous state surfacing;
- compact ledger index.

Exit criteria: crash/recovery and duplicate-call prevention tests are added red first, critic-approved, then pass with explicit ambiguous-attempt coverage.

### Phase 3: `ResourceGovernor` and scheduler

Deliver:

- fixed acquisition-order enforcement;
- fd/provider/render tokens;
- bounded lazy queue;
- status/resume from compact indexes only.

Exit criteria:

- no unauthorized files/sockets test passes;
- status/resume do not scan artifact tree.

All Phase 3 tests must be red first and critic-approved for acquisition-order, unauthorized-resource, and compact-index coverage before scheduler/resource production code is changed.

### Phase 4: Provider adapter, self-hosted profiles, and page OCR

Deliver:

- provider-neutral interface;
- `ModelOcrProfile` interface and built-in `lighton_ocr_2_1b`, `deepseek_ocr_2`, and `glm_ocr` profiles;
- OpenAI-compatible chat adapter;
- `InferenceServerProfile` and `ProviderCapacityProfile`;
- vLLM/self-hosted health checks and constrained-hardware defaults;
- retry taxonomy/backoff/circuit breaker;
- page-level OCR artifacts.

Exit criteria: fake provider integration, LightOnOCR-2-1B/DeepSeek-OCR-2/GLM-OCR profile request/parse/validate tests, vLLM-style self-hosted profile tests, health-check tests, and retry-storm tests are added red first, critic-approved for provider-neutrality and profile-specific behavior, then pass.

### Phase 5: Quality verification, assembly, and CLI

Deliver:

- deterministic incoherence heuristics for empty output, repeated n-grams, refusal boilerplate, malformed schema/frontmatter, truncation indicators, and length anomalies;
- planned extension point for an optional verifier SLM;
- separate Markdown/document assembly;
- partial-output marking;
- CLI commands: `run`, `status`, `resume`, `reconcile`, `fsck`, `repair-index`, and `doctor provider`;
- operator-facing ambiguous-state and incoherent-output reporting.

Exit criteria: quality-heuristic tests, partial assembly tests, provider doctor tests, and end-to-end fake job tests are added red first, critic-approved for operator-facing coverage and no-real-provider determinism, then pass.

## Acceptance tests

These acceptance tests are the mandatory red-test backlog. Each numbered item must map to at least one named test file, and each test file must receive critic review before the production implementation it gates. The reviewer must reject tests that only prove a proxy signal, such as a CLI returning success without observing ledger/resource/state invariants.

### Atomic filesystem writes

1. Crash after temp write before rename leaves target old/missing and temp ignored.
2. Crash after rename before directory fsync recovers valid target or reports weak durability/repair-needed.
3. Disk-full during result write leaves no committed partial result; ledger remains retryable or explicit local failure.

### Ledger-before-provider

4. Fake provider/ledger asserts reservation is durably committed before provider call.
5. Crash after reservation before provider call requeues after lease expiry without ambiguity.
6. Crash during provider call before commit becomes `ambiguous`, not silently retried.

### Lease, heartbeat, epoch

7. Stale worker heartbeat expires; new worker acquires only with higher epoch.
8. Stale worker commit is rejected.

### Resource governance

9. All file opens/provider sockets go through `ResourceGovernor`.
10. No `StateStore` lock is held during fake provider sleep.
11. Acquisition-order violation raises in tests.

### Compact indexes

12. `status` on a large fake job reads only index files, not artifact tree.
13. `resume` with missing/corrupt index fails closed.
14. `repair-index` performs full-tree scan only when explicitly invoked.

### Schema safety

15. Unknown future schema version causes normal commands to fail closed without mutation.
16. Migration/repair test handles known older schemas explicitly.

### Provider and assembly separation

17. Page OCR success remains valid if Markdown assembly fails.
18. Partial assembly is marked partial and cannot masquerade as complete.
19. OpenAI-compatible chat adapter works through provider-neutral contract using fake HTTP/client.

### Retry storm controls

20. Repeated 429/5xx triggers backoff, retry budget, circuit breaker, and no unbounded queue growth.
21. Ambiguous attempts are reported and not auto-retried by default.


### Self-hosted inference profiles

22. vLLM-style `/models` health check verifies endpoint reachability and served model compatibility.
23. `local-vllm-small` profile applies conservative defaults for concurrency, provider queue depth, timeout, and image size.
24. Simulated provider overload triggers backoff/circuit breaker without increasing client queue beyond `queue_size`.


### Model OCR profiles

25. Each `ModelOcrProfile` builds provider-neutral page requests from the same `PageTask` without provider-specific leakage into scheduler code.
26. Changing prompt version, parser/schema version, decoding options, or render options changes the request fingerprint and does not reuse stale results.
27. Built-in profiles parse/validate representative fake outputs and classify malformed output according to their retry policy.

28. `lighton_ocr_2_1b`, `deepseek_ocr_2`, and `glm_ocr` each have prompt fixtures, request-building tests, parser tests, validator tests, and request-fingerprint tests.
29. DeepSeek-OCR-2 profile tests cover document-to-Markdown mode plus dynamic-resolution/crop settings in fingerprints; free-OCR mode is not included in v1 tests.
30. GLM-OCR profile tests cover Markdown output and optional JSON/layout metadata where available.
31. LightOnOCR-2-1B profile tests cover clean text/Markdown extraction and profile-specific render/preprocessing settings.

### Incoherence detection

32. Empty/refusal/repeated-ngram/malformed-frontmatter outputs are classified as retryable or terminal according to policy before page commit.
33. Verifier metadata is persisted with page artifacts and assembly can surface pages accepted with warnings.
34. Optional verifier SLM remains an extension point; v1 deterministic heuristics work without another model dependency.

## Implementation cautions

- Add monotonic sequence numbers and checksums to journal/index records.
- Define one authoritative source for repair.
- Make `provider_request_fingerprint` include provider, model, `ModelOcrProfile` name/version, prompt hash, parser/schema version, render params, page image hash, and decoding options.
- Include `InferenceServerProfile`, `ProviderCapacityProfile`, and `ModelOcrProfile` fingerprints in request keys so self-hosted tuning or prompt/settings changes do not incorrectly reuse stale results.
- Keep vLLM/self-hosted overload handling separate from generic transport failures so constrained hardware can back off gracefully instead of producing retry storms.
- Keep verifier SLM optional and isolated behind `quality/verifier.py`; deterministic heuristics must remain the no-extra-model baseline.
- Dependency-inject file/socket/subprocess/HTTP factories so `ResourceGovernor` can enforce budgets.
- Avoid holding scarce provider permits while waiting on state locks.
- Document Linux-first fsync behavior and graceful degradation elsewhere.
- Keep SQLite as escalation path if filesystem coordination starts reproducing database complexity.


## External model references for v1 profile design

- LightOnOCR-2-1B: `lightonai/LightOnOCR-2-1B` model card and usage notes.
- DeepSeek-OCR-2: `deepseek-ai/DeepSeek-OCR-2` repository, vLLM/Transformers inference notes, main prompts, and dynamic-resolution support modes.
- GLM-OCR: `zai-org/GLM-OCR` repository, SDK/API notes, and self-hosting guidance for vLLM, SGLang, Ollama, and SDK server/client deployments.

## Recommended execution staffing

Use `$team` for implementation because ownership boundaries are clean. Run a TDD blocker before production slices: the Test Engineer creates the red tests and the Critic reviews each test file for plan alignment and coverage before executors implement against them.

1. **Test Engineer**: owns Phase 0 invariant tests, crash/fake-provider harness, and the red-test backlog for each later phase. Must not implement production code in the same pass.
2. **Critic Agent**: reviews every test file before implementation and after significant test edits. The critic must verify direct traceability to this plan, negative/crash/recovery paths, maximal practical coverage, and absence of over-mocked proxy assertions.
3. **StateStore Executor**: atomic filesystem writes, schema checks, repair-only scans, implemented only after critic-approved red tests exist.
4. **Ledger Executor**: reservation, lease, heartbeat, epoch, recovery, implemented only after critic-approved red tests exist.
5. **Resource/Scheduler Executor**: `ResourceGovernor`, acquisition order, bounded queue, compact-index status/resume, implemented only after critic-approved red tests exist.
6. **Provider/Profile Executor**: provider abstraction, OpenAI-compatible adapter, `ModelOcrProfile` interface/built-ins, LightOnOCR-2-1B, DeepSeek-OCR-2, GLM-OCR, vLLM/self-hosted profiles, health checks, constrained-hardware defaults, implemented only after critic-approved red tests exist.
7. **Quality/Assembly Executor**: incoherence heuristics, verifier SLM extension seam, OCR artifact contract, Markdown assembly separation, implemented only after critic-approved red tests exist.
8. **Verifier/Critic Final Pass**: invariant audit, race-condition review, test-to-requirement coverage matrix, and final acceptance evidence.

Use `$ralph` for final hardening after the invariant suite exists and every test file has a recorded critic disposition.

## Verification gate before merge/ship

- Confirm every production file was introduced after at least one relevant red test and critic-approved test review.
- Confirm every test file in the ownership matrix has a critic disposition: approved, approved-with-follow-up, or rejected-and-rewritten.
- Run full unit suite.
- Run crash-simulation suite.
- Run fake-provider end-to-end job.
- Run large synthetic index/status/resume test.
- Run fd/socket governance test.
- Run self-hosted/vLLM provider profile and health-check tests.
- Run `ModelOcrProfile` request/fingerprint/parser/validator tests for LightOnOCR-2-1B, DeepSeek-OCR-2, and GLM-OCR.
- Run incoherence heuristic tests.
- Run schema-version fail-closed test.
- Produce an invariant checklist mapping each safety property to at least one enforcing test.
- Produce a test-review checklist mapping every test file to the requirement numbers it covers and the critic finding that it is direct, meaningful, and not merely proxy coverage.
