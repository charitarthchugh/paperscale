# Runner Resource Governance & Capacity Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the real `DocumentOcrRunner` page-processing path through `ResourceGovernor` (managed acquisition order) and `ProviderOverloadController` (retry/backoff/circuit-breaker driven by the job's capacity profile), closing issues #4 and #7.

**Architecture:** Keep processing **sequential, one page at a time** (no threads — preserves the existing single-process durability model and matches `ResourceGovernor`'s inherently single-threaded acquisition stack). Each page acquires resources in the global order `scheduler → render → provider → page_lease → state_store`; persistence is governed at `state_store_lock` granularity (the store owns its own transient fds, so `file_descriptor` is intentionally not separately modeled for internal atomic writes). A `ProviderOverloadController`, constructed from `builtin_capacity_profile(manifest.capacity)`, retries transient provider transport errors with exponential backoff and trips a circuit breaker after the profile's failure threshold, at which point the run stops and leaves remaining pages `pending` for a later `resume`.

**Tech Stack:** Python 3.12+, stdlib `unittest`, existing `paperscale.resources.ResourceGovernor`, `paperscale.providers.self_hosted.ProviderCapacityProfile`/`ProviderOverloadController`/`builtin_capacity_profile`.

**Scope note:** `Scheduler`/`JobScheduler`/`LazyPageQueue` in `scheduler.py` are left untouched — they serve compact-index status reads on a different interface and add no behavior to a sequential runner. We adopt only the capacity profile + overload controller + governor.

---

## File Structure

- **Modify** `src/paperscale/providers/self_hosted.py` — add `ProviderOverloadController.record_failure(*, retryable)` so exception-only providers (no HTTP status) can drive backoff/circuit logic.
- **Modify** `src/paperscale/runner.py` — inject `ResourceGovernor` + sleeper; resolve capacity profile; wrap render/provider/state in governed acquisition order; add a retry/circuit loop in `_process_pages`; make `_process_page` return a `_PageOutcome`; route state writes through `state_store_lock`.
- **Create** `tests/test_runner_scheduling.py` — governor ordering, retry/backoff, circuit-breaker behavior.
- **Modify** `tests/providers/test_self_hosted.py` — `record_failure` unit tests.
- **Modify** `docs/ledger-recovery.md` — short note documenting capacity/circuit behavior on the real runner.

**Invariants to preserve (regression):** the existing happy-path runner tests (`tests/test_ocr_runner_cli_integration.py`), schema/fsck/reconcile tests, and the ledger-before-provider durability ordering must all stay green. The real `ResourceGovernor` *raises* `ResourceOrderViolation` on misordering, so a green run is itself proof the ordering is correct.

---

## Task 1: `ProviderOverloadController.record_failure`

Exception-based providers can't call `record_status(code)`. Add a transport-agnostic failure accountant that mirrors the retryable-status branch.

**Files:**
- Modify: `src/paperscale/providers/self_hosted.py` (add method to `ProviderOverloadController`, after `record_status`, ~line 220)
- Test: `tests/providers/test_self_hosted.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/providers/test_self_hosted.py` (reuse its existing imports; add a small capacity factory if one is not already present):

```python
def _capacity(threshold: int = 3, *, initial: float = 1.0, maximum: float = 8.0):
    from paperscale.providers.self_hosted import ProviderCapacityProfile

    return ProviderCapacityProfile(
        name="test",
        max_in_flight_requests=1,
        max_provider_queue=1,
        queue_size=1,
        timeout_seconds=1.0,
        backoff_initial_seconds=initial,
        backoff_max_seconds=maximum,
        circuit_breaker_failure_threshold=threshold,
        circuit_breaker_cooldown_seconds=1.0,
        render_target_longest_side=10,
    )


class RecordFailureTests(unittest.TestCase):
    def test_retryable_failures_open_circuit_after_threshold(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=3))
        first = controller.record_failure(retryable=True)
        second = controller.record_failure(retryable=True)
        third = controller.record_failure(retryable=True)
        self.assertTrue(first.should_retry)
        self.assertFalse(first.circuit_open)
        self.assertTrue(second.should_retry)
        self.assertTrue(third.circuit_open)
        self.assertFalse(third.should_retry)

    def test_backoff_grows_exponentially_then_caps(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=10, initial=1.0, maximum=8.0))
        backoffs = [controller.record_failure(retryable=True).backoff_seconds for _ in range(5)]
        self.assertEqual(backoffs, [1.0, 2.0, 4.0, 8.0, 8.0])

    def test_non_retryable_failure_resets_consecutive_count(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=2))
        controller.record_failure(retryable=True)
        decision = controller.record_failure(retryable=False)
        self.assertFalse(decision.should_retry)
        self.assertFalse(controller.circuit_open)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/providers/test_self_hosted.py -q`
Expected: FAIL — `AttributeError: 'ProviderOverloadController' object has no attribute 'record_failure'`

- [ ] **Step 3: Implement `record_failure`**

In `src/paperscale/providers/self_hosted.py`, add this method to `ProviderOverloadController` immediately after `record_status`:

```python
    def record_failure(self, *, retryable: bool) -> BackoffDecision:
        """Account for a provider transport failure surfaced as an exception.

        Transport-agnostic counterpart to record_status: callers that only see
        exceptions (no HTTP status) use this to drive backoff and the circuit breaker.
        """
        if not retryable:
            self._consecutive_failures = 0
            return BackoffDecision(False, 0.0, False, "non-retryable provider error")
        self._consecutive_failures += 1
        backoff = min(
            self.capacity.backoff_initial_seconds * (2 ** (self._consecutive_failures - 1)),
            self.capacity.backoff_max_seconds,
        )
        return BackoffDecision(
            should_retry=not self.circuit_open,
            backoff_seconds=backoff,
            circuit_open=self.circuit_open,
            diagnostic="provider transport error",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/providers/test_self_hosted.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/paperscale/providers/self_hosted.py tests/providers/test_self_hosted.py
git commit -m "feat(provider): add transport-agnostic record_failure to overload controller"
```

---

## Task 2: Runner scaffolding — inject governor + sleeper, govern state writes (no behavior change)

Wire the governor and sleeper into the runner and route the three persistence primitives through `state_store_lock`. This task is a **pure refactor**: all existing tests must stay green afterward, proving the governance is transparent on the happy path.

**Files:**
- Modify: `src/paperscale/runner.py` (imports ~line 15-28; `__init__` line 145; `_write_indexes` line 607; `_write_ledger` line 652; `_write_manifest` line 582; add `_PageOutcome` + `_capacity_for`)

- [ ] **Step 1: Add imports**

In `src/paperscale/runner.py`, extend the `self_hosted` import and add the resources import. Replace:

```python
from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
)
```

with:

```python
from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    ProviderCapacityProfile,
    ProviderOverloadController,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
)
from paperscale.resources import ResourceGovernor, ResourceKind
```

- [ ] **Step 2: Add the `_PageOutcome` dataclass**

Add near the top of `runner.py`, just after the `RendererFactory = ...` alias (~line 31):

```python
@dataclass(frozen=True, slots=True)
class _PageOutcome:
    """Result of a single page attempt, used to drive retry/circuit decisions."""

    status: str  # "succeeded" | "transport_error" | "content_failure"
```

- [ ] **Step 3: Extend `__init__` to inject governor + sleeper**

Replace the current `__init__` signature/body (line 145):

```python
    def __init__(
        self,
        config: RunnerConfig | None = None,
        *,
        provider: PageOcrProvider | None = None,
        renderer_factory: RendererFactory | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.state = FileSystemStateStore(Path(self.config.state_root))
        self.provider = provider
        self.renderer_factory = renderer_factory or (lambda path, options: PdfPageRenderer(path, render_options=options))
        self.clock = clock or time.time
        self.verifier = DeterministicQualityVerifier()
```

with:

```python
    def __init__(
        self,
        config: RunnerConfig | None = None,
        *,
        provider: PageOcrProvider | None = None,
        renderer_factory: RendererFactory | None = None,
        clock: Callable[[], float] | None = None,
        governor: ResourceGovernor | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or RunnerConfig()
        self.state = FileSystemStateStore(Path(self.config.state_root))
        self.provider = provider
        self.renderer_factory = renderer_factory or (lambda path, options: PdfPageRenderer(path, render_options=options))
        self.clock = clock or time.time
        self.verifier = DeterministicQualityVerifier()
        self.governor = governor or ResourceGovernor()
        self._sleep = sleeper or time.sleep
```

- [ ] **Step 4: Add `_capacity_for` helper**

Add this method to `DocumentOcrRunner` (place it just above `_process_pages`, ~line 395):

```python
    def _capacity_for(self, manifest: JobManifest) -> ProviderCapacityProfile:
        try:
            return builtin_capacity_profile(manifest.capacity)
        except ValueError:
            return builtin_capacity_profile("local-vllm-small")
```

- [ ] **Step 5: Govern the persistence primitives**

Replace `_write_ledger` (line 652):

```python
    def _write_ledger(self, job_id: str, attempt_id: str, payload: Json) -> None:
        self.state.write_json_atomic(self._ledger_rel(job_id, attempt_id), payload)
```

with:

```python
    def _write_ledger(self, job_id: str, attempt_id: str, payload: Json) -> None:
        with self.governor.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._ledger_rel(job_id, attempt_id), payload)
```

Replace `_write_manifest` (line 582):

```python
    def _write_manifest(self, manifest: JobManifest) -> None:
        self.state.write_json_atomic(self._manifest_rel(manifest.job_id), manifest.to_json())
```

with:

```python
    def _write_manifest(self, manifest: JobManifest) -> None:
        with self.governor.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._manifest_rel(manifest.job_id), manifest.to_json())
```

In `_write_indexes` (line 607), replace the three-write tail:

```python
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "status"), status_index)
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "resume"), resume_index)
        self.state.write_json_atomic(self._index_rel(manifest.job_id, "reconcile"), reconcile_index)
        return JobStatus.from_index(status_index)
```

with:

```python
        with self.governor.acquire(ResourceKind.STATE_STORE):
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "status"), status_index)
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "resume"), resume_index)
            self.state.write_json_atomic(self._index_rel(manifest.job_id, "reconcile"), reconcile_index)
        return JobStatus.from_index(status_index)
```

- [ ] **Step 6: Run the full suite to verify no behavior change**

Run: `poetry run pytest -q`
Expected: PASS — same count as before this plan (governance is transparent here; `STATE_STORE` acquired on an empty stack never violates order).

- [ ] **Step 7: Commit**

```bash
git add src/paperscale/runner.py
git commit -m "refactor(runner): inject ResourceGovernor/sleeper and govern state writes"
```

---

## Task 3: Govern render/provider/scheduler order and return a `_PageOutcome`

Wrap rendering and the provider attempt in the correct acquisition order and make `_process_page` report what happened.

**Files:**
- Modify: `src/paperscale/runner.py` (`_process_pages` line 395; `_process_page` line 431)
- Test: `tests/test_runner_scheduling.py` (create)

- [ ] **Step 1: Write the failing ordering test**

Create `tests/test_runner_scheduling.py`:

```python
from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.providers.base import PageOcrResponse, ProviderError
from paperscale.resources import ResourceGovernor, ResourceKind
from paperscale.runner import DocumentOcrRunner, RunnerConfig


@dataclass(frozen=True)
class _FakeRenderedPage:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class _FakeRenderer:
    def __init__(self, pages: list[bytes]) -> None:
        self._pages = pages

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def render_page(self, page_number: int) -> _FakeRenderedPage:
        image = self._pages[page_number - 1]
        return _FakeRenderedPage(page_number, image, hashlib.sha256(image).hexdigest())


class _ScriptedProvider:
    """Provider that raises ProviderError for the first `fail_first` calls, then succeeds."""

    name = "scripted"

    def __init__(self, *, fail_first: int = 0, always_fail: bool = False) -> None:
        self.fail_first = fail_first
        self.always_fail = always_fail
        self.calls = 0

    def send(self, request: Any) -> PageOcrResponse:
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_first:
            raise ProviderError(f"boom {self.calls}")
        return PageOcrResponse(markdown=f"# Page\n\nBody {request.page_id}", provider_request_id=request.fingerprint)


class RecordingGovernor(ResourceGovernor):
    def __init__(self) -> None:
        super().__init__()
        self.acquired: list[ResourceKind | str] = []

    def acquire(self, kind: ResourceKind | str):  # type: ignore[override]
        self.acquired.append(kind)
        return super().acquire(kind)


def _runner(state_root: Path, provider: Any, *, governor: Any = None, sleeper: Any = None, capacity: str = "local-vllm-small") -> DocumentOcrRunner:
    return DocumentOcrRunner(
        RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm", capacity=capacity),
        provider=provider,
        renderer_factory=lambda _path, _options: _FakeRenderer([b"page-1"]),
        governor=governor,
        sleeper=sleeper,
    )


class GovernedOrderingTests(unittest.TestCase):
    def test_page_processing_acquires_resources_in_global_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            governor = RecordingGovernor()
            runner = _runner(state_root, _ScriptedProvider(), governor=governor)
            status = runner.run(input_path=state_root / "in.pdf", output_path=state_root / "out.md", job_id="job")
            self.assertEqual(status.succeeded, 1)
            # Real ResourceGovernor raises on misordering, so a clean run is proof.
            # Assert the key managed resources were actually acquired (not bypassed).
            self.assertIn(ResourceKind.SCHEDULER, governor.acquired)
            self.assertIn(ResourceKind.RENDER, governor.acquired)
            self.assertIn(ResourceKind.PROVIDER, governor.acquired)
            self.assertIn(ResourceKind.PAGE_LEASE, governor.acquired)
            self.assertIn(ResourceKind.STATE_STORE, governor.acquired)
            # render is acquired before provider for the same page
            self.assertLess(
                governor.acquired.index(ResourceKind.RENDER),
                governor.acquired.index(ResourceKind.PROVIDER),
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/test_runner_scheduling.py::GovernedOrderingTests -q`
Expected: FAIL — `ResourceKind.SCHEDULER` (and others) not in `governor.acquired`, because the current runner never touches the governor in the processing path.

- [ ] **Step 3: Rewrite `_process_pages`**

Replace the body of `_process_pages` (line 395):

```python
    def _process_pages(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
    ) -> JobStatus:
        profile = get_builtin_profile(manifest.profile)
        for page_number in range(1, manifest.page_count + 1):
            page = pages[str(page_number)]
            state = page.get("state")
            if state == "succeeded":
                continue
            if state == "ambiguous" and not retry_ambiguous:
                continue
            if state not in {"pending", "failed_retryable", "reserved", "ambiguous"}:
                continue
            rendered = renderer.render_page(page_number)
            request = profile.build_request(
                f"{manifest.document_id}:{page_number}",
                rendered.image_bytes,
                getattr(rendered, "media_type", "image/png"),
                model=manifest.model,
            )
            self._process_page(manifest, pages, page_number, request)
        self._assemble_if_ready(manifest, pages, allow_partial=allow_partial)
        return self._write_indexes(manifest, pages, partial=allow_partial and _count_states(pages).get("succeeded", 0) < manifest.page_count)
```

with:

```python
    def _process_pages(
        self,
        manifest: JobManifest,
        pages: Json,
        *,
        renderer: Any,
        allow_partial: bool,
        retry_ambiguous: bool,
    ) -> JobStatus:
        profile = get_builtin_profile(manifest.profile)
        capacity = self._capacity_for(manifest)
        overload = ProviderOverloadController(capacity)
        for page_number in range(1, manifest.page_count + 1):
            if overload.circuit_open:
                break
            page = pages[str(page_number)]
            state = page.get("state")
            if state == "succeeded":
                continue
            if state == "ambiguous" and not retry_ambiguous:
                continue
            if state not in {"pending", "failed_retryable", "reserved", "ambiguous"}:
                continue
            with self.governor.acquire(ResourceKind.SCHEDULER):
                with self.governor.acquire(ResourceKind.RENDER):
                    rendered = renderer.render_page(page_number)
                request = profile.build_request(
                    f"{manifest.document_id}:{page_number}",
                    rendered.image_bytes,
                    getattr(rendered, "media_type", "image/png"),
                    model=manifest.model,
                )
                while True:
                    outcome = self._process_page(manifest, pages, page_number, request)
                    if outcome.status != "transport_error":
                        if outcome.status == "succeeded":
                            overload.record_success()
                        break
                    decision = overload.record_failure(retryable=True)
                    if not decision.should_retry:
                        break
                    self._sleep(decision.backoff_seconds)
        self._assemble_if_ready(manifest, pages, allow_partial=allow_partial)
        return self._write_indexes(manifest, pages, partial=allow_partial and _count_states(pages).get("succeeded", 0) < manifest.page_count)
```

- [ ] **Step 4: Rewrite `_process_page` to govern provider/lease and return `_PageOutcome`**

Replace the entire `_process_page` method (lines 431-521):

```python
    def _process_page(self, manifest: JobManifest, pages: Json, page_number: int, request: Any) -> _PageOutcome:
        page_key = str(page_number)
        previous = pages[page_key]
        epoch = int(previous.get("epoch") or 0) + 1
        attempt_id = str(uuid.uuid4())
        now = float(self.clock())
        with self.governor.acquire(ResourceKind.PROVIDER):
            with self.governor.acquire(ResourceKind.PAGE_LEASE):
                attempt = {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "kind": "page_attempt",
                    "attempt_id": attempt_id,
                    "job_id": manifest.job_id,
                    "page_id": request.page_id,
                    "page_number": page_number,
                    "state": "reserved",
                    "fingerprint": request.fingerprint,
                    "worker_id": self.config.worker_id,
                    "epoch": epoch,
                    "lease_expires_at": now + self.config.lease_seconds,
                    "heartbeat_at": now,
                    "provider_started_at": None,
                    "provider_response_committed_at": None,
                    "result_pointer": None,
                    "diagnostic": None,
                }
                self._write_ledger(manifest.job_id, attempt_id, attempt)
                pages[page_key] = {
                    "state": "reserved",
                    "epoch": epoch,
                    "attempt_id": attempt_id,
                    "fingerprint": request.fingerprint,
                }
                self._write_indexes(manifest, pages, partial=False)

                attempt = {**attempt, "state": "in_flight", "provider_started_at": float(self.clock())}
                self._write_ledger(manifest.job_id, attempt_id, attempt)
                pages[page_key] = {**pages[page_key], "state": "in_flight"}
                self._write_indexes(manifest, pages, partial=False)

                try:
                    response = self._provider_for(manifest).send(request)
                except Exception as exc:
                    self._fail_attempt(manifest, pages, page_number, attempt, state="failed_retryable", diagnostic=str(exc))
                    return _PageOutcome("transport_error")

                parsed = get_builtin_profile(manifest.profile).parse_and_validate(response.markdown)
                if not parsed.ok:
                    state = "failed_terminal" if parsed.retry_classification == "terminal" else "failed_retryable"
                    self._fail_attempt(manifest, pages, page_number, attempt, state=state, diagnostic=parsed.diagnostic)
                    return _PageOutcome("content_failure")
                finding = self.verifier.classify(parsed.markdown)
                if not finding.accepted:
                    state = "failed_terminal" if finding.retry_class == "terminal" else "failed_retryable"
                    self._fail_attempt(manifest, pages, page_number, attempt, state=state, diagnostic=finding.kind)
                    return _PageOutcome("content_failure")

                artifact_rel = self._artifact_rel(manifest.job_id, page_number)
                artifact = {
                    "schema_version": CURRENT_SCHEMA_VERSION,
                    "kind": "page_artifact",
                    "document_id": manifest.document_id,
                    "page_number": page_number,
                    "page_id": request.page_id,
                    "markdown": parsed.markdown,
                    "result_pointer": str(artifact_rel),
                    "verifier_metadata": [finding.__dict__ if hasattr(finding, "__dict__") else {
                        "accepted": finding.accepted,
                        "kind": finding.kind,
                        "retry_class": finding.retry_class,
                        "warnings": list(finding.warnings),
                    }],
                    "fingerprint": request.fingerprint,
                    "image_hash": request.image_hash,
                    "provider_request_id": response.provider_request_id,
                    "provider_metadata": response.metadata,
                    "profile_metadata": parsed.metadata,
                }
                with self.governor.acquire(ResourceKind.STATE_STORE):
                    self.state.write_json_atomic(artifact_rel, artifact)
                committed = {
                    **attempt,
                    "state": "succeeded",
                    "provider_response_committed_at": float(self.clock()),
                    "result_pointer": str(artifact_rel),
                }
                self._write_ledger(manifest.job_id, attempt_id, committed)
                pages[page_key] = {
                    **pages[page_key],
                    "state": "succeeded",
                    "artifact_path": str(artifact_rel),
                    "fingerprint": request.fingerprint,
                }
                self._write_indexes(manifest, pages, partial=False)
                return _PageOutcome("succeeded")
```

- [ ] **Step 5: Run the ordering test + full suite**

Run: `poetry run pytest tests/test_runner_scheduling.py::GovernedOrderingTests -q && poetry run pytest -q`
Expected: PASS. Note: if `ResourceOrderViolation` is raised, the nesting order is wrong — provider (5) MUST be acquired before page_lease (6), and render (3) released before provider acquired. The structure above satisfies this.

- [ ] **Step 6: Commit**

```bash
git add src/paperscale/runner.py tests/test_runner_scheduling.py
git commit -m "feat(runner): govern render/provider/state acquisition order per page"
```

---

## Task 4: Retry transient provider errors with backoff

**Files:**
- Modify: `src/paperscale/runner.py` (already loop-capable from Task 3 — this task only adds the test that proves it)
- Test: `tests/test_runner_scheduling.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_runner_scheduling.py`:

```python
class RetryBackoffTests(unittest.TestCase):
    def test_transient_provider_errors_are_retried_with_backoff_then_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            sleeps: list[float] = []
            provider = _ScriptedProvider(fail_first=2)
            runner = _runner(state_root, provider, sleeper=sleeps.append, capacity="local-vllm-small")
            status = runner.run(input_path=state_root / "in.pdf", output_path=state_root / "out.md", job_id="job")
            self.assertEqual(status.succeeded, 1)
            self.assertEqual(provider.calls, 3)          # 2 failures + 1 success
            self.assertEqual(sleeps, [1.0, 2.0])         # exponential backoff from local-vllm-small (initial 1.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/test_runner_scheduling.py::RetryBackoffTests -q`
Expected: This should already PASS if Task 3 is complete (the retry loop + sleeper are in place). If it FAILS, that means Task 3's loop is wrong — fix the loop in `_process_pages`, not the test.

> Note on TDD ordering: the retry loop lives in `_process_pages` (Task 3) because the governed `with` blocks make a standalone minimal implementation impractical. This test is the behavioral proof for the retry path; treat a failure here as a Task 3 regression.

- [ ] **Step 3: Run the full suite**

Run: `poetry run pytest -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_runner_scheduling.py
git commit -m "test(runner): transient provider errors retried with exponential backoff"
```

---

## Task 5: Circuit breaker stops the run and leaves remaining pages pending

**Files:**
- Modify: `src/paperscale/runner.py` (behavior already present from Task 3 — verify via test)
- Test: `tests/test_runner_scheduling.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_runner_scheduling.py` (uses a 2-page renderer, so override the factory inline):

```python
class CircuitBreakerTests(unittest.TestCase):
    def test_circuit_opens_after_threshold_and_leaves_remaining_pages_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp) / ".paperscale"
            sleeps: list[float] = []
            provider = _ScriptedProvider(always_fail=True)
            runner = DocumentOcrRunner(
                RunnerConfig(state_root=state_root, base_url="http://fake/v1", model="mock-vlm", capacity="local-vllm-small"),
                provider=provider,
                renderer_factory=lambda _path, _options: _FakeRenderer([b"page-1", b"page-2"]),
                sleeper=sleeps.append,
            )
            status = runner.run(
                input_path=state_root / "in.pdf",
                output_path=state_root / "out.md",
                job_id="job",
                allow_partial=True,
            )
            # local-vllm-small threshold is 3: 3 failed attempts on page 1 open the circuit.
            self.assertEqual(provider.calls, 3)
            self.assertEqual(status.succeeded, 0)
            self.assertEqual(status.failed_retryable, 1)   # page 1 left failed_retryable
            self.assertEqual(status.pending, 1)            # page 2 never started
            self.assertEqual(len(sleeps), 2)               # 2 backoffs before the 3rd failure trips the breaker
```

- [ ] **Step 2: Run it to verify it fails or passes**

Run: `poetry run pytest tests/test_runner_scheduling.py::CircuitBreakerTests -q`
Expected: PASS if Task 3's `if overload.circuit_open: break` and the retry loop are correct. If `status.pending` is 0 (page 2 was attempted) or `provider.calls != 3`, the circuit-break guard at the top of the page loop is missing/misplaced — fix `_process_pages`.

- [ ] **Step 3: Run the full suite + lint**

Run: `poetry run pytest -q && poetry run ruff check src/paperscale/runner.py src/paperscale/providers/self_hosted.py tests/test_runner_scheduling.py`
Expected: PASS, `All checks passed!`

- [ ] **Step 4: Commit**

```bash
git add tests/test_runner_scheduling.py
git commit -m "test(runner): circuit breaker halts run and preserves pending pages"
```

---

## Task 6: Document capacity/circuit behavior

**Files:**
- Modify: `docs/ledger-recovery.md`

- [ ] **Step 1: Append an "Operator behavior" note**

Add a subsection to `docs/ledger-recovery.md` documenting: the real runner processes pages sequentially under `ResourceGovernor` ordering (`scheduler → render → provider → page_lease → state_store`); transient provider errors are retried with exponential backoff bounded by the job's capacity profile (`builtin_capacity_profile(manifest.capacity)`); after `circuit_breaker_failure_threshold` consecutive failures the circuit opens, the run stops, and remaining pages stay `pending` for a later `paperscale resume`. Keep it to ~8 lines, matching the file's existing terse style.

- [ ] **Step 2: Commit**

```bash
git add docs/ledger-recovery.md
git commit -m "docs: describe real-runner capacity gating and circuit-breaker behavior"
```

---

## Self-Review Notes (for the implementer)

- **Acquisition order is load-bearing.** The governor raises on any out-of-order acquire. The only valid nesting is `SCHEDULER(2) → RENDER(3)`(release) `→ PROVIDER(5) → PAGE_LEASE(6) → STATE_STORE(7)`(per write). Never acquire a lower-numbered resource while holding a higher one.
- **`file_descriptor(4)` is deliberately not modeled** for internal atomic writes; the store encapsulates its own fds. Governing at `state_store_lock(7)` is the documented granularity. Do not route `FileSystemStateStore` opens through `governor.open_file` in this plan — that's a separate, larger change.
- **The circuit threshold is the retry budget.** There is no separate `max_attempts`; `record_failure` stops retrying once `circuit_open` (consecutive failures ≥ `circuit_breaker_failure_threshold`). This is bounded and cannot infinite-loop.
- **Content failures (parse/verify) are not retried in-run** — `_process_page` returns `"content_failure"` and the loop breaks, matching pre-existing behavior. Only `"transport_error"` drives the overload controller.
- **Type consistency:** `_PageOutcome.status` is one of exactly `"succeeded"`, `"transport_error"`, `"content_failure"`; `record_failure(*, retryable: bool) -> BackoffDecision`; `_capacity_for(manifest) -> ProviderCapacityProfile`.
- **Regression proof:** Tasks 2, 3, 5 each run the full suite. The happy-path integration tests passing under the real `ResourceGovernor` is the proof that ordering is correct.
