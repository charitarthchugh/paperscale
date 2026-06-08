from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any
import tempfile
import unittest
import warnings
from unittest.mock import patch

from paperscale.assembly import PageMarkdownArtifact, assemble_document_markdown
from paperscale.cli import CliApp, main as paperscale_main
from paperscale.contracts import PageArtifact, UnknownSchemaVersionError, load_versioned_record
from paperscale.ledger import AmbiguousAttemptError, InMemoryLedgerStore, Ledger, LedgerState, StaleEpochError
from paperscale.profiles.base import ModelOcrProfile
from paperscale.profiles.builtin import get_builtin_profile, with_profile_override
from paperscale.providers.base import ProviderError
from paperscale.providers.openai_chat import OpenAIChatProvider
from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    ProviderCapacityProfile,
    ProviderOverloadController,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
    builtin_capacity_profile_names,
)
from paperscale.quality.verifier import DeterministicQualityVerifier, assess_markdown_fragment
from paperscale.resources import ResourceGovernor, ResourceKind, ResourceOrderViolation
from paperscale.scheduler import (
    CompactIndexError,
    JobScheduler,
    ProviderCapacity,
    RetryStormController,
    Scheduler,
)
from paperscale.state.fs_store import FileSystemStateStore
from tests.harness.fakes import RecordingStateStore


class AssemblyAndCliCoverageTests(unittest.TestCase):
    def test_assemble_document_rejects_invalid_inputs_and_quality_failures(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pages"):
            assemble_document_markdown([])
        with self.assertRaisesRegex(ValueError, "single document"):
            assemble_document_markdown(
                [
                    PageMarkdownArtifact("doc-a", 1, "ok"),
                    PageMarkdownArtifact("doc-b", 2, "ok"),
                ]
            )
        with self.assertRaisesRegex(ValueError, "one-based"):
            assemble_document_markdown([PageMarkdownArtifact("doc", 0, "ok")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assemble_document_markdown(
                [PageMarkdownArtifact("doc", 1, "ok"), PageMarkdownArtifact("doc", 1, "ok again")]
            )
        with self.assertRaisesRegex(ValueError, "quality check failed"):
            assemble_document_markdown([PageMarkdownArtifact("doc", 1, "same same same same same same")], enforce_quality=True)

    def test_assemble_document_marks_partial_and_normalizes_fragments(self) -> None:
        markdown = assemble_document_markdown(
            [PageMarkdownArtifact("doc", 2, "  ## Page 2  "), PageMarkdownArtifact("doc", 1, "# Page 1\n")],
            title=" Title ",
            partial=True,
        )
        self.assertTrue(markdown.startswith("# Title\n\n<!-- partial -->"))
        self.assertLess(markdown.index("# Page 1"), markdown.index("## Page 2"))
        self.assertTrue(markdown.endswith("\n"))

    def test_cli_assemble_reads_jsonl_skips_blanks_and_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "pages.jsonl"
            output_path = root / "out" / "doc.md"
            input_path.write_text(
                '{"document_id":"doc","page_number":1,"markdown":"# One"}\n\n'
                '{"document_id":"doc","page_number":2,"markdown":"# Two"}\n',
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = paperscale_main(["assemble", "--input", str(input_path), "--output", str(output_path)])
            self.assertEqual(exit_code, 0)
            self.assertIn("assembled 2 pages", stdout.getvalue())
            self.assertIn("# One", output_path.read_text(encoding="utf-8"))

    def test_cli_run_requires_arguments_and_doctor_reports_provider_health(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            paperscale_main(["run"])
        self.assertEqual(cm.exception.code, 2)

        fake_payload = {
            "endpoint": "http://fake/v1/",
            "observed_models": ["mock-vlm"],
            "model": "mock-vlm",
            "ocr_profile": "generic_vlm_markdown",
            "capacity_profile": "local-vllm-small",
            "compatible": True,
            "diagnostic": "ok",
        }
        stdout = io.StringIO()
        with patch("paperscale.runner.doctor_provider", return_value=fake_payload), contextlib.redirect_stdout(stdout):
            exit_code = paperscale_main(["doctor", "provider", "--base-url", "http://fake", "--model", "mock-vlm"])
        self.assertEqual(exit_code, 0)
        self.assertIn("compatible: true", stdout.getvalue())

    def test_cli_rejects_bad_page_artifact_rows(self) -> None:
        bad_rows = [
            ({"page_number": 1, "markdown": "ok"}, "missing required field 'document_id'"),
            ({"document_id": "", "page_number": 1, "markdown": "ok"}, "invalid document_id"),
            ({"document_id": "doc", "page_number": "1", "markdown": "ok"}, "invalid page_number"),
            ({"document_id": "doc", "page_number": 1, "markdown": 3}, "invalid markdown"),
        ]
        for row, message in bad_rows:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                input_path = Path(tmp) / "pages.jsonl"
                output_path = Path(tmp) / "out.md"
                input_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    paperscale_main(["assemble", "--input", str(input_path), "--output", str(output_path)])

    def test_cli_app_repair_index_scans_tree_and_fsync_parent_tolerates_open_error(self) -> None:
        store = RecordingStateStore()
        self.assertEqual(CliApp(store=store).run(["repair-index", "job-1"]), 0)
        self.assertEqual(store.tree_scans, 1)
        from paperscale import cli

        with patch.object(cli.os, "open", side_effect=OSError("no dir fsync")):
            cli._fsync_parent(Path("/tmp"))


class SchedulerAndContractsCoverageTests(unittest.TestCase):
    def test_scheduler_paths_cover_status_resume_repair_capacity_and_overload(self) -> None:
        class Index:
            def __init__(self) -> None:
                self.status_reads = 0

            def iter_pending_page_ids(self):
                return iter(["doc:1", "doc:2", "doc:3"])

            def read_status_index(self):
                self.status_reads += 1
                return {"ok": True}

            def read_resume_index(self):
                return {"resume": True}

            def full_tree_scan(self):
                return {"scanned": True}

        index = Index()
        scheduler = Scheduler(index, queue_size=10, capacity=ProviderCapacity(max_in_flight=1, max_provider_queue=2))
        self.assertEqual(scheduler.status(), {"ok": True})
        self.assertEqual(scheduler.resume_plan(), {"resume": True})
        self.assertEqual(scheduler.repair_index(), {"scanned": True})
        scheduler.fill_queue()
        self.assertEqual(scheduler.queued_count, 2)
        controller = RetryStormController(ProviderCapacity(max_in_flight=1, max_provider_queue=1, circuit_breaker_threshold=1))
        controller.record_overload("429")
        scheduler.on_provider_overload(controller)
        self.assertEqual(scheduler.queued_count, 2)

    def test_scheduler_error_and_zero_capacity_paths(self) -> None:
        with self.assertRaises(CompactIndexError):
            JobScheduler(store=RecordingStateStore(records={"job-index": []}), queue_size=1).status("job")

        class BadResumeIndex:
            def iter_pending_page_ids(self):
                return iter([])

            def read_resume_index(self):
                raise RuntimeError("broken")

        scheduler = Scheduler(BadResumeIndex(), queue_size=3, capacity=ProviderCapacity(max_in_flight=1, max_provider_queue=0))
        with self.assertRaises(CompactIndexError):
            scheduler.resume_plan()
        scheduler.fill_queue()
        self.assertEqual(scheduler.queued_count, 0)

    def test_lazy_queue_iteration_and_contract_error_paths(self) -> None:
        queue = JobScheduler(store=RecordingStateStore(), queue_size=2).build_lazy_queue(document_id="doc", page_count=3)
        self.assertEqual(list(queue), ["doc:1", "doc:2", "doc:3"])
        with self.assertRaises(UnknownSchemaVersionError):
            load_versioned_record({"kind": "page_attempt"}, expected_kind="page_attempt")
        mutations: list[str] = []
        loaded = load_versioned_record(
            {"schema_version": 1, "kind": "page_attempt", "id": "a"},
            expected_kind="page_attempt",
            on_mutation=lambda event, _record: mutations.append(event),
        )
        self.assertEqual(loaded["id"], "a")
        self.assertEqual(mutations, ["validated"])
        with self.assertRaisesRegex(ValueError, "expected record kind"):
            load_versioned_record({"schema_version": 1, "kind": "other"}, expected_kind="page_attempt")
        with self.assertRaisesRegex(ValueError, "numeric page number"):
            _ = PageArtifact(page_id="doc:not-a-number", markdown="", result_pointer="x").page_number


class ProvidersProfilesAndQualityCoverageTests(unittest.TestCase):
    def test_openai_provider_raises_when_response_has_no_text(self) -> None:
        class BadClient:
            class Responses:
                def create(self, **_kwargs: object) -> object:
                    return object()

            responses = Responses()

        request = get_builtin_profile("generic_vlm_markdown").build_request("doc:1", b"image", "image/png")
        with self.assertRaisesRegex(ProviderError, "did not include text"):
            OpenAIChatProvider(client=BadClient()).send(request)

    def test_profile_constructor_and_builtin_error_paths(self) -> None:
        base_kwargs = {
            "name": "x",
            "default_model": "m",
            "prompt_template": "Return Markdown for {page_id}",
            "prompt_version": "v1",
            "parser_version": "p1",
            "output_format": "markdown",
            "decoding": {},
            "render_options": {},
        }
        with self.assertRaisesRegex(ValueError, "document_to_markdown"):
            ModelOcrProfile(**{**base_kwargs, "task": "visual_qa"})
        with self.assertRaisesRegex(ValueError, "non-Markdown"):
            ModelOcrProfile(**{**base_kwargs, "public_modes": ("document_to_markdown", "visual_qa")})
        with self.assertRaisesRegex(ValueError, "Markdown"):
            ModelOcrProfile(**{**base_kwargs, "prompt_template": "Return text for {page_id}"})
        with self.assertRaisesRegex(ValueError, "unknown OCR profile"):
            get_builtin_profile("missing")
        overridden = with_profile_override("generic_vlm_markdown", prompt_version="v2")
        self.assertEqual(overridden.prompt_version, "v2")

    def test_self_hosted_provider_validation_and_health_error_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "endpoint"):
            InferenceServerProfile(endpoint="", served_model="m")
        with self.assertRaisesRegex(ValueError, "served_model"):
            InferenceServerProfile(endpoint="http://x", served_model="")
        with self.assertRaisesRegex(ValueError, "timeout"):
            InferenceServerProfile(endpoint="http://x", served_model="m", timeout_seconds=0)
        with self.assertRaisesRegex(ValueError, "unknown capacity"):
            builtin_capacity_profile("missing")
        self.assertIn("local-vllm-small", builtin_capacity_profile_names())
        with self.assertRaisesRegex(ValueError, "positive"):
            ProviderCapacityProfile("bad", 0, 1, 1, 1, 1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "queue_size"):
            ProviderCapacityProfile("bad", 1, 1, 2, 1, 1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "backoff_max"):
            ProviderCapacityProfile("bad", 1, 1, 1, 1, 2, 1, 1, 1, 1)

    def test_self_hosted_health_non_200_exception_and_bad_payload(self) -> None:
        class Response:
            def __init__(self, status_code: int, payload: object) -> None:
                self.status_code = status_code
                self._payload = payload

            def json(self) -> object:
                return self._payload

        class Client:
            def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
                self.response = response
                self.error = error

            def get(self, url: str, *, timeout: float) -> Any:
                del url, timeout
                if self.error:
                    raise self.error
                return self.response

        server = InferenceServerProfile("http://localhost:1/v1", "model")
        capacity = builtin_capacity_profile("local-vllm-small")
        self.assertFalse(SelfHostedOpenAICompatibleProvider(server, capacity, http_client=Client(error=RuntimeError("down"))).health_check().ok)
        self.assertFalse(SelfHostedOpenAICompatibleProvider(server, capacity, http_client=Client(Response(503, {}))).health_check().ok)
        self.assertEqual(
            SelfHostedOpenAICompatibleProvider(server, capacity, http_client=Client(Response(200, []))).health_check().observed_models,
            (),
        )
        self.assertEqual(
            SelfHostedOpenAICompatibleProvider(server, capacity, http_client=Client(Response(200, {"data": "bad"}))).health_check().observed_models,
            (),
        )
        self.assertTrue(SelfHostedOpenAICompatibleProvider(server, capacity, http_client=Client(Response(200, {"data": [{"id": "model"}]}))).profile_fingerprint())

    def test_provider_overload_controller_success_and_non_retry_paths(self) -> None:
        controller = ProviderOverloadController(builtin_capacity_profile("local-vllm-small"))
        self.assertTrue(controller.try_enqueue())
        controller.mark_dequeued()
        self.assertEqual(controller.queued_requests, 0)
        self.assertEqual(controller.record_status(400).diagnostic, "non-retryable HTTP 400")
        self.assertEqual(controller.record_success().diagnostic, "success")

    def test_quality_verifier_detects_all_mock_failure_shapes(self) -> None:
        cases = {
            "control_characters": "hello\x00world",
            "refusal_boilerplate": "I'm sorry, I cannot help with that.",
            "malformed_frontmatter": "---\ntitle: [oops\n---\nbody",
            "repeated_character": "A" * 24,
            "repeated_ngram": "same same same same same same",
            "truncation_indicator": "This output is truncated",
            "length_anomaly": "tiny",
        }
        for expected_code, markdown in cases.items():
            with self.subTest(expected_code=expected_code):
                report = assess_markdown_fragment(markdown)
                self.assertFalse(report.accepted)
                self.assertIn(expected_code, [issue.code for issue in report.issues])
        refusal = DeterministicQualityVerifier().classify("I'm sorry, I cannot provide that")
        self.assertEqual(refusal.kind, "refusal")
        self.assertEqual(refusal.retry_class, "terminal")
        ok = DeterministicQualityVerifier(optional_slm=object()).classify("# Heading\n\nClean markdown body")
        self.assertTrue(ok.accepted)


class ResourcesLedgerAndStoreCoverageTests(unittest.TestCase):
    def test_resource_governor_enum_file_management_and_errors(self) -> None:
        handles: list[io.StringIO] = []

        def opener(_path: Path, _mode: str, **_kwargs: object) -> io.StringIO:
            handle = io.StringIO()
            handles.append(handle)
            return handle

        governor = ResourceGovernor(file_opener=opener)
        with governor.acquire(ResourceKind.FILE_DESCRIPTOR):
            self.assertTrue(governor.is_held(ResourceKind.FILE_DESCRIPTOR))
        with governor.managed_open_file("ignored", "w") as handle:
            handle.write("x")
        self.assertTrue(handles[-1].closed)
        with self.assertRaisesRegex(ValueError, "unknown resource"):
            governor.is_held("unknown")
        with self.assertRaises(ResourceOrderViolation):
            with governor.acquire_many(["provider_concurrency", "file_descriptor"]):
                pass

    def test_filesystem_store_status_list_and_weak_fsync_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FileSystemStateStore(root)
            self.assertEqual(store.list_committed("missing"), [])
            self.assertEqual(store.status_from_index(), {})
            store.write_json_atomic("index.json", {"ok": True})
            self.assertEqual(store.status_from_index(), {"ok": True})
            weak = FileSystemStateStore(root / "weak", allow_weak_fs=True)
            with patch.object(os, "open", side_effect=OSError("weak fs")):
                weak._fsync_dir(root / "weak")

    def test_inmemory_ledger_low_level_error_and_recovery_paths(self) -> None:
        now = 100.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: now)
        attempt = ledger.reserve_page(page_id="doc:1", provider_request_fingerprint="fp", worker_id="w1", lease_seconds=1)
        with self.assertRaisesRegex(RuntimeError, "not reservable"):
            ledger.reserve_page(page_id="doc:1", provider_request_fingerprint="fp", worker_id="w2", lease_seconds=1)
        with self.assertRaises(StaleEpochError):
            ledger.heartbeat(attempt.attempt_id, worker_id="other", epoch=attempt.epoch)
        started = ledger.mark_provider_call_started(attempt.attempt_id, worker_id="w1", epoch=attempt.epoch)
        ledger.recover_expired_leases(now=started.lease_expires_at + 1)
        with self.assertRaises(AmbiguousAttemptError):
            ledger.reserve_page(page_id="doc:1", provider_request_fingerprint="fp", worker_id="w2", lease_seconds=1)
        retry = ledger.reserve_page(
            page_id="doc:1",
            provider_request_fingerprint="fp",
            worker_id="w2",
            lease_seconds=1,
            retry_ambiguous=True,
        )
        self.assertEqual(retry.epoch, 2)
        self.assertEqual(store.require_latest("doc:1").state, LedgerState.RESERVED)

    def test_inmemory_ledger_requeues_reserved_and_commits_success(self) -> None:
        now = 10.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: now)
        ledger.reserve_page(page_id="doc:2", provider_request_fingerprint="fp", worker_id="w", lease_seconds=1)
        report = ledger.recover_expired_leases(now=12.0)
        self.assertEqual(report.requeued_pages, ["doc:2"])
        retry = ledger.reserve_page(page_id="doc:2", provider_request_fingerprint="fp", worker_id="w", lease_seconds=1)
        committed = ledger.commit_success(retry.attempt_id, worker_id="w", epoch=retry.epoch, result_pointer="out.md")
        self.assertEqual(committed.result_pointer, "out.md")
        self.assertEqual(committed.state, LedgerState.SUCCEEDED)

class NearFullCoverageEdgesTests(unittest.TestCase):
    def test_cli_atomic_write_cleanup_and_module_main_guards(self) -> None:
        from paperscale import cli

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.md"
            with patch.object(cli.os, "replace", side_effect=RuntimeError("replace failed")):
                with self.assertRaisesRegex(RuntimeError, "replace failed"):
                    cli._atomic_write_text(output, "content")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

        store = RecordingStateStore()
        self.assertEqual(CliApp(store=store).run(["fsck", "job-1"]), 0)

        import runpy
        import sys

        stdout = io.StringIO()
        fake_payload = {
            "endpoint": "http://fake/v1/",
            "observed_models": ["mock-vlm"],
            "model": "mock-vlm",
            "ocr_profile": "generic_vlm_markdown",
            "capacity_profile": "local-vllm-small",
            "compatible": True,
            "diagnostic": "ok",
        }
        with (
            patch.object(sys, "argv", ["paperscale", "doctor", "provider", "--base-url", "http://fake", "--model", "mock-vlm"]),
            patch("paperscale.runner.doctor_provider", return_value=fake_payload),
            contextlib.redirect_stdout(stdout),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(SystemExit) as cm:
                runpy.run_module("paperscale.cli", run_name="__main__")
        self.assertEqual(cm.exception.code, 0)

    def test_mock_cli_missing_dependency_and_module_main_guard(self) -> None:
        import builtins
        import runpy
        import sys

        from paperscale.mock_api import cli as mock_cli

        real_import = builtins.__import__

        def blocked_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "uvicorn":
                raise ModuleNotFoundError("uvicorn")
            return real_import(name, *args, **kwargs)

        stderr = io.StringIO()
        with patch.object(builtins, "__import__", side_effect=blocked_import), contextlib.redirect_stderr(stderr):
            exit_code = mock_cli.main(["serve"])
        self.assertEqual(exit_code, 2)
        self.assertIn("paperscale[mock-api]", stderr.getvalue())

        stdout = io.StringIO()
        with patch.object(sys, "argv", ["paperscale-mock-api", "--help"]), contextlib.redirect_stdout(stdout), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaises(SystemExit) as cm:
                runpy.run_module("paperscale.mock_api.cli", run_name="__main__")
        self.assertEqual(cm.exception.code, 0)

    def test_mock_api_malformed_parts_and_direct_fallback_branch(self) -> None:
        from paperscale.mock_api.app import MockApiConfig, _scenario_output, ValidatedRequest
        from tests.test_mock_api import _data_url
        from fastapi.testclient import TestClient
        from paperscale.mock_api import create_app

        client = TestClient(create_app(MockApiConfig()))
        response = client.post(
            "/v1/responses",
            json={
                "model": "mock-vlm",
                "input": [
                    "ignored",
                    {"role": "user", "content": "ignored"},
                    {
                        "role": "user",
                        "content": [
                            "ignored",
                            {"type": "input_text", "text": "OCR"},
                            {"type": "input_image", "image_url": _data_url(b"ok")},
                        ],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-vlm",
                "messages": [
                    {"role": "user", "content": 3},
                    {
                        "role": "user",
                        "content": [
                            3,
                            {"type": "text", "text": "OCR"},
                            {"type": "image_url", "image_url": _data_url(b"ok")},
                        ],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        fallback = _scenario_output(
            "unknown",
            ValidatedRequest(
                endpoint="responses",
                model="mock-vlm",
                prompt="p",
                prompt_sha256="ph",
                image_sha256="ih",
                image_media_type="image/png",
                image_bytes=b"i",
                request_id="rid",
            ),
        )
        self.assertIn("request_id: rid", fallback)

    def test_page_ledger_retryable_heartbeat_and_skip_paths(self) -> None:
        from paperscale.contracts import PageAttemptState, PageTask
        from paperscale.ledger import PageLedger
        from tests.harness.fakes import FakeClock

        clock = FakeClock()
        ledger = PageLedger(store=RecordingStateStore(), clock=clock)
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        attempt = ledger.reserve_page_attempt(task, worker_id="w1", fingerprint="fp", lease_seconds=10)
        self.assertIsNone(ledger.next_retryable_page())
        self.assertEqual(ledger.recover_expired_attempts(), {})
        heartbeat = ledger.heartbeat(attempt.attempt_id, epoch=attempt.epoch, lease_seconds=20)
        self.assertGreater(heartbeat.lease_expires_at, attempt.lease_expires_at)
        with self.assertRaises(StaleEpochError):
            ledger.heartbeat(attempt.attempt_id, epoch=attempt.epoch + 1)
        clock.advance(21)
        recovered = ledger.recover_expired_attempts()[attempt.attempt_id]
        self.assertEqual(recovered.state, PageAttemptState.PENDING)
        self.assertEqual(ledger.next_retryable_page(), task)

        retry = ledger.steal_expired_attempt(attempt.attempt_id, worker_id="w2")
        ledger.mark_provider_started(retry.attempt_id)
        clock.advance(30)
        recovered_retry = ledger.recover_expired_attempts()[retry.attempt_id]
        self.assertEqual(recovered_retry.state, PageAttemptState.AMBIGUOUS)
        self.assertEqual(ledger.next_retryable_page(allow_ambiguous=True), task)
        with self.assertRaises(AmbiguousAttemptError):
            ledger.reserve_page_attempt(task, worker_id="w3", fingerprint="fp", lease_seconds=1)

    def test_low_level_ledger_heartbeat_and_not_expired_skip(self) -> None:
        now = 1.0
        store = InMemoryLedgerStore()
        ledger = Ledger(store, now=lambda: now)
        attempt = ledger.reserve_page(page_id="doc:9", provider_request_fingerprint="fp", worker_id="w", lease_seconds=10)
        heartbeat = ledger.heartbeat(attempt.attempt_id, worker_id="w", epoch=attempt.epoch, lease_seconds=20)
        self.assertEqual(heartbeat.lease_expires_at, now + 20)
        self.assertEqual(ledger.recover_expired_leases(now=now).requeued_pages, [])

    def test_remaining_provider_quality_resource_and_store_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            ProviderCapacityProfile("bad", 1, 1, 1, 0, 1, 1, 1, 1, 1)
        report = assess_markdown_fragment(("alpha beta gamma " * 20).strip())
        self.assertFalse(report.accepted)
        self.assertIn("repeated_ngram", [issue.code for issue in report.issues])
        self.assertTrue(assess_markdown_fragment("---").issues)
        self.assertTrue(assess_markdown_fragment("---\ntitle: ok\nbody").issues)
        governor = ResourceGovernor()
        with self.assertRaises(ResourceOrderViolation):
            governor._release("file_descriptor")
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemStateStore(Path(tmp), allow_weak_fs=False)
            with patch.object(os, "open", side_effect=OSError("strict fs")):
                with self.assertRaisesRegex(OSError, "strict fs"):
                    store._fsync_dir(Path(tmp))

class FinalCoverageGapTests(unittest.TestCase):
    def test_cli_cleanup_ignores_missing_temp_and_resume_success(self) -> None:
        from paperscale import cli

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out.md"
            with patch.object(cli.os, "replace", side_effect=RuntimeError("replace failed")), patch.object(
                cli.os, "unlink", side_effect=FileNotFoundError
            ):
                with self.assertRaisesRegex(RuntimeError, "replace failed"):
                    cli._atomic_write_text(output, "content")

        store = RecordingStateStore(records={"resume-index": {"pending": 1}})
        self.assertEqual(JobScheduler(store=store, queue_size=1).resume("job"), {"pending": 1})

    def test_mock_cli_reraises_unrelated_missing_dependency(self) -> None:
        import builtins

        from paperscale.mock_api import cli as mock_cli

        real_import = builtins.__import__

        def blocked_import(name: str, *args: Any, **kwargs: Any) -> object:
            if name == "uvicorn":
                raise ModuleNotFoundError("totally_missing")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=blocked_import):
            with self.assertRaises(ModuleNotFoundError):
                mock_cli.main(["serve"])

    def test_profile_and_provider_remaining_success_error_paths(self) -> None:
        profile = get_builtin_profile("generic_vlm_markdown")
        self.assertTrue(profile.parse_and_validate("one").ok)
        self.assertFalse(profile.parse_and_validate("a b a b a b a b").ok)

        glm = get_builtin_profile("glm_ocr")
        malformed = glm.parse_and_validate('{"markdown":')
        self.assertFalse(malformed.ok)
        self.assertEqual(malformed.retry_classification, "retryable")
        missing = glm.parse_and_validate('{"markdown":""}')
        self.assertFalse(missing.ok)
        non_string = glm.parse_and_validate('{"markdown": 3}')
        self.assertFalse(non_string.ok)

        from paperscale.providers.openai_chat import _extract_output_text

        self.assertEqual(_extract_output_text({"output_text": "# From dict"}), "# From dict")

    def test_quality_clean_long_text_exercises_ngram_continue_without_error(self) -> None:
        words = " ".join(f"word{i}" for i in range(24))
        report = assess_markdown_fragment(words)
        self.assertTrue(report.accepted)

    def test_strict_fsync_temp_cleanup_ignores_missing_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = FileSystemStateStore(Path(tmp))
            with patch.object(os, "replace", side_effect=RuntimeError("replace failed")), patch.object(
                os, "unlink", side_effect=FileNotFoundError
            ):
                with self.assertRaisesRegex(RuntimeError, "replace failed"):
                    store.write_json_atomic("x.json", {"x": 1})
