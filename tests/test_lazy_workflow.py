"""Integration tests for the lazy plan->work candidate workflow (T8/T9/T10).

Covers: plan->work end-to-end; first-provider-before-last-materialize ordering
(a deterministic proxy for "work begins before the whole batch is materialized");
corrupt-PDF-at-work-time durable failure + drain-continues; ``--retry-failed``
clears the failed marker and re-attempts; done-precedence at work time; and a CLI
smoke test for the new ``plan`` command and ``work --retry-failed`` flag.

Reuses the ``_Renderer`` / ``_GoodProvider`` fake patterns from
``tests/test_concurrency_pool.py``.
"""

from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperscale.candidates import plan_candidates, read_candidates
from paperscale.providers.base import PageOcrResponse
from paperscale.runner import DocumentOcrRunner, RunnerConfig


@dataclass(frozen=True)
class _Rendered:
    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"


class _Renderer:
    def __init__(self, n: int) -> None:
        self._n = n

    @property
    def page_count(self) -> int:
        return self._n

    def render_page(self, page_number: int) -> _Rendered:
        image = f"page-{page_number}".encode()
        return _Rendered(page_number, image, hashlib.sha256(image).hexdigest())


class _GoodProvider:
    name = "good"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0

    def send(self, request: Any) -> PageOcrResponse:
        with self._lock:
            self.calls += 1
        return PageOcrResponse(
            markdown=f"# {request.page_id}\n\nBody for {request.page_id}.",
            provider_request_id=request.fingerprint,
        )


def _runner(root: Path, provider: Any, renderer_factory: Any, *, worker_id: str = "local") -> DocumentOcrRunner:
    return DocumentOcrRunner(
        RunnerConfig(
            state_root=root,
            base_url="http://fake/v1",
            model="m",
            worker_id=worker_id,
            max_in_flight_requests=4,
            server_max_num_seqs=4,
        ),
        provider=provider,
        renderer_factory=renderer_factory,
        sleeper=lambda _s: None,
    )


def _write_inputs(directory: Path, names: list[str]) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in names:
        path = directory / name
        path.write_text("%PDF-1.4 placeholder\n", encoding="utf-8")
        paths.append(path)
    return paths


def _plan(root: Path, inputs: list[Path], out_dir: Path) -> Path:
    return plan_candidates(
        inputs,
        output_dir=out_dir,
        state_root=root,
        profile="generic_vlm_markdown",
        model="m",
        base_url="http://fake/v1",
        capacity="local-vllm-small",
    )


class PlanWorkEndToEndTests(unittest.TestCase):
    def test_plan_then_work_completes_all_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["alpha.pdf", "beta.pdf", "gamma.pdf"])
            out_dir = base / "out"
            _plan(root, inputs, out_dir)

            provider = _GoodProvider()
            runner = _runner(root, provider, lambda _p, _o: _Renderer(2))
            statuses = runner.work()

            self.assertEqual(len(statuses), 3)
            self.assertTrue(all(s.complete for s in statuses))
            store = runner._claim_store()
            for job_id in ("alpha", "beta", "gamma"):
                self.assertTrue(store.is_done(job_id), f"{job_id} not done")
                self.assertTrue((out_dir / f"{job_id}.md").exists(), f"{job_id}.md missing")
            self.assertEqual(provider.calls, 6)  # 3 jobs x 2 pages


class OrderingTests(unittest.TestCase):
    """Deterministic perf proxy: the first provider call precedes the last materialize.

    If work begins only after the whole batch is materialized, the last
    ``materialize:`` entry would precede every ``provider:`` entry. Because work
    materializes-then-processes each candidate before moving to the next, job-1's
    provider calls land before job-N's materialize — proving work starts before the
    batch is fully materialized.
    """

    def test_first_provider_call_precedes_last_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["one.pdf", "two.pdf", "three.pdf"])
            out_dir = base / "out"
            _plan(root, inputs, out_dir)

            events: list[str] = []
            events_lock = threading.Lock()
            materialized: set[str] = set()

            class _OrderingProvider:
                name = "ord"

                def send(self, request: Any) -> PageOcrResponse:
                    job = request.page_id.split(":")[0]
                    with events_lock:
                        events.append(f"provider:{job}")
                    return PageOcrResponse(
                        markdown=f"# {request.page_id}\n\nbody",
                        provider_request_id=request.fingerprint,
                    )

            def renderer_factory(path: Path, _options: dict[str, Any]) -> _Renderer:
                job = Path(path).stem
                with events_lock:
                    if job not in materialized:
                        materialized.add(job)
                        events.append(f"materialize:{job}")
                return _Renderer(2)

            runner = _runner(root, _OrderingProvider(), renderer_factory)
            runner.work()

            provider_events = [i for i, e in enumerate(events) if e.startswith("provider:")]
            materialize_events = [i for i, e in enumerate(events) if e.startswith("materialize:")]
            self.assertTrue(provider_events, "no provider calls recorded")
            self.assertTrue(materialize_events, "no materialize events recorded")
            self.assertGreaterEqual(len(materialize_events), 2, "need >=2 candidates")
            self.assertLess(
                provider_events[0],
                materialize_events[-1],
                f"first provider call did not precede the last materialize: {events}",
            )


class CorruptPdfTests(unittest.TestCase):
    def test_corrupt_pdf_fails_but_drain_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["good1.pdf", "bad.pdf", "good2.pdf"])
            out_dir = base / "out"
            _plan(root, inputs, out_dir)

            def renderer_factory(path: Path, _options: dict[str, Any]) -> _Renderer:
                if Path(path).stem == "bad":
                    raise RuntimeError("corrupt PDF: cannot read page count")
                return _Renderer(2)

            runner = _runner(root, _GoodProvider(), renderer_factory)
            statuses = runner.work()

            store = runner._claim_store()
            self.assertTrue(store.is_failed("bad"))
            self.assertTrue((root / "jobs" / "bad" / "failed.json").exists())
            self.assertFalse(store.is_done("bad"))
            # The two healthy candidates still completed (drain not aborted).
            self.assertTrue(store.is_done("good1"))
            self.assertTrue(store.is_done("good2"))
            done_ids = {s.job_id for s in statuses if s.complete}
            self.assertEqual(done_ids, {"good1", "good2"})
            self.assertNotIn("bad", {s.job_id for s in statuses})


class RetryFailedTests(unittest.TestCase):
    def test_retry_failed_clears_marker_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["doc.pdf"])
            out_dir = base / "out"
            _plan(root, inputs, out_dir)

            flaky = {"corrupt": True}

            def renderer_factory(path: Path, _options: dict[str, Any]) -> _Renderer:
                if flaky["corrupt"]:
                    raise RuntimeError("corrupt PDF")
                return _Renderer(2)

            runner = _runner(root, _GoodProvider(), renderer_factory)
            runner.work()
            store = runner._claim_store()
            self.assertTrue(store.is_failed("doc"))

            # Render now works, but a plain work() leaves the failed job skipped.
            flaky["corrupt"] = False
            self.assertEqual(runner.work(), [])
            self.assertTrue(store.is_failed("doc"))
            self.assertFalse(store.is_done("doc"))

            # --retry-failed clears the marker and completes the job.
            statuses = runner.work(retry_failed=True)
            self.assertEqual(len(statuses), 1)
            self.assertTrue(statuses[0].complete)
            self.assertTrue(store.is_done("doc"))
            self.assertFalse(store.is_failed("doc"))
            self.assertTrue((out_dir / "doc.md").exists())


class DonePrecedenceTests(unittest.TestCase):
    def test_done_marker_short_circuits_even_with_failed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["pre.pdf"])
            out_dir = base / "out"
            _plan(root, inputs, out_dir)

            calls = {"render": 0}

            def renderer_factory(path: Path, _options: dict[str, Any]) -> _Renderer:
                calls["render"] += 1
                return _Renderer(2)

            runner = _runner(root, _GoodProvider(), renderer_factory)
            store = runner._claim_store()
            # Pre-write both markers: done precedence must win and skip processing.
            store.mark_done("pre")
            store.mark_failed("pre", "spurious")

            statuses = runner.work()
            self.assertEqual(statuses, [])
            self.assertEqual(calls["render"], 0, "done job was re-materialized/processed")


class CliSmokeTests(unittest.TestCase):
    def test_plan_cli_writes_candidates_and_work_parses(self) -> None:
        from paperscale.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / ".paperscale"
            inputs = _write_inputs(base / "in", ["x.pdf", "y.pdf"])
            list_file = base / "inputs.txt"
            list_file.write_text("\n".join(str(p) for p in inputs) + "\n", encoding="utf-8")
            out_dir = base / "out"

            rc = main([
                "plan",
                "--input-list", str(list_file),
                "--output-dir", str(out_dir),
                "--state-root", str(root),
                "--base-url", "http://x/v1",
                "--model", "m",
            ])
            self.assertEqual(rc, 0)
            manifests = list((root / "candidates").glob("*.jsonl"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(read_candidates(manifests[0])), 2)

            # work --retry-failed parses and runs cleanly against empty state.
            rc_work = main([
                "work",
                "--state-root", str(base / "empty-state"),
                "--retry-failed",
                "--quiet",
            ])
            self.assertEqual(rc_work, 0)

    def test_plan_cli_requires_input_list_and_output_dir(self) -> None:
        from paperscale.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(SystemExit):
                main([
                    "plan",
                    "--state-root", str(base / ".paperscale"),
                    "--base-url", "http://x/v1",
                    "--model", "m",
                ])


if __name__ == "__main__":
    unittest.main()
