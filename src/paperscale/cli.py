"""Paperscale command line interface for v1 document-to-Markdown OCR workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from paperscale.assembly import PageMarkdownArtifact, assemble_document_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperscale",
        description="Local-first VLM OCR runner for public v1 document-to-Markdown workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="assemble completed page OCR Markdown artifacts into a document",
        description="Assemble completed page OCR Markdown artifacts into one document-to-Markdown output.",
    )
    assemble_parser.add_argument("--input", required=True, type=Path, help="JSONL file of completed page artifacts")
    assemble_parser.add_argument("--output", required=True, type=Path, help="Markdown output path")
    assemble_parser.add_argument("--title", help="optional document title to prepend as an H1")
    assemble_parser.add_argument(
        "--enforce-quality",
        action="store_true",
        help="reject empty, mojibake, or repeated OCR fragments before writing output",
    )
    assemble_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="mark the assembled Markdown as partial",
    )
    assemble_parser.set_defaults(handler=_handle_assemble)

    run_parser = subparsers.add_parser("run", help="run page OCR for a document-to-Markdown workload")
    _add_runner_options(run_parser, include_job_id=True)
    _add_input_options(run_parser)
    run_parser.add_argument("--allow-partial", action="store_true", help="assemble succeeded pages even if some pages fail")
    _add_quiet_option(run_parser)
    run_parser.set_defaults(handler=_handle_run)

    enqueue_parser = subparsers.add_parser("enqueue", help="register job(s) for later `work` processing (does not run them)")
    _add_runner_options(enqueue_parser, include_job_id=True)
    _add_input_options(enqueue_parser)
    enqueue_parser.set_defaults(handler=_handle_enqueue)

    plan_parser = subparsers.add_parser(
        "plan",
        help="plan a batch into one immutable candidate manifest for lazy `work` materialization",
    )
    _add_runner_options(plan_parser, include_job_id=False)
    _add_input_options(plan_parser)
    plan_parser.set_defaults(handler=_handle_plan)

    status_parser = subparsers.add_parser("status", help="show compact-index workload status")
    _add_state_job_options(status_parser)
    status_parser.add_argument("--json", action="store_true", help="emit JSON")
    status_parser.set_defaults(handler=_handle_status)

    resume_parser = subparsers.add_parser("resume", help="resume pending work using compact indexes")
    _add_state_job_options(resume_parser)
    _add_recovery_concurrency_options(resume_parser)
    resume_parser.add_argument("--retry-ambiguous", action="store_true", help="retry ambiguous in-flight attempts")
    resume_parser.add_argument("--allow-partial", action="store_true", help="assemble succeeded pages even if some pages fail")
    _add_quiet_option(resume_parser)
    resume_parser.set_defaults(handler=_handle_resume)

    work_parser = subparsers.add_parser(
        "work", help="claim and run available incomplete jobs (single machine, multi-process)"
    )
    work_parser.add_argument("--state-root", default=Path(".paperscale"), type=Path, help="state root directory")
    work_parser.add_argument("--worker-id", default="local", help="worker identity stamped on claims/attempts")
    work_parser.add_argument("--max-jobs", type=int, default=None, help="stop after claiming this many jobs")
    work_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="clear failed markers and re-attempt failed candidates",
    )
    _add_recovery_concurrency_options(work_parser)
    _add_quiet_option(work_parser)
    work_parser.set_defaults(handler=_handle_work)

    reconcile_parser = subparsers.add_parser("reconcile", help="surface ambiguous attempts and duplicate-call risk")
    _add_state_job_options(reconcile_parser)
    reconcile_parser.add_argument("--json", action="store_true", help="emit JSON")
    resolve_group = reconcile_parser.add_mutually_exclusive_group()
    resolve_group.add_argument(
        "--supersede", type=int, metavar="PAGE", help="mark the ambiguous attempt superseded and requeue the page as pending"
    )
    resolve_group.add_argument(
        "--accept", type=int, metavar="PAGE", help="accept the page's existing artifact and mark it succeeded"
    )
    reconcile_parser.set_defaults(handler=_handle_reconcile)

    fsck_parser = subparsers.add_parser("fsck", help="scan-only filesystem consistency check")
    _add_state_job_options(fsck_parser)
    fsck_parser.set_defaults(handler=_handle_fsck)

    repair_parser = subparsers.add_parser("repair-index", help="explicit scan path that rebuilds compact indexes")
    _add_state_job_options(repair_parser)
    repair_parser.set_defaults(handler=_handle_repair_index)

    doctor_parser = subparsers.add_parser("doctor", help="diagnose provider configuration")
    doctor_subparsers = doctor_parser.add_subparsers(dest="doctor_command", required=True)
    provider_parser = doctor_subparsers.add_parser(
        "provider",
        help="validate provider reachability and OCR profile compatibility",
        description="Validate provider reachability and OCR profile compatibility without starting OCR work.",
    )
    provider_parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL; /v1 is appended when omitted")
    provider_parser.add_argument("--model", required=True, help="served model id to validate")
    provider_parser.add_argument("--capacity", default="local-vllm-small", help="capacity profile")
    provider_parser.add_argument("--profile", default="generic_vlm_markdown", help="OCR profile")
    provider_parser.set_defaults(handler=_handle_provider_doctor)

    return parser


def _add_runner_options(parser: argparse.ArgumentParser, *, include_job_id: bool) -> None:
    if include_job_id:
        parser.add_argument("--job-id", help="workload identifier")
    parser.add_argument("--state-root", default=Path(".paperscale"), type=Path, help="state root directory")
    parser.add_argument("--profile", default="generic_vlm_markdown", help="OCR profile")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL; /v1 is appended when omitted")
    parser.add_argument("--model", help="served model id")
    parser.add_argument("--capacity", default="local-vllm-small", help="capacity profile")
    _add_recovery_concurrency_options(parser)


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    """Single-document (--input/--output) or batch (--input-list/--output-dir) inputs.

    --input-list points at a text file with one document path per line ('-' reads
    stdin); blank lines and '#' comments are ignored. Validation that exactly one
    mode is supplied happens in the handler so the error message is actionable.
    """
    parser.add_argument("--input", type=Path, help="PDF input path (single document)")
    parser.add_argument("--output", type=Path, help="Markdown output path (single document)")
    parser.add_argument(
        "--input-list",
        type=str,
        metavar="FILE",
        help="text file with one document path per line ('-' for stdin); one job per line",
    )
    parser.add_argument("--output-dir", type=Path, help="directory for batch Markdown outputs (<job-id>.md)")


def _add_quiet_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress/throughput logging (errors still shown); logging is on by default",
    )


def _read_path_list(source: str) -> list[Path]:
    import sys

    if source == "-":
        text = sys.stdin.read()
    else:
        text = Path(source).read_text(encoding="utf-8")
    paths: list[Path] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        paths.append(Path(stripped))
    return paths


def _add_recovery_concurrency_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metered",
        action="store_true",
        help="treat the provider as billable/non-idempotent: escalate crashed in-flight pages to ambiguous (not auto-requeue). NOT persisted; pass it on every invocation.",
    )
    parser.add_argument("--max-attempts", type=int, default=8, help="hard per-page retry cap before terminal demotion")
    parser.add_argument(
        "--server-max-num-seqs", type=int, default=128,
        help="server's --max-num-seqs; max_in_flight defaults to 4x this",
    )
    parser.add_argument("--max-in-flight-requests", type=int, default=None, help="process-wide cap on requests sent to the server")
    parser.add_argument("--max-open-documents", type=int, default=128, help="cap on concurrently open PDF handles")
    parser.add_argument("--render-ahead", type=int, default=None, help="bounded decoded-image prefetch (pages)")


def _add_state_job_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id", help="workload identifier")
    parser.add_argument("--state-root", default=Path(".paperscale"), type=Path, help="state root directory")

def main(argv: Sequence[str] | None = None) -> int:
    from paperscale.observability import configure_logging

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(quiet=bool(getattr(args, "quiet", False)))
    handler = args.handler
    return int(handler(args))


def _handle_assemble(args: argparse.Namespace) -> int:
    pages = _read_page_artifacts(args.input)
    markdown = assemble_document_markdown(
        pages,
        title=args.title,
        enforce_quality=args.enforce_quality,
        partial=args.allow_partial,
    )
    _atomic_write_text(args.output, markdown)
    print(f"assembled {len(pages)} pages into {args.output}")
    return 0


def _runner_from_args(args: argparse.Namespace):
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    return DocumentOcrRunner(
        RunnerConfig(
            state_root=args.state_root,
            profile=getattr(args, "profile", "generic_vlm_markdown"),
            base_url=getattr(args, "base_url", ""),
            model=getattr(args, "model", None),
            capacity=getattr(args, "capacity", "local-vllm-small"),
            worker_id=getattr(args, "worker_id", "local"),
            metered=bool(getattr(args, "metered", False)),
            max_attempts=int(getattr(args, "max_attempts", 8)),
            server_max_num_seqs=int(getattr(args, "server_max_num_seqs", 128)),
            max_in_flight_requests=getattr(args, "max_in_flight_requests", None),
            max_open_documents=int(getattr(args, "max_open_documents", 128)),
            render_ahead=getattr(args, "render_ahead", None),
        )
    )


def _print_run_status(status) -> int:
    """Print a status line and pick the exit code, distinguishing settled-with-failures."""
    print(f"job {status.job_id}: succeeded={status.succeeded}/{status.pages_total} output={status.output_path}")
    if status.complete:
        return 0
    if status.settled and status.has_terminal_failures:
        # Reporting-only: resume cannot progress. Make the message actionable.
        print(
            f"job {status.job_id}: settled with {status.failed_terminal} terminal failure(s); "
            f"resume will not progress — pass --allow-partial to assemble "
            f"{status.succeeded}/{status.pages_total}, or re-source the failed page(s)."
        )
        return 2
    if status.succeeded > 0 and status.partial:
        return 0
    return 1


def _validate_input_mode(args: argparse.Namespace) -> str:
    """Return 'single' or 'batch'; exit with a clear message if the mode is ambiguous."""
    single = args.input is not None or args.output is not None
    batch = args.input_list is not None
    if batch and single:
        raise SystemExit("error: use either --input/--output or --input-list/--output-dir, not both")
    if batch:
        if args.output_dir is None:
            raise SystemExit("error: --input-list requires --output-dir")
        return "batch"
    if args.input is not None and args.output is not None:
        return "single"
    raise SystemExit("error: provide --input and --output, or --input-list and --output-dir")


def _enqueue_batch(runner, args: argparse.Namespace) -> tuple[list[str], list[tuple]]:
    inputs = _read_path_list(args.input_list)
    enqueued, skipped = runner.enqueue_many(inputs, output_dir=args.output_dir)
    for path, reason in skipped:
        print(f"skipped {path}: {reason}")
    print(f"enqueued {len(enqueued)} job(s)" + (f", skipped {len(skipped)}" if skipped else ""))
    return enqueued, skipped


def _handle_run(args: argparse.Namespace) -> int:
    runner = _runner_from_args(args)
    if _validate_input_mode(args) == "batch":
        enqueued, _skipped = _enqueue_batch(runner, args)
        if not enqueued:
            return 1
        statuses = runner.work()
        for status in statuses:
            _print_run_status(status)
        return 0 if all(s.complete for s in statuses) else 1
    status = runner.run(input_path=args.input, output_path=args.output, job_id=args.job_id, allow_partial=args.allow_partial)
    return _print_run_status(status)


def _handle_enqueue(args: argparse.Namespace) -> int:
    runner = _runner_from_args(args)
    if _validate_input_mode(args) == "batch":
        enqueued, _skipped = _enqueue_batch(runner, args)
        return 0 if enqueued else 1
    job_id = runner.enqueue(input_path=args.input, output_path=args.output, job_id=args.job_id)
    print(f"enqueued job {job_id}")
    return 0


def _handle_work(args: argparse.Namespace) -> int:
    from paperscale.observability import get_logger

    runner = _runner_from_args(args)
    statuses = runner.work(
        max_jobs=getattr(args, "max_jobs", None),
        retry_failed=bool(getattr(args, "retry_failed", False)),
    )
    if not statuses:
        get_logger().info("no claimable jobs")
        return 0
    for status in statuses:
        state = "complete" if status.complete else ("settled" if status.settled else "in-progress")
        print(f"job {status.job_id}: {state} succeeded={status.succeeded}/{status.pages_total}")
    return 0 if all(s.complete for s in statuses) else 1


def _handle_plan(args: argparse.Namespace) -> int:
    from paperscale.candidates import plan_candidates, read_candidates

    if args.input_list is None or args.output_dir is None:
        raise SystemExit("error: plan requires --input-list and --output-dir")
    inputs = _read_path_list(args.input_list)
    target = plan_candidates(
        inputs,
        output_dir=args.output_dir,
        state_root=args.state_root,
        profile=args.profile,
        model=args.model,
        base_url=args.base_url,
        capacity=args.capacity,
    )
    print(f"planned {len(read_candidates(target))} candidate(s) into {target}")
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    status = DocumentOcrRunner(RunnerConfig(state_root=args.state_root)).status(args.job_id)
    if args.json:
        print(json.dumps(status.to_json_summary(), sort_keys=True))
    else:
        print(f"job {status.job_id}: succeeded={status.succeeded}/{status.pages_total} pending={status.pending} failed_retryable={status.failed_retryable} ambiguous={status.ambiguous}")
    return 0


def _handle_resume(args: argparse.Namespace) -> int:
    runner = _runner_from_args(args)
    status = runner.resume(args.job_id, retry_ambiguous=args.retry_ambiguous, allow_partial=args.allow_partial)
    return _print_run_status(status)


def _handle_reconcile(args: argparse.Namespace) -> int:
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    runner = DocumentOcrRunner(RunnerConfig(state_root=args.state_root))
    if args.supersede is not None:
        status = runner.resolve_ambiguous(args.job_id, args.supersede, action="supersede")
        print(f"job {status.job_id}: page {args.supersede} superseded; succeeded={status.succeeded}/{status.pages_total} pending={status.pending}")
        return 0
    if args.accept is not None:
        status = runner.resolve_ambiguous(args.job_id, args.accept, action="accept")
        print(f"job {status.job_id}: page {args.accept} accepted; succeeded={status.succeeded}/{status.pages_total}")
        return 0
    payload = runner.reconcile(args.job_id)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        ambiguous = payload.get("ambiguous_attempts", [])
        print(format_ambiguous_attempts(count=len(ambiguous), page_sample=[str(item.get("page_number")) for item in ambiguous[:5]]))
    return 0


def _handle_fsck(args: argparse.Namespace) -> int:
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    payload = DocumentOcrRunner(RunnerConfig(state_root=args.state_root)).fsck(args.job_id)
    print(json.dumps(payload, sort_keys=True))
    return 0 if not payload.get("issues") else 1


def _handle_repair_index(args: argparse.Namespace) -> int:
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    status = DocumentOcrRunner(RunnerConfig(state_root=args.state_root)).repair_index(args.job_id)
    print(f"job {status.job_id}: rebuilt indexes succeeded={status.succeeded}/{status.pages_total}")
    return 0


def _handle_provider_doctor(args: argparse.Namespace) -> int:
    from paperscale.runner import doctor_provider

    payload = doctor_provider(base_url=args.base_url, model=args.model, capacity=args.capacity, profile=args.profile)
    print(f"endpoint: {payload['endpoint']}")
    print(f"observed_models: {', '.join(payload['observed_models']) or '<none>'}")
    print(f"ocr_profile: {payload['ocr_profile']}")
    print(f"capacity_profile: {payload['capacity_profile']}")
    print(f"compatible: {str(payload['compatible']).lower()}")
    print(f"diagnostic: {payload['diagnostic']}")
    return 0 if payload["compatible"] else 1


def _read_page_artifacts(path: Path) -> list[PageMarkdownArtifact]:
    pages: list[PageMarkdownArtifact] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            pages.append(_page_from_json(payload, line_number=line_number))
    return pages


def _page_from_json(payload: dict[str, Any], *, line_number: int) -> PageMarkdownArtifact:
    try:
        document_id = payload["document_id"]
        page_number = payload["page_number"]
        markdown = payload["markdown"]
    except KeyError as exc:
        raise ValueError(f"page artifact line {line_number} is missing required field {exc.args[0]!r}") from exc

    if not isinstance(document_id, str) or not document_id:
        raise ValueError(f"page artifact line {line_number} has invalid document_id")
    if not isinstance(page_number, int):
        raise ValueError(f"page artifact line {line_number} has invalid page_number")
    if not isinstance(markdown, str):
        raise ValueError(f"page artifact line {line_number} has invalid markdown")
    return PageMarkdownArtifact(document_id=document_id, page_number=page_number, markdown=markdown)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
        _fsync_parent(path.parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_parent(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())


class CliApp:
    """Injectable CLI facade used by tests and future command wiring."""

    def __init__(self, *, store) -> None:
        self.store = store

    def run(self, argv: Sequence[str]) -> int:
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.command == "status":
            self.store.read_index("job-index")
            return 0
        if args.command == "repair-index":
            self.store.scan_tree(getattr(args, "job_id", ".") or ".")
            return 0
        return 0


def format_ambiguous_attempts(*, count: int, page_sample: list[str]) -> str:
    sample = ", ".join(page_sample)
    return (
        f"{count} ambiguous OCR attempts require operator reconciliation. "
        f"Sample pages: {sample}. Retrying can create duplicate provider calls; "
        "use --retry-ambiguous only after accepting that risk or confirming idempotency. "
        "Per page, resolve with `reconcile --supersede PAGE` (discard the uncertain attempt and requeue) "
        "or `reconcile --accept PAGE` (keep the page's already-written artifact)."
    )
