#!/usr/bin/env python
"""Load-test paperscale against a real vLLM server using the multi-process work queue.

Pre-enqueues N PDFs as jobs, then launches P concurrent ``paperscale work`` processes
(each claiming jobs via the O_EXCL ClaimStore and running pages through the asyncio
pool). Reports throughput (pages/sec) and per-page provider latency percentiles read
from the per-attempt ledger (the durable truth).

Usage:
  poetry run python scripts/loadtest.py \
      --pdf-dir /run/media/cc/data/law/pdfs --count 12 --processes 3 \
      --base-url http://127.0.0.1:8000 --model deepseek-ai/DeepSeek-OCR-2 \
      --profile deepseek_ocr_2 --max-in-flight 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_pdfs(pdf_dir: Path, count: int) -> list[Path]:
    # Over-collect so we can skip unreadable/corrupt PDFs and still hit `count`.
    found: list[Path] = []
    for path in pdf_dir.rglob("*.pdf"):
        found.append(path)
        if len(found) >= count * 4:
            break
    return found


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _collect_metrics(state_root: Path) -> dict[str, object]:
    succeeded = 0
    terminal = 0
    pages_total = 0
    latencies: list[float] = []
    jobs_dir = state_root / "jobs"
    for job_dir in jobs_dir.iterdir() if jobs_dir.exists() else []:
        status_path = job_dir / "indexes" / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            succeeded += int(status.get("succeeded", 0))
            terminal += int(status.get("failed_terminal", 0))
            pages_total += int(status.get("pages_total", 0))
        ledger_dir = job_dir / "ledger"
        if ledger_dir.exists():
            for rec_path in ledger_dir.glob("*.json"):
                rec = json.loads(rec_path.read_text())
                started = rec.get("provider_started_at")
                committed = rec.get("provider_response_committed_at")
                if isinstance(started, (int, float)) and isinstance(committed, (int, float)):
                    latencies.append(float(committed) - float(started))
    return {
        "succeeded": succeeded,
        "failed_terminal": terminal,
        "pages_total": pages_total,
        "provider_calls_committed": len(latencies),
        "latency_p50_s": round(_percentile(latencies, 50), 3),
        "latency_p95_s": round(_percentile(latencies, 95), 3),
        "latency_max_s": round(max(latencies), 3) if latencies else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR-2")
    parser.add_argument("--profile", default="deepseek_ocr_2")
    parser.add_argument("--max-in-flight", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--server-max-num-seqs", type=int, default=32)
    parser.add_argument("--state-root", type=Path, default=Path("/tmp/ps-loadtest/.paperscale"))
    parser.add_argument("--keep-state", action="store_true")
    args = parser.parse_args(argv)

    if not args.keep_state and args.state_root.exists():
        shutil.rmtree(args.state_root)
    args.state_root.mkdir(parents=True, exist_ok=True)
    out_dir = args.state_root.parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _find_pdfs(args.pdf_dir, args.count)
    if not pdfs:
        print(f"no PDFs found under {args.pdf_dir}", file=sys.stderr)
        return 1

    # Enqueue jobs in-process (front door for the work queue).
    from paperscale.runner import DocumentOcrRunner, RunnerConfig

    enqueuer = DocumentOcrRunner(
        RunnerConfig(
            state_root=args.state_root, base_url=args.base_url, model=args.model,
            profile=args.profile, server_max_num_seqs=args.server_max_num_seqs,
        )
    )
    enqueued = 0
    for pdf in pdfs:
        if enqueued >= args.count:
            break
        try:
            enqueuer.enqueue(input_path=pdf, output_path=out_dir / f"job-{enqueued}.md", job_id=f"job-{enqueued}")
            enqueued += 1
        except Exception as exc:  # noqa: BLE001 - skip corrupt/unreadable PDFs
            print(f"skip {pdf.name}: {exc}")
    print(f"enqueued {enqueued} jobs; launching {args.processes} work processes...")

    # `work` reads base_url/model/profile from each job's manifest, so it only needs
    # the state root, worker identity, and the concurrency knobs.
    cmd = [
        sys.executable, "-m", "paperscale.cli", "work",
        "--state-root", str(args.state_root),
        "--server-max-num-seqs", str(args.server_max_num_seqs),
        "--max-in-flight-requests", str(args.max_in_flight),
        "--max-attempts", str(args.max_attempts),
    ]
    start = time.monotonic()
    procs = [
        subprocess.Popen([*cmd, "--worker-id", f"w{i}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for i in range(args.processes)
    ]
    for proc in procs:
        out, _ = proc.communicate()
        sys.stdout.write(out or "")
    wall = time.monotonic() - start

    metrics = _collect_metrics(args.state_root)
    pages_per_sec = round(int(metrics["succeeded"]) / wall, 2) if wall > 0 else 0.0
    print("\n==== load test results ====")
    print(json.dumps({**metrics, "wall_seconds": round(wall, 2), "pages_per_sec": pages_per_sec,
                      "processes": args.processes, "max_in_flight": args.max_in_flight}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
