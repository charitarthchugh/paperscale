"""`paperscale evaluate` — reference-free OCR-model comparison.

Reached via a shim in `pipeline.cli_main`: when the first CLI token is
``evaluate`` the pipeline delegates to `main` here. Heavy scoring deps
(wordfreq, rapidfuzz) are imported inside the handler so importing this module
stays cheap.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperscale")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ev = subparsers.add_parser(
        "evaluate",
        help="rank OCR models against each other from existing run outputs (no ground truth)",
        description="Corpus-level, reference-free leaderboard over one or more model runs.",
    )
    ev.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a model run: LABEL=PATH where PATH is a workspace dir, a dir of .jsonl, or a .jsonl file. Repeatable.",
    )
    ev.add_argument("--db", type=Path, default=Path("./evaluation.sqlite"), help="SQLite output path (default ./evaluation.sqlite)")
    ev.add_argument("--dictionary", action="append", type=Path, default=[], help="extra word-list file (one word per line/whitespace). Repeatable.")
    ev.add_argument("--pplx", action="store_true", help="also score perplexity via an external vLLM (raw + dictionary-corrected)")
    ev.add_argument("--pplx-url", default="http://localhost:8000", help="base URL of the vLLM OpenAI-compatible server")
    ev.add_argument("--pplx-model", default=None, help="model id served by --pplx-url (required with --pplx)")
    ev.add_argument("--tui", action="store_true", help="show a live progress dashboard (needs the 'tui' extra)")
    ev.add_argument("--tui-poll-interval", type=float, default=5.0, help="seconds between vLLM /metrics scrapes for the dashboard")
    ev.add_argument("--disk-logging", default=None, help="write the full log here (defaults beside --db when --tui is on)")
    ev.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="worker count for CPU-bound metrics and pdftotext (default: cpu count)")
    ev.add_argument("--pplx-concurrency", type=int, default=8, help="docs scored in flight against the vLLM server (default 8)")
    ev.add_argument("--no-resume", action="store_true", help="rescore pplx from scratch instead of resuming already-scored docs")
    ev.set_defaults(handler=_handle_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


def _parse_runs(entries: list[str]) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--run must be LABEL=PATH, got {entry!r}")
        label, path = entry.split("=", 1)
        label = label.strip()
        if not label or not path:
            raise SystemExit(f"--run must be LABEL=PATH, got {entry!r}")
        if label in seen:
            raise SystemExit(f"duplicate --run label {label!r}")
        seen.add(label)
        runs.append((label, path))
    return runs


def _load_dictionary(paths: list[Path]) -> frozenset[str]:
    words: set[str] = set()
    for p in paths:
        for token in p.read_text(encoding="utf-8").split():
            words.add(token.lower())
    return frozenset(words)


def _evaluate_log_path(db: Path) -> str:
    """Where evaluate's logs go when the dashboard owns the screen.

    evaluate has no workspace, so the database's directory is the natural home.
    """
    return str(db.resolve().parent / "logs" / f"evaluate-{os.getpid()}.log")


def _handle_evaluate(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from paperscale.evaluation.db import EvalDB
    from paperscale.evaluation.metrics import garbage_token_fraction, peer_rows_for_page
    from paperscale.evaluation.runs import load_run
    from paperscale.evaluation.spell import _pool_init, build_dictionary, pool_correction_counts
    from paperscale.evaluation.textlayer import compute_textlayer_agreement

    if args.pplx and not args.pplx_model:
        raise SystemExit("--pplx requires --pplx-model (the model id served by --pplx-url)")

    runs = _parse_runs(args.run)
    extra_words = _load_dictionary(args.dictionary)

    all_pages = []
    all_metas = []
    pages_by_model: dict[str, list] = {}
    for label, path in runs:
        pages, metas = load_run(label, path)
        pages_by_model[label] = pages
        all_pages.extend(pages)
        all_metas.extend(metas)

    from paperscale.tui import NullReporter, install_tui_logging, make_reporter, open_log_file, restore_console_logging
    from paperscale.vllm_stats import VLLMStats, VLLMStatsPoller, metrics_url, push_vllm_stats

    rep = make_reporter(args.tui, title="paperscale evaluate")
    live = not isinstance(rep, NullReporter)

    stats = poller = tui_handler = None
    if live:
        # The dashboard's own path unless the user named one, and either way
        # install_tui_logging is the only thing that opens it. The pipeline splits
        # this differently because it has already opened a user-supplied
        # --disk_logging by now and must not open a second handler on the same
        # file; here nothing has, and handing the path over is what keeps it
        # attached -- install strips *every* handler off the root logger, which is
        # where evaluate's file has to live, so a handler added before the call
        # would be taken straight back off for the length of the run.
        args.disk_logging = args.disk_logging or _evaluate_log_path(args.db)
        tui_handler = install_tui_logging(rep, args.disk_logging)
        if args.pplx:
            stats = VLLMStats()
            poller = VLLMStatsPoller(metrics_url(args.pplx_url), stats, interval=args.tui_poll_interval)
            poller.start()
    elif args.disk_logging:
        # No frame to protect, but the flag still has to write the file. evaluate
        # owns no logger of its own, so root -- where everything it and its
        # dependencies emit ends up -- is the only place to attach it.
        logging.getLogger().addHandler(open_log_file(args.disk_logging))

    db = EvalDB(args.db)
    leaderboard = ""
    try:
        with rep:
            rep.set_stat("models", len(pages_by_model))
            rep.set_stat("documents", len({m.doc for m in all_metas}))
            rep.set_stat("pages", len(all_pages))

            ph = rep.phase("register runs", total=len(runs))
            for label, path in runs:
                db.register_run(label, path, args.pplx_model if args.pplx else None)
                ph.advance()
            ph.done()

            # CPU-bound metrics (symspell corrections, peer agreement) share one
            # process pool: pure Python => GIL-bound, threads would not help.
            # Workers build their own dictionary via the initializer; the main
            # thread keeps its own copy for pplx. All DB writes and Phase.advance()
            # stay on this thread -- workers only return plain tuples.
            sym = build_dictionary(extra_words)
            n_workers = max(1, min(args.jobs, len(all_pages) or 1))
            with ProcessPoolExecutor(max_workers=n_workers, initializer=_pool_init, initargs=(extra_words,)) as pool:
                # Corrections — how much a spell checker must change the text
                # (correctable) and how much it cannot fix (uncorrectable).
                ph = rep.phase("corrections", total=len(all_pages))
                correction_rows = []
                counts_iter = pool.map(pool_correction_counts, [p.text for p in all_pages], chunksize=32)
                for p, counts in zip(all_pages, counts_iter):
                    ph.advance()
                    if counts is None:
                        continue
                    n, corrected, uncorrectable = counts
                    correction_rows.append((p.model, p.doc, p.page, corrected / n, uncorrectable / n))
                db.write_correction_rate(correction_rows)
                ph.done()

                # Garbage-token fraction + peer agreement.
                ph = rep.phase("token & peer metrics", total=2)
                garbage_rows = [(p.model, p.doc, p.page, s) for p in all_pages if (s := garbage_token_fraction(p.text)) is not None]
                db.write_garbage_fraction(garbage_rows)
                ph.advance()
                page_texts: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
                for p in all_pages:
                    page_texts[(p.doc, p.page)][p.model] = p.text
                items = [it for it in page_texts.items() if len(it[1]) >= 2]
                peer_rows = [r for rows in pool.map(peer_rows_for_page, items, chunksize=32) for r in rows]
                if len(pages_by_model) < 2:
                    rep.log("only one model run — peer agreement skipped (needs >=2 runs).")
                db.write_peer_agreement(peer_rows)
                ph.advance()
                ph.done()

            # Metric 4 — text-layer agreement (calibration subset).
            ph = rep.phase("text-layer", total=len(all_pages))
            textlayer_rows, skip = compute_textlayer_agreement(all_pages, all_metas, progress=lambda note: ph.advance(), jobs=args.jobs)
            ph.done()
            db.write_textlayer_agreement(textlayer_rows)
            rep.log(
                f"text-layer: {len(textlayer_rows)} rows; skipped {skip.docs_with_fallback} fallback-docs, "
                f"{skip.docs_missing_pdf} missing-PDF docs, {skip.pages_blank_layer} blank-layer pages."
            )

            # Metric 5 — quality-reject rate (doc-level).
            db.write_reject_rate([(m.model, m.doc, m.fallback_pages, m.total_pages) for m in all_metas])

            # Optional — perplexity. Resumes by default: docs already scored (and
            # committed doc-by-doc via on_doc) are skipped on re-run; --no-resume
            # drops prior scores. A scorer-model change invalidates them automatically
            # (see EvalDB.register_run).
            if args.pplx:
                from paperscale.evaluation.pplx import score_run_pplx

                todo: dict[str, list] = {}
                for label, pages in pages_by_model.items():
                    if args.no_resume:
                        db.clear_pplx(label)
                    done = db.pplx_done_docs(label)
                    todo[label] = [p for p in pages if p.doc not in done]
                    if done:
                        rep.log(f"pplx {label}: resuming — {len(done)} docs already scored, {len({p.doc for p in todo[label]})} to go.")
                total_docs = sum(len({p.doc for p in pages}) for pages in todo.values())
                ph = rep.phase("perplexity", total=total_docs)
                # Docs a resume already scored: constant for the phase, so it is
                # written once rather than re-pushed per doc.
                rep.set_stat("skipped", sum(len({p.doc for p in pages}) for pages in pages_by_model.values()) - total_docs, group="issues")
                # Seed the panel before the first doc lands -- scoring a document
                # can take minutes, and an empty vllm column reads as a broken one.
                push_vllm_stats(rep, stats, poller)

                def scored(doc: str, label: str, ph=ph) -> None:
                    ph.advance()
                    rep.log(f"pplx {label}: {doc}")
                    push_vllm_stats(rep, stats, poller)

                for label, pages in todo.items():
                    score_run_pplx(
                        pages,
                        pplx_url=args.pplx_url,
                        pplx_model=args.pplx_model,
                        extra_words=extra_words,
                        sym=sym,  # reuse the dictionary built for the correction metric
                        concurrency=args.pplx_concurrency,
                        on_doc=lambda doc, rows, label=label: db.write_pplx_doc(label, rows),
                        progress=lambda doc, label=label: scored(doc, label),
                    )
                ph.done()

        leaderboard = db.leaderboard()
    finally:
        db.close()
        if poller is not None:
            poller.stop()
        if tui_handler is not None:
            restore_console_logging(tui_handler)
    print(leaderboard)
    # The alternate screen is gone by now, so this lands on the real screen --
    # without it a --tui run leaves no trace of where its log went.
    if live and args.disk_logging:
        print(f"Full log: {args.disk_logging}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
