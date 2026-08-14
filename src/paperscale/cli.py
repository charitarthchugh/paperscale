"""`paperscale evaluate` — reference-free OCR-model comparison.

Reached via a shim in `pipeline.cli_main`: when the first CLI token is
``evaluate`` the pipeline delegates to `main` here. Heavy scoring deps
(wordfreq, rapidfuzz) are imported inside the handler so importing this module
stays cheap.
"""

from __future__ import annotations

import argparse
import os
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
    ev.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="worker count for CPU-bound metrics and pdftotext (default: cpu count)")
    ev.add_argument("--pplx-concurrency", type=int, default=64, help="chunk requests in flight against the vLLM server (default 64)")
    ev.add_argument("--pplx-chunk-tokens", type=int, default=None, help="approx token cap per pplx request (default 32000); smaller prompts let vLLM batch far more of them, at the cost of cross-page conditioning at each chunk boundary")
    ev.add_argument("--no-resume", action="store_true", help="drop every cached score for these runs and rescore from scratch (default: reuse scores for docs whose text is unchanged)")
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


def _doc_checksums(pages: list) -> dict[str, str]:
    """``{doc: sha1}`` over each doc's page texts, in page order.

    This is what makes resume safe: re-running OCR under the same ``--run`` label
    changes the checksum of exactly the docs whose output changed, so those docs
    rescore and the rest are reused. Page numbers are hashed alongside the text so
    a re-paginated doc cannot collide with its old form.
    """
    import hashlib
    from collections import defaultdict as _dd

    by_doc: dict[str, list[tuple[int, str]]] = _dd(list)
    for p in pages:
        by_doc[p.doc].append((p.page, p.text))
    out: dict[str, str] = {}
    for doc, items in by_doc.items():
        h = hashlib.sha1()
        for page, text in sorted(items):
            h.update(f"{page}\0".encode())
            h.update(text.encode())
            h.update(b"\0")
        out[doc] = h.hexdigest()
    return out


def _log_resume(rep, phase: str, all_pages: list, todo_pages: list) -> None:
    """Say what resume skipped, so a fast run is never mistaken for a broken one."""
    skipped = len(all_pages) - len(todo_pages)
    if skipped:
        rep.log(f"{phase}: resuming — {skipped} pages already scored, {len(todo_pages)} to go.")


class _DocWriter:
    """Buffers one phase's rows and commits at each doc boundary.

    The pool.map streams stay in page order within a doc, so a change of doc means
    the previous one is complete: write its rows and mark it done in one
    transaction. An interrupted phase then keeps every doc that finished.
    """

    def __init__(self, db, phase: str, checksums: dict[str, dict[str, str]]) -> None:
        self.db, self.phase, self.checksums = db, phase, checksums
        self.key: tuple[str, str] | None = None
        self.buf: list[tuple] = []

    def add(self, model: str, doc: str, row: tuple | None = None) -> None:
        if self.key != (model, doc):
            self.flush()
            self.key = (model, doc)
        if row is not None:
            self.buf.append(row)

    def flush(self) -> None:
        if self.key is not None:
            model, doc = self.key
            self.db.write_doc(self.phase, model, doc, self.buf, self.checksums[model][doc])
        self.key, self.buf = None, []


def _handle_evaluate(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from paperscale.evaluation.db import EvalDB
    from paperscale.evaluation.metrics import (
        garbage_token_fraction,
        missing_peer_pairs,
        peer_rows_for_page,
    )
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

    from paperscale.tui import make_reporter

    db = EvalDB(args.db)
    leaderboard = ""
    try:
        with make_reporter(args.tui, title="paperscale evaluate") as rep:
            rep.set_stat("models", len(pages_by_model))
            rep.set_stat("documents", len({m.doc for m in all_metas}))
            rep.set_stat("pages", len(all_pages))

            ph = rep.phase("register runs", total=len(runs))
            for label, path in runs:
                db.register_run(label, path, args.pplx_model if args.pplx else None)
                ph.advance()
            ph.done()

            # --- resume bookkeeping ---------------------------------------
            # Every phase below skips the docs whose checksum already matches what
            # is on disk. --no-resume forgets each label first; docs that vanished
            # from a run are pruned so the leaderboard stops averaging them in.
            checksums = {label: _doc_checksums(pages) for label, pages in pages_by_model.items()}
            for label in pages_by_model:
                if args.no_resume:
                    db.clear_model(label)
                db.prune_missing_docs(label, set(checksums[label]))

            def todo(phase: str) -> list:
                """Pages whose doc still needs ``phase``, ordered so each doc's pages
                are contiguous -- _DocWriter commits on the doc boundary."""
                out = []
                for label, pages in pages_by_model.items():
                    done = db.done_docs(label, phase, checksums[label])
                    out.extend(p for p in pages if p.doc not in done)
                out.sort(key=lambda p: (p.model, p.doc, p.page))
                return out

            # CPU-bound metrics (symspell corrections, peer agreement) share one
            # process pool: pure Python => GIL-bound, threads would not help.
            # Workers build their own dictionary via the initializer; the main
            # thread keeps its own copy for pplx. All DB writes and Phase.advance()
            # stay on this thread -- workers only return plain tuples.
            sym = build_dictionary(extra_words)
            n_workers = max(1, min(args.jobs, len(all_pages) or 1))
            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=_pool_init, initargs=(extra_words,)
            ) as pool:
                # Corrections — how much a spell checker must change the text
                # (correctable) and how much it cannot fix (uncorrectable).
                corr_pages = todo("corrections")
                _log_resume(rep, "corrections", all_pages, corr_pages)
                ph = rep.phase("corrections", total=len(corr_pages))
                writer = _DocWriter(db, "corrections", checksums)
                counts_iter = pool.map(pool_correction_counts, [p.text for p in corr_pages], chunksize=32)
                for p, counts in zip(corr_pages, counts_iter):
                    ph.advance()
                    if counts is None:
                        writer.add(p.model, p.doc)  # still marks the doc seen
                        continue
                    n, corrected, uncorrectable = counts
                    writer.add(p.model, p.doc, (p.model, p.doc, p.page, corrected / n, uncorrectable / n))
                writer.flush()
                ph.done()

                # Garbage-token fraction + peer agreement.
                ph = rep.phase("token & peer metrics", total=2)
                gb_pages = todo("garbage")
                writer = _DocWriter(db, "garbage", checksums)
                for p in gb_pages:
                    s = garbage_token_fraction(p.text)
                    writer.add(p.model, p.doc, None if s is None else (p.model, p.doc, p.page, s))
                writer.flush()
                ph.advance()

                # Peer agreement resumes per model PAIR, not per doc: adding a third
                # run must fill in a-c and b-c while leaving a-b alone. The stored
                # rows are the bookkeeping; eval_doc only carries the checksum, so a
                # doc whose text changed has its pairs dropped first (both directions).
                stale_peer = []
                for label, sums in checksums.items():
                    fresh = db.done_docs(label, "peer", sums)
                    stale_peer += [(label, doc) for doc in sums if doc not in fresh]
                db.clear_peer_docs(stale_peer)
                page_texts: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
                for p in all_pages:
                    page_texts[(p.doc, p.page)][p.model] = p.text
                stored = db.stored_peer_pairs()
                items = []
                for (doc, page), by_model in page_texts.items():
                    if len(by_model) < 2:
                        continue
                    wanted = missing_peer_pairs(list(by_model), stored.get((doc, page), set()))
                    if wanted:
                        items.append(((doc, page), by_model, wanted))
                peer_rows = [r for rows in pool.map(peer_rows_for_page, items, chunksize=32) for r in rows]
                if len(pages_by_model) < 2:
                    rep.log("only one model run — peer agreement skipped (needs >=2 runs).")
                db.append_peer_agreement(peer_rows)
                db.mark_docs_done(
                    (label, doc, "peer", sha)
                    for label, sums in checksums.items()
                    for doc, sha in sums.items()
                )
                ph.advance()
                ph.done()

            # Metric 4 — text-layer agreement (calibration subset). The costliest
            # phase (a pdftotext subprocess per page), so it commits per doc via
            # on_doc rather than only at the end.
            tl_pages = todo("textlayer")
            tl_docs = {(p.model, p.doc) for p in tl_pages}
            tl_metas = [m for m in all_metas if (m.model, m.doc) in tl_docs]
            _log_resume(rep, "text-layer", all_pages, tl_pages)
            ph = rep.phase("text-layer", total=len(tl_pages))

            def commit_textlayer_doc(model: str, doc: str, rows: list) -> None:
                db.write_doc("textlayer", model, doc, rows, checksums[model][doc])

            textlayer_rows, skip = compute_textlayer_agreement(
                tl_pages, tl_metas, progress=lambda note: ph.advance(), jobs=args.jobs,
                on_doc=commit_textlayer_doc,
            )
            ph.done()
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

                pplx_todo: dict[str, list] = {}
                for label, pages in pages_by_model.items():
                    sums = checksums[label]
                    # Adoption: pplx resume predates eval_doc. Docs with scores but no
                    # checksum record were written by an older version -- adopt them at
                    # their current checksum rather than discard expensive GPU work.
                    # Anything that changes after this point invalidates normally.
                    for doc in db.pplx_done_docs(label) - db.marked_docs(label, "pplx"):
                        if doc in sums:
                            db.mark_doc_done(label, doc, "pplx", sums[doc])
                    done = db.done_docs(label, "pplx", sums) & db.pplx_done_docs(label)
                    pplx_todo[label] = [p for p in pages if p.doc not in done]
                    if done:
                        rep.log(f"pplx {label}: resuming — {len(done)} docs already scored, {len({p.doc for p in pplx_todo[label]})} to go.")
                total_docs = sum(len({p.doc for p in pages}) for pages in pplx_todo.values())
                ph = rep.phase("perplexity", total=total_docs)
                for label, pages in pplx_todo.items():
                    score_run_pplx(
                        pages,
                        pplx_url=args.pplx_url,
                        pplx_model=args.pplx_model,
                        extra_words=extra_words,
                        sym=sym,  # reuse the dictionary built for the correction metric
                        concurrency=args.pplx_concurrency,
                        chunk_tokens=args.pplx_chunk_tokens,
                        on_doc=lambda doc, rows, label=label: (
                            db.write_pplx_doc(label, rows),
                            db.mark_doc_done(label, doc, "pplx", checksums[label][doc]),
                        ),
                        progress=lambda doc, label=label: (ph.advance(), rep.log(f"pplx {label}: {doc}")),
                    )
                ph.done()

        leaderboard = db.leaderboard()
    finally:
        db.close()
    print(leaderboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
