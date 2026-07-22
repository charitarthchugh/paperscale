"""`paperscale evaluate` — reference-free OCR-model comparison.

Reached via a shim in `pipeline.cli_main`: when the first CLI token is
``evaluate`` the pipeline delegates to `main` here. Heavy scoring deps
(wordfreq, rapidfuzz) are imported inside the handler so importing this module
stays cheap.
"""

from __future__ import annotations

import argparse
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


def _handle_evaluate(args: argparse.Namespace) -> int:
    from paperscale.evaluation.db import EvalDB
    from paperscale.evaluation.metrics import bow_f1, garbage_token_fraction, one_minus_ned
    from paperscale.evaluation.runs import load_run
    from paperscale.evaluation.spell import build_dictionary, correction_counts
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

            # Corrections — how much a spell checker must change the text (correctable)
            # and how much it cannot fix (uncorrectable). Shares pplx's dictionary.
            sym = build_dictionary(extra_words)
            ph = rep.phase("corrections", total=len(all_pages))
            correction_rows = []
            for p in all_pages:
                ph.advance()
                counts = correction_counts(p.text, sym)
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
            peer_rows = []
            for (doc, page), mt in page_texts.items():
                models_here = sorted(mt)
                if len(models_here) < 2:
                    continue
                for m in models_here:
                    for peer in models_here:
                        if m != peer:
                            peer_rows.append((m, peer, doc, page, bow_f1(mt[m], mt[peer]), one_minus_ned(mt[m], mt[peer])))
            if len(pages_by_model) < 2:
                rep.log("only one model run — peer agreement skipped (needs >=2 runs).")
            db.write_peer_agreement(peer_rows)
            ph.advance()
            ph.done()

            # Metric 4 — text-layer agreement (calibration subset).
            ph = rep.phase("text-layer", total=len(all_pages))
            textlayer_rows, skip = compute_textlayer_agreement(all_pages, all_metas, progress=lambda note: ph.advance())
            ph.done()
            db.write_textlayer_agreement(textlayer_rows)
            rep.log(
                f"text-layer: {len(textlayer_rows)} rows; skipped {skip.docs_with_fallback} fallback-docs, "
                f"{skip.docs_missing_pdf} missing-PDF docs, {skip.pages_blank_layer} blank-layer pages."
            )

            # Metric 5 — quality-reject rate (doc-level).
            db.write_reject_rate([(m.model, m.doc, m.fallback_pages, m.total_pages) for m in all_metas])

            # Optional — perplexity.
            if args.pplx:
                from paperscale.evaluation.pplx import score_run_pplx

                total_docs = sum(len({p.doc for p in pages}) for pages in pages_by_model.values())
                ph = rep.phase("perplexity", total=total_docs)
                for label, pages in pages_by_model.items():
                    by_doc = score_run_pplx(
                        pages,
                        pplx_url=args.pplx_url,
                        pplx_model=args.pplx_model,
                        extra_words=extra_words,
                        progress=lambda doc, label=label: (ph.advance(), rep.log(f"pplx {label}: {doc}")),
                    )
                    db.write_pplx(label, [row for rows in by_doc.values() for row in rows])
                ph.done()

        leaderboard = db.leaderboard()
    finally:
        db.close()
    print(leaderboard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
