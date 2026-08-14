"""Reference-free text-layer agreement: how well each model's OCR of a page
agrees with the PDF's embedded text layer (pdftotext).

Only a calibration signal: docs with fallback pages (whose natural_text is
already filled with pdftotext output) are skipped, as are docs whose PDF is
missing and pages whose text layer is effectively blank (scanned images).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from paperscale.anchor import get_anchor_text
from paperscale.evaluation.metrics import bow_f1, one_minus_ned
from paperscale.evaluation.runs import DocMeta, PageText

# Below this many alphanumeric chars the "text layer" is a scanned image, not real text.
_MIN_ALNUM = 25


@dataclass
class SkipReport:
    docs_missing_pdf: int
    docs_with_fallback: int
    pages_blank_layer: int


def _pdf_exists(path: str) -> bool:
    # Patchable seam for tests.
    return os.path.exists(path)


def _extract_layer(doc: str, page: int) -> str:
    """pdftotext, falling back to pypdf if poppler is unavailable/errors."""
    try:
        return get_anchor_text(doc, page, "pdftotext")
    except (FileNotFoundError, AssertionError, OSError):
        return get_anchor_text(doc, page, "pypdf")


def compute_textlayer_agreement(
    pages: list[PageText], metas: list[DocMeta], progress=None, jobs: int | None = None,
    on_doc=None,
) -> tuple[list[tuple[str, str, int, float, float]], SkipReport]:
    """``progress``: optional callable(note) invoked once per page (instrumentation only).

    ``jobs``: thread-pool width for the pdftotext subprocess calls (I/O-bound,
    GIL released while blocked on the child process).

    ``on_doc``: optional callable(model, doc, rows) invoked as each doc finishes, so
    the caller can commit per doc and resume an interrupted phase. It fires for
    skipped docs too (with no rows): the doc IS complete, and a caller that only
    heard about row-producing docs would re-attempt the skipped ones every run.
    """
    fallback = {(m.model, m.doc): m.fallback_pages for m in metas}
    report = SkipReport(docs_missing_pdf=0, docs_with_fallback=0, pages_blank_layer=0)

    skip_docs: set[tuple[str, str]] = set()
    for (model, doc), fb in fallback.items():
        # Skip the WHOLE doc if any page fell back: fallback pages fill natural_text with
        # pdftotext output, so comparing them to the text layer is circular (~1.0). This
        # sacrifices the doc's good pages too -- acceptable since this metric is only a
        # calibration subset, not full coverage.
        if fb > 0:
            report.docs_with_fallback += 1
            skip_docs.add((model, doc))
        elif not _pdf_exists(doc):
            report.docs_missing_pdf += 1
            skip_docs.add((model, doc))

    candidates = [pg for pg in pages if (pg.model, pg.doc) not in skip_docs]
    rows: list[tuple[str, str, int, float, float]] = []
    # Threads: the work is a blocking pdftotext subprocess per page (GIL released),
    # and thread pools keep the tests' monkeypatched module seams visible.
    # ex.map preserves candidate (= page) order, so rows/progress/counters stay
    # deterministic on this (the main) thread.
    workers = max(1, jobs or min(32, (os.cpu_count() or 4) * 4))
    # Rows of the doc currently being consumed, flushed to on_doc at each boundary.
    # `pages` is grouped by doc in practice (load_run emits a doc's pages together);
    # a doc split across non-adjacent runs would simply flush more than once, which
    # write_doc tolerates because it replaces that doc's rows wholesale.
    doc_key: tuple[str, str] | None = None
    doc_rows: list[tuple[str, str, int, float, float]] = []

    def flush() -> None:
        nonlocal doc_key, doc_rows
        if doc_key is not None and on_doc is not None:
            on_doc(doc_key[0], doc_key[1], doc_rows)
        doc_key, doc_rows = None, []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        layers = iter(ex.map(lambda p: _extract_layer(p.doc, p.page), candidates))
        for pg in pages:
            if progress is not None:
                progress(f"{pg.doc}#{pg.page}")
            if (pg.model, pg.doc) != doc_key:
                flush()
                doc_key = (pg.model, pg.doc)
            if (pg.model, pg.doc) in skip_docs:
                continue
            layer = next(layers)
            if sum(c.isalnum() for c in layer) < _MIN_ALNUM:
                report.pages_blank_layer += 1
                continue
            row = (pg.model, pg.doc, pg.page, bow_f1(pg.text, layer), one_minus_ned(pg.text, layer))
            rows.append(row)
            doc_rows.append(row)
        flush()

    return rows, report
