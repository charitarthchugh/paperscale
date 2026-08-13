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
    pages: list[PageText], metas: list[DocMeta], progress=None, jobs: int | None = None
) -> tuple[list[tuple[str, str, int, float, float]], SkipReport]:
    """``progress``: optional callable(note) invoked once per page (instrumentation only).

    ``jobs``: thread-pool width for the pdftotext subprocess calls (I/O-bound,
    GIL released while blocked on the child process).
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
    with ThreadPoolExecutor(max_workers=workers) as ex:
        layers = iter(ex.map(lambda p: _extract_layer(p.doc, p.page), candidates))
        for pg in pages:
            if progress is not None:
                progress(f"{pg.doc}#{pg.page}")
            if (pg.model, pg.doc) in skip_docs:
                continue
            layer = next(layers)
            if sum(c.isalnum() for c in layer) < _MIN_ALNUM:
                report.pages_blank_layer += 1
                continue
            rows.append((pg.model, pg.doc, pg.page, bow_f1(pg.text, layer), one_minus_ned(pg.text, layer)))

    return rows, report
