"""Synthetic Dolma-JSONL fixtures mirroring pipeline.build_dolma_document output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def make_dolma_record(source_file: str, page_texts: list[str], *, fallback_pages: int = 0) -> dict:
    """Build one Dolma document record from per-page texts.

    An empty string for a page produces a zero-length span (blank page), exactly
    as build_dolma_document does when natural_text is None.
    """
    document_text = ""
    spans = []
    for i, content in enumerate(page_texts):
        piece = (content + ("\n" if i < len(page_texts) - 1 else "")) if content else ""
        start = len(document_text)
        document_text += piece
        spans.append([start, len(document_text), i + 1])
    return {
        "id": hashlib.sha1(document_text.encode()).hexdigest(),
        "text": document_text,
        "source": "paperscale",
        "metadata": {
            "Source-File": source_file,
            "pdf-total-pages": len(page_texts),
            "total-fallback-pages": fallback_pages,
        },
        "attributes": {"pdf_page_numbers": spans},
    }


def write_run(tmpdir: Path, records: list[dict], *, as_workspace: bool = True) -> Path:
    """Write records as a run. Returns the path to pass to load_run.

    as_workspace=True writes results/output_0.jsonl (workspace layout); otherwise
    writes a bare run.jsonl file and returns that file.
    """
    if as_workspace:
        results = tmpdir / "results"
        results.mkdir(parents=True, exist_ok=True)
        path = results / "output_0.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return tmpdir
    path = tmpdir / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path
