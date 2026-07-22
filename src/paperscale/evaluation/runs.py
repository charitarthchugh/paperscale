"""Load OCR run outputs (Dolma JSONL) into per-page rows for evaluation.

A run's results are Dolma JSONL documents (one record per PDF) written by
`pipeline.build_dolma_document`. Each record carries:

- ``text``      full document text (all pages concatenated)
- ``metadata``  ``{"Source-File", "pdf-total-pages", "total-fallback-pages", ...}``
- ``attributes.pdf_page_numbers``  ``[[start_char, end_char, page_num], ...]`` spans

The model name is NOT recorded in the JSONL, so the caller supplies an explicit
``label`` (see ``paperscale evaluate --run label=path``). Documents join across
models by ``metadata["Source-File"]`` -- never by the record ``id`` (it is a
sha1 of the text and therefore differs per model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PageText:
    """One page of one model's output. ``doc`` is the source PDF path (join key)."""

    model: str
    doc: str
    page: int
    text: str


@dataclass(frozen=True)
class DocMeta:
    """Per-document metadata for a single model's run of one PDF."""

    model: str
    doc: str
    total_pages: int
    fallback_pages: int
    source_file: str


class DuplicateSourceFileError(RuntimeError):
    """Two records in one run share a Source-File -- the join key is ambiguous."""


def _iter_jsonl_paths(path: Path) -> list[Path]:
    """Resolve an input into concrete .jsonl files.

    Accepts a workspace dir (globs ``results/*.jsonl``), a bare dir of ``*.jsonl``,
    or a single ``.jsonl`` file.
    """
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"run path does not exist: {path}")
    results = sorted((path / "results").glob("*.jsonl"))
    if results:
        return results
    return sorted(path.glob("*.jsonl"))


def _iter_records(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_run(label: str, path: str | Path) -> tuple[list[PageText], list[DocMeta]]:
    """Load one model's run into ``(pages, metas)``.

    Pages with a zero-length span (blank/blank-accepted pages -> empty text) are
    skipped entirely. Raises ``DuplicateSourceFileError`` if a Source-File appears
    in more than one record (ambiguous cross-model join key).
    """
    paths = _iter_jsonl_paths(Path(path))
    pages: list[PageText] = []
    metas: list[DocMeta] = []
    seen: set[str] = set()

    for rec in _iter_records(paths):
        source_file = rec["metadata"]["Source-File"]
        if source_file in seen:
            raise DuplicateSourceFileError(f"{label}: duplicate Source-File {source_file!r} in run")
        seen.add(source_file)

        text = rec["text"]
        for start, end, page_num in rec["attributes"]["pdf_page_numbers"]:
            if end <= start:
                continue  # blank / blank-accepted page -> no row
            pages.append(PageText(model=label, doc=source_file, page=page_num, text=text[start:end]))

        metas.append(
            DocMeta(
                model=label,
                doc=source_file,
                total_pages=rec["metadata"].get("pdf-total-pages", len(rec["attributes"]["pdf_page_numbers"])),
                fallback_pages=rec["metadata"].get("total-fallback-pages", 0),
                source_file=source_file,
            )
        )

    return pages, metas
