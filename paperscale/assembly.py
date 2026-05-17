"""Document-level Markdown assembly from completed page OCR artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from paperscale.quality.verifier import assess_markdown_fragment

PAGE_BREAK = "<!-- page-break -->"


@dataclass(frozen=True, slots=True)
class PageMarkdownArtifact:
    """A durable page OCR artifact consumed by document assembly."""

    document_id: str
    page_number: int
    markdown: str


def assemble_document_markdown(
    pages: list[PageMarkdownArtifact],
    *,
    title: str | None = None,
    enforce_quality: bool = False,
    partial: bool = False,
) -> str:
    """Assemble completed page Markdown artifacts into one document.

    Assembly is intentionally separate from page OCR. It validates that all
    inputs belong to a single document, sorts by page number, and can apply the
    deterministic quality gate before emitting the final Markdown string.
    """

    if not pages:
        raise ValueError("cannot assemble a document with no pages")

    document_ids = {page.document_id for page in pages}
    if len(document_ids) != 1:
        raise ValueError("assembly requires pages from a single document")

    seen_pages: set[int] = set()
    for page in pages:
        if page.page_number < 1:
            raise ValueError("page numbers must be one-based positive integers")
        if page.page_number in seen_pages:
            raise ValueError(f"duplicate page number {page.page_number}")
        seen_pages.add(page.page_number)

        if enforce_quality:
            report = assess_markdown_fragment(page.markdown)
            if not report.accepted:
                codes = ", ".join(issue.code for issue in report.issues)
                raise ValueError(f"quality check failed for page {page.page_number}: {codes}")

    ordered_pages = sorted(pages, key=lambda page: page.page_number)
    fragments = [_normalize_fragment(page.markdown) for page in ordered_pages]
    body = f"\n\n{PAGE_BREAK}\n\n".join(fragments)
    if partial:
        body = "<!-- partial -->\n\n" + body
    if title:
        body = f"# {title.strip()}\n\n{body}"
    return f"{body.rstrip()}\n"


def _normalize_fragment(markdown: str) -> str:
    return markdown.strip()
