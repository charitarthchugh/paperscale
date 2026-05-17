"""Document-level Markdown assembly from completed page OCR artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from paperscale.contracts import PageArtifact
from paperscale.quality.verifier import assess_markdown_fragment

PAGE_BREAK = "<!-- page-break -->"


class AssemblyError(RuntimeError):
    """Raised when document assembly cannot produce a complete artifact."""


@dataclass(frozen=True, slots=True)
class PageMarkdownArtifact:
    """A durable page OCR artifact consumed by document assembly."""

    document_id: str
    page_number: int
    markdown: str


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    markdown: str
    partial: bool
    missing_pages: list[int]


class MarkdownAssembler:
    """Assemble durable page OCR artifacts without mutating page results."""

    def __init__(self, *, required_pages: list[int]) -> None:
        self.required_pages = list(required_pages)

    def assemble(self, artifacts: list[PageArtifact], allow_partial: bool = False) -> AssemblyResult:
        by_page = {artifact.page_number: artifact for artifact in artifacts}
        missing = [page for page in self.required_pages if page not in by_page]
        if missing and not allow_partial:
            raise AssemblyError(f"missing required pages: {missing}")
        ordered = [by_page[page] for page in self.required_pages if page in by_page]
        fragments = [artifact.markdown.strip() for artifact in ordered]
        markdown = f"\n\n{PAGE_BREAK}\n\n".join(fragments).rstrip()
        partial = bool(missing)
        if partial:
            markdown = f"<!-- PARTIAL: missing pages {missing} -->\n\n{markdown}".rstrip()
        return AssemblyResult(markdown=f"{markdown}\n", partial=partial, missing_pages=missing)


def assemble_document_markdown(
    pages: list[PageMarkdownArtifact],
    *,
    title: str | None = None,
    enforce_quality: bool = False,
    partial: bool = False,
) -> str:
    """Assemble completed page Markdown artifacts into one document."""

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
