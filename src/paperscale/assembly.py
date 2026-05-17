from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from paperscale.quality.verifier import assess_markdown_fragment


@dataclass(frozen=True)
class PageMarkdownArtifact:
    document_id: str
    page_number: int
    markdown: str


def assemble_document_markdown(
    artifacts: list[PageMarkdownArtifact],
    *,
    title: str | None = None,
    enforce_quality: bool = False,
) -> str:
    if not artifacts:
        return ""

    document_ids = {artifact.document_id for artifact in artifacts}
    if len(document_ids) != 1:
        raise ValueError("assembly requires a single document")

    seen_pages: set[int] = set()
    ordered: list[PageMarkdownArtifact] = []
    for artifact in sorted(artifacts, key=lambda item: item.page_number):
        if artifact.page_number in seen_pages:
            raise ValueError(f"duplicate page {artifact.page_number}")
        seen_pages.add(artifact.page_number)
        if enforce_quality:
            report = assess_markdown_fragment(artifact.markdown)
            if not report.accepted:
                raise ValueError(f"quality check failed for page {artifact.page_number}")
        ordered.append(artifact)

    parts: list[str] = []
    if title:
        parts.append(f"# {title}\n")

    for index, artifact in enumerate(ordered):
        if index:
            parts.append("\n<!-- page-break -->\n\n")
        parts.append(artifact.markdown.rstrip() + "\n")

    return "".join(parts)


def load_page_markdown_artifacts(path: str | Path) -> list[PageMarkdownArtifact]:
    artifacts: list[PageMarkdownArtifact] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            artifacts.append(
                PageMarkdownArtifact(
                    document_id=record["document_id"],
                    page_number=int(record["page_number"]),
                    markdown=record["markdown"],
                )
            )
    return artifacts


def write_document_markdown(path: str | Path, markdown: str) -> None:
    Path(path).write_text(markdown, encoding="utf-8")
