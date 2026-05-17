"""Production-facing deterministic test helpers for fake end-to-end jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FakePageRequest:
    page_id: str
    image_bytes: bytes
    debug_event_index: int


@dataclass(slots=True)
class FakeDocumentResult:
    markdown: str
    debug_side_effects: list[str] = field(default_factory=list)


class FakeDocumentRunner:
    """Run a no-network document-to-Markdown flow while exposing invariant ordering."""

    def __init__(self, *, store: Any, resources: Any, provider: Any) -> None:
        self.store = store
        self.resources = resources
        self.provider = provider

    def run_document_to_markdown(self, *, document_id: str, pages: list[bytes]) -> FakeDocumentResult:
        fragments: list[str] = []
        for index, image_bytes in enumerate(pages, start=1):
            page_id = f"{document_id}:{index}"
            attempt_id = f"{page_id}:attempt-1"
            with self.resources.acquire("provider_concurrency"):
                self.store.mutate("attempt_reserved", attempt_id)
                self.store.read_index(f"ledger-reservation-{attempt_id}")
                request = FakePageRequest(
                    page_id=page_id,
                    image_bytes=image_bytes,
                    debug_event_index=len(self.store.events),
                )
                response = self.provider.send(request)
                fragments.append(response.markdown)
        return FakeDocumentResult(markdown="\n\n".join(fragments), debug_side_effects=[])
