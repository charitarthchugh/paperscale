"""PDF page rendering adapters for document-to-Markdown OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from pdf_oxide import PdfDocument


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Rendered public 1-based PDF page image."""

    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"
    render_options: dict[str, Any] = field(default_factory=dict)


class PdfPageRenderer:
    """Render PDF pages as PNG bytes while exposing a 1-based public API."""

    def __init__(self, path: Path | str, *, render_options: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.render_options = dict(render_options or {})
        self._document = PdfDocument(str(self.path))

    @property
    def page_count(self) -> int:
        return int(self._document.page_count())

    def render_page(self, page_number: int) -> RenderedPage:
        if page_number < 1 or page_number > self.page_count:
            raise ValueError(f"page_number must be between 1 and {self.page_count}, got {page_number}")
        page_index = page_number - 1
        image_format = str(self.render_options.get("image_format", "png"))
        if image_format != "png":
            raise ValueError("v1 OCR rendering only supports PNG output")
        dpi = self.render_options.get("dpi")
        image_bytes = self._document.render_page(page_index, dpi=dpi, format="png")
        return RenderedPage(
            page_number=page_number,
            image_bytes=image_bytes,
            image_hash=hashlib.sha256(image_bytes).hexdigest(),
            media_type="image/png",
            render_options=dict(self.render_options),
        )
