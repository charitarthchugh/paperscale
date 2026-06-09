"""PDF page rendering for document-to-Markdown OCR.

Rasterizes with **PDFium** (``pypdfium2``), which renders the full range of
real-world PDFs — including scanned pages stored as ImageMask / CCITT / JBIG2
stencil masks. (The earlier ``pdf_oxide`` backend silently skipped those XObjects,
yielding a blank image and therefore empty OCR.) Rendering stays lazy: one page is
decoded on demand at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image


def png_dark_fraction(image_bytes: bytes) -> float:
    """Fraction of non-near-white pixels in a PNG — a cheap "how much ink" measure.

    Used to distinguish a genuinely blank page (which legitimately OCRs to empty)
    from a content page the model failed to read. Near-blank scans sit around
    0.002–0.005; content pages are an order of magnitude higher.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    histogram = image.histogram()
    total = sum(histogram) or 1
    dark = sum(histogram[:230])
    return dark / total


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Rendered public 1-based PDF page image."""

    page_number: int
    image_bytes: bytes
    image_hash: str
    media_type: str = "image/png"
    render_options: dict[str, Any] = field(default_factory=dict)


# Resolution bounds. ``target_longest_side`` is honored when no explicit ``dpi`` is
# given; both are clamped so a remediation DPI bump cannot produce a runaway bitmap.
_DEFAULT_DPI = 144.0
_MIN_SCALE = 0.2
_MAX_LONGEST_SIDE_PX = 4000


class PdfPageRenderer:
    """Render PDF pages as PNG bytes while exposing a 1-based public API."""

    def __init__(self, path: Path | str, *, render_options: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.render_options = dict(render_options or {})
        image_format = str(self.render_options.get("image_format", "png"))
        if image_format != "png":
            raise ValueError("v1 OCR rendering only supports PNG output")
        self._document = pdfium.PdfDocument(str(self.path))

    @property
    def page_count(self) -> int:
        return int(len(self._document))

    def render_page(self, page_number: int) -> RenderedPage:
        if page_number < 1 or page_number > self.page_count:
            raise ValueError(f"page_number must be between 1 and {self.page_count}, got {page_number}")
        page = self._document[page_number - 1]
        width_pt, height_pt = page.get_size()
        scale = self._scale_for(float(width_pt), float(height_pt))
        pil_image = page.render(scale=scale).to_pil()
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return RenderedPage(
            page_number=page_number,
            image_bytes=image_bytes,
            image_hash=hashlib.sha256(image_bytes).hexdigest(),
            media_type="image/png",
            render_options=dict(self.render_options),
        )

    def _scale_for(self, width_pt: float, height_pt: float) -> float:
        """PDFium render scale (pixels per point; 1.0 == 72 dpi).

        Priority: explicit ``dpi`` (set by render remediation) > ``target_longest_side``
        (honored exactly) > a sensible default.
        """
        dpi = self.render_options.get("dpi")
        if dpi:
            return max(_MIN_SCALE, float(dpi) / 72.0)
        target = self.render_options.get("target_longest_side")
        longest_pt = max(width_pt, height_pt, 1.0)
        if target:
            target_px = min(float(target), float(_MAX_LONGEST_SIDE_PX))
            return max(_MIN_SCALE, target_px / longest_pt)
        return _DEFAULT_DPI / 72.0
