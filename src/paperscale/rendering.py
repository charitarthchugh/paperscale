"""PDF page rendering for document-to-Markdown OCR.

Rasterizes with **poppler** (``pdftoppm``), shelling out one page at a time and
reading the PNG straight off stdout; page count comes from ``pdfinfo``. poppler
renders the full range of real-world PDFs — including scanned pages stored as
ImageMask / CCITT / JBIG2 stencil masks — so a scanned page yields real ink, not
a blank image and therefore empty OCR. Rendering stays lazy: one page is decoded
on demand at a time.

Requires the ``poppler-utils`` binaries (``pdftoppm`` and ``pdfinfo``) on PATH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from PIL import Image

_PDFTOPPM = "pdftoppm"
_PDFINFO = "pdfinfo"
_PAGES_RE = re.compile(r"^Pages:\s+(\d+)", re.MULTILINE)


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
_MIN_DPI = 14.4  # 0.2 px/pt floor, matching the previous PDFium scale clamp
_MAX_LONGEST_SIDE_PX = 4000


class PdfPageRenderer:
    """Render PDF pages as PNG bytes via poppler while exposing a 1-based public API."""

    def __init__(self, path: Path | str, *, render_options: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.render_options = dict(render_options or {})
        image_format = str(self.render_options.get("image_format", "png"))
        if image_format != "png":
            raise ValueError("v1 OCR rendering only supports PNG output")
        if shutil.which(_PDFTOPPM) is None or shutil.which(_PDFINFO) is None:
            raise RuntimeError(
                "poppler-utils (pdftoppm + pdfinfo) must be installed and on PATH for PDF rendering"
            )
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._page_count: int | None = None

    @property
    def page_count(self) -> int:
        if self._page_count is None:
            self._page_count = self._read_page_count()
        return self._page_count

    def _read_page_count(self) -> int:
        result = subprocess.run(
            [_PDFINFO, str(self.path)],
            capture_output=True,
            text=True,
            check=True,
        )
        match = _PAGES_RE.search(result.stdout)
        if match is None:
            raise RuntimeError(f"pdfinfo reported no page count for {self.path}")
        return int(match.group(1))

    def render_page(self, page_number: int) -> RenderedPage:
        if page_number < 1 or page_number > self.page_count:
            raise ValueError(f"page_number must be between 1 and {self.page_count}, got {page_number}")
        command = [
            _PDFTOPPM,
            "-png",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            *self._sizing_args(),
            str(self.path),
        ]
        result = subprocess.run(command, capture_output=True, check=True)
        image_bytes = result.stdout
        if not image_bytes:
            stderr = result.stderr.decode("utf-8", "replace").strip()
            raise RuntimeError(f"pdftoppm produced no output for page {page_number}: {stderr}")
        return RenderedPage(
            page_number=page_number,
            image_bytes=image_bytes,
            image_hash=hashlib.sha256(image_bytes).hexdigest(),
            media_type="image/png",
            render_options=dict(self.render_options),
        )

    def _sizing_args(self) -> list[str]:
        """poppler resolution flags.

        Priority: explicit ``dpi`` (set by render remediation) > ``target_longest_side``
        (``-scale-to`` fits the larger page dimension to N px) > a sensible default DPI.
        """
        dpi = self.render_options.get("dpi")
        if dpi:
            return ["-r", _format_number(max(_MIN_DPI, float(dpi)))]
        target = self.render_options.get("target_longest_side")
        if target:
            target_px = min(int(float(target)), _MAX_LONGEST_SIDE_PX)
            return ["-scale-to", str(max(1, target_px))]
        return ["-r", _format_number(_DEFAULT_DPI)]


def _format_number(value: float) -> str:
    """Render a DPI without a trailing ``.0`` so pdftoppm gets a clean integer when possible."""
    return str(int(value)) if value == int(value) else str(value)
