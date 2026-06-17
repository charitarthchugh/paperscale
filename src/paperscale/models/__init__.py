"""OCR model registry — the decoupled-model bolt-on.

The pipeline selects an adapter by name (``--ocr-model``); ``markdown`` is the
default and works with any OpenAI-compatible model that emits Markdown.
"""

from __future__ import annotations

from paperscale.models.base import OCRModel
from paperscale.models.lightonocr import LightOnOCRModel
from paperscale.models.markdown import MarkdownModel
from paperscale.models.olmocr import OlmOCRModel

DEFAULT_MODEL = "markdown"

MODEL_REGISTRY: dict[str, type[OCRModel]] = {
    "markdown": MarkdownModel,
    "olmocr": OlmOCRModel,
    "lightonocr2": LightOnOCRModel,
}


def build_ocr_model(name: str) -> OCRModel:
    """Instantiate a registered OCR model adapter by name."""
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown --ocr-model {name!r}. Choose from: {choices}") from None
    return factory()


__all__ = [
    "OCRModel",
    "MarkdownModel",
    "OlmOCRModel",
    "LightOnOCRModel",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "build_ocr_model",
]
