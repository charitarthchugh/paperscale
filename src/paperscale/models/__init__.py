"""OCR model registry — the decoupled-model bolt-on.

The pipeline selects an adapter by name (``--ocr-model``); ``markdown`` is the
default and works with any OpenAI-compatible model that emits Markdown.
"""

from __future__ import annotations

from paperscale.models.base import OCRModel
from paperscale.models.glmocr import GLMOCRModel
from paperscale.models.infinity_parser import InfinityParser2FlashModel
from paperscale.models.lightonocr import LightOnOCRModel, LightOnOCRSoupModel
from paperscale.models.markdown import MarkdownModel
from paperscale.models.olmocr import OlmOCRModel
from paperscale.models.qianfan import QianfanOCRModel
from paperscale.models.surya import Surya2Model

DEFAULT_MODEL = "markdown"

MODEL_REGISTRY: dict[str, type[OCRModel]] = {
    "markdown": MarkdownModel,
    "olmocr": OlmOCRModel,
    "lightonocr2": LightOnOCRModel,
    "lightonocr2-soup": LightOnOCRSoupModel,
    "glm-ocr": GLMOCRModel,
    "qianfan-ocr": QianfanOCRModel,
    "infinity-parser2-flash": InfinityParser2FlashModel,
    "surya2": Surya2Model,
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
    "LightOnOCRSoupModel",
    "GLMOCRModel",
    "QianfanOCRModel",
    "InfinityParser2FlashModel",
    "Surya2Model",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "build_ocr_model",
]
