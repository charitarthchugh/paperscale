"""Qianfan-OCR adapter: single image+text call, Markdown output.

Qianfan-OCR (https://huggingface.co/baidu/Qianfan-OCR) is a ~4B document model
pairing a Qianfan-ViT encoder with a Qwen3-4B decoder. It is served on vLLM as an
``InternVLChatModel`` via ``--hf-overrides`` (see the serve command in the README);
one image + text prompt yields clean reading-ordered Markdown with HTML tables and
``$$ .. $$`` LaTeX. Selected with ``--ocr-model qianfan-ocr``.

Parsing reuses the generic Markdown path, with one defensive twist: when
``enable_thinking=True`` Qianfan first emits a ``<think>...</think>`` block carrying
``<COORD_*>`` layout tokens before the answer. paperscale never enables thinking, so
no block should appear, but :meth:`parse` strips one anyway and keeps the text after
the final ``</think>``.
"""

from __future__ import annotations

import re
from dataclasses import replace

from paperscale.models.markdown import MarkdownModel
from paperscale.prompts import PageResponse

# A complete <think>...</think> reasoning block (only emitted when
# enable_thinking=True, which paperscale never sets). Matched-pair only, so a
# literal "</think>" transcribed from the page can't truncate real content.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class QianfanOCRModel(MarkdownModel):
    """Drives Qianfan-OCR, which transcribes a page image to reading-ordered Markdown."""

    default_model_name = "baidu/Qianfan-OCR"
    preferred_longest_image_dim = 1540

    def __init__(self) -> None:
        super().__init__(prompt="Parse this document to Markdown.")

    def sampling_params(self) -> dict:
        # Raise max_tokens above the pipeline's 8000 default: Qianfan pages run long.
        # Temperature stays pipeline-owned.
        return {"top_p": 1.0, "max_tokens": 12000}

    def parse(self, content: str) -> PageResponse:
        # Defensive: drop any <think>...</think> reasoning block. paperscale never
        # sets enable_thinking, so this is normally a no-op; matched-pair only, so a
        # literal "</think>" in the page can't truncate the transcription.
        content = _THINK_BLOCK.sub("", content)
        result = super().parse(content)
        # Qianfan returns tables as HTML, which paperscale accepts as-is. Recompute
        # is_table from the emitted Markdown rather than convert it.
        is_table = "<table" in (result.natural_text or "").lower()
        return replace(result, is_table=is_table)
