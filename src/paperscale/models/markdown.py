"""Default adapter: any OCR model that emits Markdown directly."""

from __future__ import annotations

import re

from paperscale.models.base import OCRModel
from paperscale.prompts import PageResponse

DEFAULT_MARKDOWN_PROMPT = (
    "Transcribe this document page into clean GitHub-flavored Markdown.\n"
    "Render equations as LaTeX (use \\( \\) for inline math and \\[ \\] for block math) "
    "and render tables as Markdown tables. Preserve the reading order, headings, and lists.\n"
    "Output only the Markdown content of the page: no commentary, no explanations, and no "
    "surrounding code fences. If the page contains no readable text, output nothing."
)

_CODE_FENCE = re.compile(r"^\s*```(?:markdown|md)?\s*\n(?P<body>.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)

# A complete <think>...</think> reasoning block. Several OCR VLMs emit one when
# thinking is enabled; paperscale never enables it, so stripping is defensive.
# Matched-pair only, so a literal "</think>" transcribed from the page cannot
# truncate real content.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_code_fence(content: str) -> str:
    """Remove a single wrapping ```markdown ... ``` fence some models add."""
    match = _CODE_FENCE.match(content)
    return match.group("body") if match else content


def strip_think_blocks(content: str) -> str:
    """Drop any complete ``<think>...</think>`` reasoning block from a response."""
    return _THINK_BLOCK.sub("", content)


def has_html_table(markdown: str | None) -> bool:
    """Whether Markdown carries an HTML ``<table>``, which paperscale keeps as-is."""
    return "<table" in (markdown or "").lower()


class MarkdownModel(OCRModel):
    """Generic adapter for OCR models whose response *is* the page Markdown.

    No front matter and no rotation protocol: ``parse`` returns the text as-is
    with rotation reported valid, so the pipeline's model-agnostic retry path
    treats a well-formed response as an immediate success.
    """

    default_model_name = "paperscale-markdown"

    def __init__(self, prompt: str = DEFAULT_MARKDOWN_PROMPT) -> None:
        self._prompt = prompt

    def build_messages(self, image_base64: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

    def parse(self, content: str) -> PageResponse:
        text = _strip_code_fence(content).strip()
        return PageResponse(
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
            natural_text=text or None,
        )
