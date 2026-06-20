"""Surya OCR 2 adapter: reading-ordered *layout-HTML* output, Markdown parse.

Surya OCR 2 (https://huggingface.co/datalab-to/surya-ocr-2) is a ~0.65B
Qwen3.5-style vision-language model (``Qwen3_5ForConditionalGeneration``). Unlike
the Markdown-emitting adapters, its full-page-OCR mode returns one **layout-HTML**
document per page: a flat sequence of ``<div>`` blocks *in reading order*, each
carrying ``data-bbox`` (normalized 0-1000 coordinates) and a ``data-label``
(``Text``, ``Section-Header``, ``List-Group``, ``Table``, ``Figure``,
``Image`` …). Equations are embedded as ``<math>`` (KaTeX LaTeX) and tables as
real HTML ``<table>`` markup (with ``rowspan``/``colspan``). Selected with
``--ocr-model surya2``.

One VLM call produces the whole page — the layout pre-stage that ``surya`` runs
locally is optional and is skipped here, since the high-accuracy-bbox prompt makes
the model emit labelled blocks directly. Serve it on a recent vLLM (v0.20.0+) with
native ``qwen3_5`` support; avoid v0.18.0, which predates the architecture.

``parse`` walks the layout-HTML once with a tiny :class:`markdownify`
``MarkdownConverter`` subclass that maps each labelled ``<div>`` to its Markdown
equivalent (headings, list items, paragraphs), rewrites ``<math>`` into
paperscale's backslash-delimited LaTeX, and lets markdownify's built-in handlers
turn ``<table>`` and inline emphasis into Markdown.
"""

from __future__ import annotations

import html
import re

from markdownify import MarkdownConverter

from paperscale.models.base import OCRModel
from paperscale.prompts import PageResponse

_TAG = re.compile(r"<[^>]+>")


def _strip_tags(content: str) -> str:
    """Last-resort plain-text fallback when HTML conversion fails.

    A degenerate response (e.g. a repetition loop nesting ``<div>`` thousands
    deep) can overflow markdownify's recursive walk; a regex strip never recurses.
    """
    return html.unescape(_TAG.sub(" ", content))

# Surya's full-page-OCR instruction. This is the high-accuracy-bbox prompt from
# the surya-ocr package (surya/inference/prompts.py, HIGH_ACCURACY_BBOX_PROMPT);
# it is what drives the labelled ``<div data-label=… data-bbox=…>`` layout-HTML
# output this adapter parses. The sibling block/layout prompts flip the model into
# block-OCR or layout-JSON mode, so this exact string matters.
SURYA_OCR_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)

# Surya's block ``data-label`` values (its LAYOUT_LABEL_SET; note the hyphens).
# Section headers/titles become Markdown headings; the rest of a block's body is
# already semantic HTML (<p>/<b>/<ol>/<table>/<math> …) that markdownify converts.
_HEADING_LABELS = {"Section-Header", "Title"}
_DIAGRAM_LABELS = ("Figure", "Image", "Diagram")

# Table/diagram page flags, read from the block labels. Tolerant of the model's
# attribute quoting (single or double quotes, optional surrounding whitespace) so a
# correctly-transcribed page isn't mis-flagged on a quoting change.
_TABLE_LABEL_RE = re.compile(r"""data-label\s*=\s*["']Table["']""")
_DIAGRAM_LABEL_RE = re.compile(r"""data-label\s*=\s*["'](?:%s)["']""" % "|".join(_DIAGRAM_LABELS))

# Collapse runs of 3+ newlines (markdownify's block spacing) down to a blank line.
_EXCESS_BLANKS = re.compile(r"\n{3,}")

# A list item whose text already opens with its own enumerator ("1.", "(a)", "iv.").
# Surya numbers items inline, so we keep that marker instead of adding another.
_INLINE_LIST_MARKER = re.compile(r"^\(?(?:\d{1,3}|[a-zA-Z]|[ivxlcdm]{1,4})[.)]\s")


class _SuryaHTMLConverter(MarkdownConverter):
    """markdownify converter for Surya's labelled layout-HTML.

    Only the model-specific tags need custom handling: ``<div>`` (read its
    ``data-label`` to pick a Markdown shape) and ``<math>`` (paperscale LaTeX
    delimiters). Tables and inline emphasis fall through to markdownify's
    built-ins.
    """

    def convert_div(self, el, text, parent_tags):
        # data-bbox/data-label are layout metadata, not content. Surya already
        # emits semantic HTML inside each block, so markdownify's built-ins handle
        # the body; we only special-case section headers, which the model wraps in
        # <b><u> rather than <hN> — emit them as a Markdown heading from plain text.
        label = (el.get("data-label") or "").strip()
        if label in _HEADING_LABELS:
            heading = el.get_text(" ", strip=True)
            return f"\n# {heading}\n\n" if heading else ""
        body = text.strip()
        return f"{body}\n\n" if body else ""

    def convert_li(self, el, text, parent_tags):
        # Surya frequently numbers list items *inline* (e.g. "1. The Plaintiff…").
        # Letting markdownify prepend its own marker then yields "1. 1. …", so when
        # the item text already opens with an enumerator, emit it as a plain line;
        # otherwise defer to markdownify's normal bullet/number handling.
        line = text.strip()
        if _INLINE_LIST_MARKER.match(line):
            return f"{line}\n"
        return super().convert_li(el, text, parent_tags)  # type: ignore  (markdownify resolves convert_* dynamically)

    def convert_math(self, el, text, parent_tags):
        # KaTeX LaTeX → paperscale's backslash-delimited convention. ``display``
        # math becomes a block ``\[ … \]``; everything else is inline ``\( … \)``.
        latex = text.strip()
        if not latex:
            return ""
        if (el.get("display") or "").lower() == "block":
            return f"\\[ {latex} \\]"
        return f"\\( {latex} \\)"


class Surya2Model(OCRModel):
    """Drives Surya OCR 2, which returns reading-ordered layout-HTML per page."""

    default_model_name = "datalab-to/surya-ocr-2"
    preferred_longest_image_dim = 1540

    def build_messages(self, image_base64: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SURYA_OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

    def parse(self, content: str) -> PageResponse:
        # One pass over the whole layout-HTML; then squeeze excess blank lines.
        # A degenerate model response (a repetition loop nesting <div>s thousands
        # deep) can overflow markdownify's recursive walk — fall back to a
        # tag-stripped best-effort so the page is retried/verified, never crashed.
        try:
            markdown = _SuryaHTMLConverter().convert(content)
        except Exception:
            markdown = _strip_tags(content)
        markdown = _EXCESS_BLANKS.sub("\n\n", markdown).strip()

        # Label markers are the cheapest reliable signal for table/diagram pages.
        is_table = bool(_TABLE_LABEL_RE.search(content))
        is_diagram = bool(_DIAGRAM_LABEL_RE.search(content))

        return PageResponse(
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=is_table,
            is_diagram=is_diagram,
            natural_text=markdown or None,
        )
