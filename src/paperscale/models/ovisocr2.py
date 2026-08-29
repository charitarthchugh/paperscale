"""OvisOCR2 adapter: 0.8B end-to-end page parser, full-page Markdown.

OvisOCR2 (https://huggingface.co/ATH-MaaS/OvisOCR2) is a 0.8B document-parsing
VLM post-trained from Qwen3.5-0.8B (``Qwen3_5ForConditionalGeneration``,
``model_type: qwen3_5`` — the same family as ``surya2``, but with hybrid
Gated-DeltaNet ``linear_attention`` layers). One page image plus the vendor
instruction yields the whole page as Markdown in reading order. It tops
OmniDocBench v1.6 (96.58) and PureDocBench (Avg3 75.06). Selected with
``--ocr-model ovisocr2``.

Output shape: standard Markdown for prose, **LaTeX** for formulas, **HTML**
``<table>`` for tables (paperscale accepts these as-is, like ``qianfan-ocr``),
and visual regions as placeholder ``<img src="images/bbox_{l}_{t}_{r}_{b}.jpg" />``
tags with coordinates scaled to [0, 1000). paperscale never writes the crop
files those tags reference, so :meth:`parse` strips them and reports the page as
a diagram instead.

vLLM serving (vendor pins ``vllm==0.22.1``; the checkpoint declares
``transformers_version 4.57.0.dev0``, so pair it with transformers >= 4.57.
Verified end-to-end on vLLM 0.27.1, which registers the arch natively)::

    vllm serve ATH-MaaS/OvisOCR2 --port 8000 \
        --gdn-prefill-backend triton \
        --limit-mm-per-prompt '{"image": 1}' \
        --mm-processor-kwargs '{"images_kwargs": {"min_pixels": 200704, "max_pixels": 8294400}}'

``--gdn-prefill-backend triton`` is the vendor's ``gdn_prefill_backend="triton"``
LLM kwarg; the mm-processor bounds are their ``min_pixels=448*448`` /
``max_pixels=2880*2880``. The default 1540px render sits comfortably inside that
band (~1.8 MP for a letter page), and the band tolerates up to ~3270px on the
long edge via ``--target_longest_image_dim`` if fine print needs it.

Degenerate output: the vendor's own parser trims a repeating tail out of long
responses before returning them. This adapter deliberately does **not** — a
looping page is an *incomplete* page (the model spent its budget repeating
instead of transcribing the rest), and rewriting it here would hand the pipeline
something that reads as clean and complete. The same detection lives in the
quality gate instead, as
:func:`paperscale.quality.verifier._has_repeated_tail_loop`, so the page is
rejected and retried at a higher temperature. Disable it with
``--disable-quality-check repeated_tail`` if a corpus trips it systematically.
"""

from __future__ import annotations

import re
from dataclasses import replace

from paperscale.models.markdown import MarkdownModel, has_html_table, strip_think_blocks
from paperscale.prompts import PageResponse

# Vendor inference prompt, verbatim from the model card's OvisOCR2Parser. The
# leading newline and the literal ``{left}``/``{top}``/… braces are part of the
# string (the vendor builds it as a plain, non-f string). Keep both: the chat
# template trims only the *ends* of the rendered user turn, so with the image
# part sent first this newline survives as the image/instruction separator,
# reproducing the vendor's exact token sequence.
OVIS_OCR2_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe "
    "all other text as standard Markdown. Preserve the original text without "
    "translation or paraphrasing."
)

# Placeholder tags for visual regions: <img src="images/bbox_12_34_56_78.jpg" />.
# The vendor's own parser drops these by default (filter_imgtags=True) because the
# referenced crops only exist if you save them alongside the Markdown; paperscale
# never does, so every occurrence — block-level or inline — is a dead reference.
_BBOX_IMG = re.compile(r'<img\s+src="images/bbox_\d+_\d+_\d+_\d+\.jpg"\s*/?>')

# Blank lines left behind where a stripped <img> tag was the whole block.
_EXCESS_BLANKS = re.compile(r"\n{3,}")


class OvisOCR2Model(MarkdownModel):
    """Drives OvisOCR2, which transcribes a page image to full-page Markdown."""

    default_model_name = "ATH-MaaS/OvisOCR2"
    preferred_longest_image_dim = 1540

    def __init__(self) -> None:
        super().__init__(prompt=OVIS_OCR2_PROMPT)

    def build_messages(self, image_base64: str) -> list[dict]:
        # Image part first, then the instruction: this is the vendor's content
        # order, and it is what keeps OVIS_OCR2_PROMPT's leading newline from
        # being trimmed off the front of the rendered user turn.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

    def sampling_params(self) -> dict:
        # Vendor decodes with max_tokens=16384; full pages of dense HTML tables
        # overrun the pipeline's 8000 default. Temperature stays pipeline-owned.
        return {"max_tokens": 16384}

    def parse(self, content: str) -> PageResponse:
        # Defensive: the chat template pre-closes thinking unless enable_thinking is
        # explicitly true, which paperscale never sets, so no block should appear.
        content = strip_think_blocks(content)
        # A bbox img tag is the model's marker for a chart/figure region, so read
        # the diagram flag off the raw text before the dead references go away.
        is_diagram = bool(_BBOX_IMG.search(content))
        content = _EXCESS_BLANKS.sub("\n\n", _BBOX_IMG.sub("", content))
        result = super().parse(content)
        # Tables come back as HTML, which paperscale keeps as-is; recompute the
        # flag from the emitted Markdown rather than converting it.
        return replace(result, is_table=has_html_table(result.natural_text), is_diagram=is_diagram)
