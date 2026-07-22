"""Unlimited-OCR adapter: Baidu's DeepSeek-OCR successor, full-page Markdown.

Unlimited-OCR (https://huggingface.co/baidu/Unlimited-OCR) is a ~3.3B
vision-language model from Baidu that pushes DeepSeek-OCR one step further
("one-shot long-horizon parsing"). One page image plus the vendor prompt
``<image>document parsing.`` yields the page as Markdown with embedded
grounding tokens; :meth:`parse` unwraps ``<|ref|>text<|/ref|>`` and drops
``<|det|>[[boxes]]<|/det|>`` to leave clean Markdown. Selected with
``--ocr-model unlimited-ocr``.

vLLM serving (recipe: https://recipes.vllm.ai/baidu/Unlimited-OCR, needs
vLLM >= 0.25.0 / the ``vllm/vllm-openai:unlimited-ocr`` image)::

    vllm serve baidu/Unlimited-OCR --trust-remote-code \
        --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
        --no-enable-prefix-caching --mm-processor-cache-gb 0

The custom logits processor is mandatory (long documents loop on coordinate
tokens without it); this adapter passes its per-request args (``ngram_size=35``,
``window_size=128``) via ``vllm_xargs``. The prompt text must begin with a
literal ``<image>`` tag and requests must set ``skip_special_tokens=false`` —
missing either produces empty output. Single-image requests use the model's
crop ("gundam") mode with base_size=1024, so pages render at 1024 px.
"""

from __future__ import annotations

import re

from paperscale.models.markdown import MarkdownModel
from paperscale.prompts import PageResponse

# Vendor prompt; the literal <image> prefix is required by the vLLM chat path.
UNLIMITED_OCR_PROMPT = "<image>document parsing."

# Grounding output: <|ref|>text<|/ref|> optionally followed by a
# <|det|>[[x1,y1,x2,y2],...]<|/det|> coordinate block. Keep the text, drop the boxes.
_REF = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_DET = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
# skip_special_tokens=false leaves structural specials (e.g. end-of-sentence) in the text.
_SPECIAL = re.compile(r"<[|｜][^<>]*?[|｜]>")


class UnlimitedOCRModel(MarkdownModel):
    """Drives Unlimited-OCR, which emits page Markdown with grounding tokens."""

    default_model_name = "baidu/Unlimited-OCR"
    preferred_longest_image_dim = 1024

    def __init__(self) -> None:
        super().__init__(prompt=UNLIMITED_OCR_PROMPT)

    def sampling_params(self) -> dict:
        # skip_special_tokens=false is required for non-empty output; the ngram
        # anti-loop processor args ride along per request via vllm_xargs.
        # Temperature stays pipeline-owned.
        return {
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        }

    def parse(self, content: str) -> PageResponse:
        content = _REF.sub(r"\1", content)
        content = _DET.sub("", content)
        content = _SPECIAL.sub("", content)
        return super().parse(content)
