"""Infinity-Parser2 Flash adapter: Qwen3.5-VL document model in doc2md mode.

Infinity-Parser2-Flash (https://huggingface.co/infly/Infinity-Parser2-Flash) is a
~2B Qwen3.5-VL-family document parser. Its native mode is doc2json (bounding-box
JSON); paperscale instead drives it in *doc2md* mode by prompting it to emit
Markdown, so parsing reuses the generic Markdown path. Selected with
``--ocr-model infinity-parser2-flash``.

The model is a Qwen3 reasoning model, so the adapter disables thinking at the
vLLM server via the top-level ``chat_template_kwargs`` request key and also
strips any stray ``<think>...</think>`` block defensively in :meth:`parse`.

Serve it with vLLM >= ~0.20 on Python 3.13::

    vllm serve infly/Infinity-Parser2-Flash --trust-remote-code \\
        --reasoning-parser qwen3 --mm-encoder-tp-mode data \\
        --mm-processor-cache-type shm --enable-prefix-caching

In doc2md mode the response is Markdown with HTML tables and LaTeX equations.
"""

from __future__ import annotations

import re
from dataclasses import replace

from paperscale.models.markdown import MarkdownModel
from paperscale.prompts import PageResponse

# Removes a stray reasoning block if the server's reasoning parser leaks one
# through despite enable_thinking=False.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class InfinityParser2FlashModel(MarkdownModel):
    """Drives Infinity-Parser2-Flash in doc2md mode (Markdown + HTML tables)."""

    default_model_name = "infly/Infinity-Parser2-Flash"
    preferred_longest_image_dim = 1540

    def __init__(self) -> None:
        # doc2md mode: ask for Markdown instead of the model's default
        # doc2json/bbox-JSON output.
        super().__init__(prompt="Please transform the document's contents into Markdown format.")

    def sampling_params(self) -> dict:
        # Recommended decoding plus the Qwen3 thinking switch. The
        # chat_template_kwargs key merges at the top level of the request (NOT
        # under extra_body) and disables reasoning at the vLLM server.
        return {"top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}}

    def parse(self, content: str) -> PageResponse:
        # Defensively drop any leaked reasoning block, then reuse the generic
        # Markdown parse (code-fence strip + normalization).
        content = _THINK_BLOCK.sub("", content)
        result = super().parse(content)
        # doc2md emits tables as HTML, so flag the page when one is present.
        is_table = "<table" in (result.natural_text or "").lower()
        return replace(result, is_table=is_table)
