"""GLM-OCR adapter: GLM-4.1V-derived VLM, single-call full-page Markdown.

GLM-OCR (https://huggingface.co/zai-org/GLM-OCR) is a ~0.9B vision-language
model derived from GLM-4.1V and fine-tuned for document transcription. It takes
a page image plus a full-page instruction and emits the whole page as Markdown
in one call, so parsing reuses the generic Markdown path. Selected with
``--ocr-model glm-ocr``.

vLLM serves it natively via the ``glm_ocr`` architecture (added in vLLM
PR #33005); run a recent build carrying that arch and upgrade ``transformers``
alongside. The vendor's best-quality path is a two-stage layout pipeline
(detect regions, then transcribe each); this adapter uses the simpler
single-call full-page path, which keeps it one image -> one text -> one
``PageResponse`` like the other Markdown adapters.
"""

from __future__ import annotations

from paperscale.models.markdown import MarkdownModel

# Full-page transcription instruction sent with every page image.
GLM_OCR_PROMPT = (
    "Recognize the text in the image and output it in Markdown format. Preserve the "
    "original layout (headings, paragraphs, lists, tables, formulas). Render tables as "
    "Markdown tables and formulas as LaTeX. Do not fabricate content that does not exist "
    "in the image."
)


class GLMOCRModel(MarkdownModel):
    """Drives GLM-OCR, which transcribes a page image to full-page Markdown."""

    default_model_name = "zai-org/GLM-OCR"
    preferred_longest_image_dim = 1540

    def __init__(self) -> None:
        super().__init__(prompt=GLM_OCR_PROMPT)
