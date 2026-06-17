"""LightOnOCR-2 adapter: image-only prompting, Markdown output.

LightOnOCR-2-1B (https://huggingface.co/lightonai/LightOnOCR-2-1B) bakes the
transcription instruction into the model, so the request carries the page image
and *no* text prompt. Output is plain Markdown, so parsing reuses the generic
Markdown path. Selected with ``--ocr-model lightonocr2``.

The ``-ocr-soup`` checkpoint is a task-arithmetic merge of the base and
RLVR-trained weights for extra robustness; its usage is identical, so the soup
adapter only swaps the served model name (``--ocr-model lightonocr2-soup``). The
bbox / bbox-soup variants emit bounding boxes (a different output format) and are
out of scope here.
"""

from __future__ import annotations

from paperscale.models.markdown import MarkdownModel


class LightOnOCRModel(MarkdownModel):
    """Drives LightOnOCR-2, which transcribes a bare page image to Markdown."""

    default_model_name = "lightonai/LightOnOCR-2-1B"
    preferred_longest_image_dim = 1540

    def build_messages(self, image_base64: str) -> list[dict]:
        # Image only: the model is trained to transcribe the page with no text
        # instruction and no system prompt.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

    def sampling_params(self) -> dict:
        # LightOnOCR-2's recommended decoding. Temperature stays pipeline-owned.
        return {"top_p": 0.9}


class LightOnOCRSoupModel(LightOnOCRModel):
    """LightOnOCR-2 ``-ocr-soup``: a merged checkpoint for extra robustness.

    Usage is identical to the base adapter (image-only prompt, Markdown output,
    top_p=0.9, 1540px render); only the served checkpoint differs.
    """

    default_model_name = "lightonai/LightOnOCR-2-1B-ocr-soup"
