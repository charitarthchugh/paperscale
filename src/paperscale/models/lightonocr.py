"""LightOnOCR-2 adapter: image-only prompting, Markdown output.

LightOnOCR-2-1B (https://huggingface.co/lightonai/LightOnOCR-2-1B) bakes the
transcription instruction into the model, so the request carries the page image
and *no* text prompt. Output is plain Markdown, so parsing reuses the generic
Markdown path. Selected with ``--ocr-model lightonocr2``.
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
