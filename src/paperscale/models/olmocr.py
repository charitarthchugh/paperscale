"""olmOCR adapter: the original YAML-front-matter + rotation OCR model.

Preserved so paperscale stays 1:1 with olmOCR when pointed at an olmOCR model.
Selected with ``--ocr-model olmocr``.
"""

from __future__ import annotations

from paperscale.front_matter import FrontMatterParser
from paperscale.models.base import OCRModel
from paperscale.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt

# Constrains output to the olmOCR front-matter shape (see olmocr.pipeline).
_OLMOCR_GUIDED_REGEX = (
    r"---\nprimary_language: (?:[a-z]{2}|null)\nis_rotation_valid: (?:True|False|true|false)\n"
    r"rotation_correction: (?:0|90|180|270)\nis_table: (?:True|False|true|false)\n"
    r"is_diagram: (?:True|False|true|false)\n(?:---|---\n[\s\S]+)"
)


class OlmOCRModel(OCRModel):
    """Drives an olmOCR-style model that returns YAML front matter + Markdown."""

    default_model_name = "olmocr"

    def build_messages(self, image_base64: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

    def guided_regex(self) -> str | None:
        return _OLMOCR_GUIDED_REGEX

    def parse(self, content: str) -> PageResponse:
        parser = FrontMatterParser(front_matter_class=PageResponse)
        front_matter, text = parser._extract_front_matter_and_text(content)
        return parser._parse_front_matter(front_matter, text)
