"""Built-in document-to-Markdown OCR profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any

from paperscale.profiles.base import ParsedOcrResult
from paperscale.providers.base import PageOcrRequest
from paperscale.quality.verifier import assess_markdown_fragment


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BuiltinOcrProfile:
    name: str
    default_model: str
    prompt_template: str
    prompt_version: str = "v1"
    profile_version: str = "v1"
    parser_version: str = "parser-v1"
    output_format: str = "structured_markdown"
    decoding: dict[str, Any] | None = None
    render_options: dict[str, Any] | None = None
    provider: str = "openai-compatible-chat"
    task: str = "document_to_markdown"
    public_modes: tuple[str, ...] = ("document_to_markdown",)

    @property
    def version(self) -> str:
        return self.profile_version

    def build_request(self, page_id: str, image_bytes: bytes, image_media_type: str) -> PageOcrRequest:
        decoding = dict(self.decoding or {})
        render_options = dict(self.render_options or {})
        prompt = self.prompt_template.format(page_id=page_id)
        prompt_hash = _hash_text(f"{self.name}:{self.prompt_version}:{prompt}")
        return PageOcrRequest(
            page_id=page_id,
            provider=self.provider,
            model=self.default_model,
            profile_name=self.name,
            profile_version=self.profile_version,
            prompt_hash=prompt_hash,
            parser_version=self.parser_version,
            image_hash=_hash_bytes(image_bytes),
            prompt=prompt,
            image_media_type=image_media_type,
            image_bytes=image_bytes,
            decoding=decoding,
            render_options=render_options,
        )

    def parse_and_validate(self, output: str) -> ParsedOcrResult:
        metadata: dict[str, Any] = {}
        markdown = output.strip()
        if self.name == "glm_ocr" and markdown.startswith("{"):
            try:
                payload = json.loads(markdown)
            except json.JSONDecodeError:
                return ParsedOcrResult(False, "", "retryable", issues=("malformed_json",), metadata={})
            markdown = str(payload.get("markdown", "")).strip()
            metadata = {key: value for key, value in payload.items() if key != "markdown"}

        report = assess_markdown_fragment(markdown)
        if not report.accepted:
            return ParsedOcrResult(
                False,
                markdown,
                "retryable",
                metadata=metadata,
                issues=tuple(issue.code for issue in report.issues),
            )
        return ParsedOcrResult(True, markdown, "none", metadata=metadata, issues=())

    def with_overrides(self, **overrides: Any) -> "BuiltinOcrProfile":
        mapped: dict[str, Any] = {}
        for key, value in overrides.items():
            if key == "decoding":
                merged = dict(self.decoding or {})
                merged.update(value)
                mapped[key] = merged
            elif key == "render_options":
                merged = dict(self.render_options or {})
                merged.update(value)
                mapped[key] = merged
            elif key == "prompt_version":
                mapped[key] = value
            else:
                mapped[key] = value
        return replace(self, **mapped)


_PROFILES: dict[str, BuiltinOcrProfile] = {
    "lighton_ocr_2_1b": BuiltinOcrProfile(
        name="lighton_ocr_2_1b",
        default_model="lightonai/LightOnOCR-2-1B",
        prompt_template=(
            "Convert page {page_id} into clean, naturally ordered Markdown. "
            "Preserve tables, forms, math, headings, and reading order. Return Markdown only."
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1280, "image_format": "png", "preprocess": "lighton_conservative"},
    ),
    "deepseek_ocr_2": BuiltinOcrProfile(
        name="deepseek_ocr_2",
        default_model="deepseek-ai/DeepSeek-OCR-2",
        prompt_template=(
            "Convert this document page {page_id} to Markdown. Use document-to-Markdown mode only; "
            "do not use free OCR, QA, or extraction modes. Preserve layout-sensitive text where useful."
        ),
        decoding={"temperature": 0.0, "top_p": 0.95, "max_tokens": 4096, "repetition_penalty": 1.05},
        render_options={"target_longest_side": 1280, "dynamic_resolution": True, "crop_mode": "dynamic"},
    ),
    "glm_ocr": BuiltinOcrProfile(
        name="glm_ocr",
        default_model="zai-org/GLM-OCR",
        prompt_template=(
            "Parse page {page_id} into Markdown for document assembly. Return Markdown, and when available "
            "preserve JSON/layout metadata without changing the document-to-Markdown task contract."
        ),
        output_format="markdown_with_optional_layout_json",
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1536, "context_length_hint": 8192},
    ),
    "generic_vlm_markdown": BuiltinOcrProfile(
        name="generic_vlm_markdown",
        default_model="generic-vlm",
        prompt_template="Convert page {page_id} to Markdown. Return Markdown only.",
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1280, "image_format": "png"},
    ),
    "strict_json_ocr": BuiltinOcrProfile(
        name="strict_json_ocr",
        default_model="generic-vlm-json",
        prompt_template='Convert page {page_id} to Markdown and return JSON: {{"markdown":"..."}}.',
        output_format="json_markdown",
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1280, "image_format": "png"},
    ),
}


def builtin_profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_builtin_profile(name: str) -> BuiltinOcrProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown OCR profile {name!r}") from exc
