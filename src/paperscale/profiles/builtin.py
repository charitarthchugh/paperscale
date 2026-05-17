from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from paperscale.profiles.base import ModelOcrProfile, ProfileValidationResult


class GlmOcrProfile(ModelOcrProfile):
    def parse_and_validate(self, output: str) -> ProfileValidationResult:
        stripped = output.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                return ProfileValidationResult(
                    ok=False,
                    markdown=stripped,
                    retry_classification="retryable",
                    diagnostic="malformed GLM JSON/layout payload",
                )
            markdown = payload.get("markdown")
            if not isinstance(markdown, str) or not markdown.strip():
                return ProfileValidationResult(
                    ok=False,
                    markdown="" if not isinstance(markdown, str) else markdown,
                    retry_classification="retryable",
                    diagnostic="GLM payload missing markdown",
                )
            metadata = {key: value for key, value in payload.items() if key != "markdown"}
            return ProfileValidationResult(
                ok=True, markdown=markdown.strip(), metadata=metadata
            )
        return super().parse_and_validate(output)


def _profile(**kwargs: object) -> ModelOcrProfile:
    return ModelOcrProfile(**kwargs)  # type: ignore[arg-type]


_BUILTINS: dict[str, ModelOcrProfile] = {
    "generic_vlm_markdown": _profile(
        name="generic_vlm_markdown",
        default_model="generic-vlm-markdown",
        prompt_version="v1",
        parser_version="markdown-parser-v1",
        output_format="structured_markdown",
        prompt_template=(
            "Convert document page {page_id} into clean Markdown for ordered "
            "document-level assembly. Return only Markdown; do not answer questions."
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1280, "image_format": "png"},
    ),
    "strict_json_ocr": _profile(
        name="strict_json_ocr",
        default_model="generic-vlm-markdown",
        prompt_version="v1",
        parser_version="strict-json-parser-v1",
        output_format="json_markdown",
        prompt_template=(
            "Convert document page {page_id} into Markdown and return JSON with "
            "a markdown field plus optional metadata only."
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={"target_longest_side": 1280, "image_format": "png"},
    ),
    "lighton_ocr_2_1b": _profile(
        name="lighton_ocr_2_1b",
        default_model="lightonai/LightOnOCR-2-1B",
        prompt_version="v1",
        parser_version="lighton-markdown-parser-v1",
        output_format="structured_markdown",
        prompt_template=(
            "Using LightOnOCR-2-1B guidance, convert document page {page_id} "
            "into clean, naturally ordered Markdown. Preserve headings, tables, "
            "forms, math, and reading order. Return only Markdown."
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={
            "target_longest_side": 1280,
            "image_format": "png",
            "preprocessing": "lighton_conservative",
        },
    ),
    "deepseek_ocr_2": _profile(
        name="deepseek_ocr_2",
        default_model="deepseek-ai/DeepSeek-OCR-2",
        prompt_version="v1",
        parser_version="deepseek-markdown-parser-v1",
        output_format="structured_markdown",
        prompt_template=(
            "Convert document page {page_id} to Markdown using DeepSeek-OCR-2 "
            "document parsing mode. Preserve document structure and avoid repeated "
            "tokens. Return only Markdown; free OCR mode is not available."
        ),
        decoding={
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "frequency_penalty": 0.2,
        },
        render_options={
            "target_longest_side": 1600,
            "image_format": "png",
            "dynamic_resolution": True,
            "crop_mode": "dynamic",
        },
    ),
    "glm_ocr": GlmOcrProfile(
        name="glm_ocr",
        default_model="zai-org/GLM-OCR",
        prompt_version="v1",
        parser_version="glm-markdown-layout-parser-v1",
        output_format="markdown_with_optional_layout_metadata",
        prompt_template=(
            "Use GLM-OCR document parsing behavior to convert page {page_id} "
            "into Markdown for ordered assembly. If available, include layout "
            "metadata in JSON, but preserve Markdown as the primary page artifact."
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_tokens": 4096},
        render_options={
            "target_longest_side": 1536,
            "image_format": "png",
            "deployment_routes": ["hosted", "vllm", "sglang", "ollama", "sdk-server"],
            "context_length_hint": 8192,
        },
    ),
}


def builtin_profile_names() -> tuple[str, ...]:
    return tuple(_BUILTINS)


def get_builtin_profile(name: str) -> ModelOcrProfile:
    try:
        return _BUILTINS[name]
    except KeyError as exc:
        raise ValueError(f"unknown OCR profile: {name}") from exc


def with_profile_override(name: str, **overrides: Any) -> ModelOcrProfile:
    return replace(get_builtin_profile(name), **overrides)
