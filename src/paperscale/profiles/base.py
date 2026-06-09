from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from paperscale.providers.base import PageOcrRequest


@dataclass(frozen=True)
class ProfileValidationResult:
    ok: bool
    markdown: str
    retry_classification: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostic: str = "ok"


@dataclass(frozen=True)
class ModelOcrProfile:
    """Model-specific behavior for the single v1 document-to-Markdown task."""

    name: str
    default_model: str
    prompt_template: str
    prompt_version: str
    parser_version: str
    output_format: str
    decoding: dict[str, Any]
    render_options: dict[str, Any]
    task: str = "document_to_markdown"
    public_modes: tuple[str, ...] = ("document_to_markdown",)
    provider: str = "openai-compatible-chat"
    # Per-dimension remediation ceilings (monotonic escalation cannot exceed these).
    max_decode_tokens: int = 16384
    max_render_longest_side: int = 2600
    max_repetition_penalty: float = 1.5
    # DPI is what the renderer actually honors; bump it alongside target_longest_side
    # so a mojibake remediation produces a genuinely higher-resolution image.
    base_render_dpi: int = 144
    max_render_dpi: int = 400

    def __post_init__(self) -> None:
        if self.task != "document_to_markdown":
            raise ValueError("v1 profiles must use document_to_markdown")
        if self.public_modes != ("document_to_markdown",):
            raise ValueError("v1 profiles must not expose non-Markdown OCR modes")
        if "Markdown" not in self.prompt_template and "markdown" not in self.prompt_template:
            raise ValueError("profile prompt must target Markdown output")

    def build_request(
        self,
        page_id: str,
        image_bytes: bytes,
        image_media_type: str,
        *,
        model: str | None = None,
    ) -> PageOcrRequest:
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        prompt = self.prompt_template.format(page_id=page_id)
        provider_options = {
            "profile_fingerprint": self.profile_fingerprint(),
            "output_format": self.output_format,
        }
        return PageOcrRequest(
            page_id=page_id,
            provider=self.provider,
            model=model or self.default_model,
            profile_name=self.name,
            profile_version=self.prompt_version,
            prompt_hash=_hash_text(prompt),
            parser_version=self.parser_version,
            image_hash=image_hash,
            prompt=prompt,
            image_media_type=image_media_type,
            image_bytes=image_bytes,
            decoding=copy.deepcopy(self.decoding),
            render_options=copy.deepcopy(self.render_options),
            provider_options=provider_options,
        )

    def parse_and_validate(self, output: str) -> ProfileValidationResult:
        markdown = output.strip()
        if not markdown:
            # Canonical diagnostic codes (shared with the verifier + remediation table)
            # so diagnostic-driven remediation can react to parse-stage failures too.
            return ProfileValidationResult(
                ok=False,
                markdown="",
                retry_classification="retryable",
                diagnostic="empty_output",
            )
        if _has_repeated_ngram(markdown):
            return ProfileValidationResult(
                ok=False,
                markdown=markdown,
                retry_classification="retryable",
                diagnostic="repeated_ngram",
            )
        return ProfileValidationResult(ok=True, markdown=markdown)

    def profile_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "schema_version": 1,
                "name": self.name,
                "default_model": self.default_model,
                "prompt_template_hash": _hash_text(self.prompt_template),
                "prompt_version": self.prompt_version,
                "parser_version": self.parser_version,
                "output_format": self.output_format,
                "decoding": self.decoding,
                "render_options": self.render_options,
                "task": self.task,
                "provider": self.provider,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def with_overrides(
        self,
        *,
        prompt_version: str | None = None,
        parser_version: str | None = None,
        decoding: dict[str, Any] | None = None,
        render_options: dict[str, Any] | None = None,
    ) -> "ModelOcrProfile":
        next_decoding = copy.deepcopy(self.decoding)
        if decoding:
            next_decoding.update(decoding)
        next_render = copy.deepcopy(self.render_options)
        if render_options:
            next_render.update(render_options)
        return replace(
            self,
            prompt_version=prompt_version or self.prompt_version,
            parser_version=parser_version or self.parser_version,
            decoding=next_decoding,
            render_options=next_render,
        )


    def remediation_for(
        self, diagnostic: str, *, accumulated: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Map a failure diagnostic to accumulated decoding/render overrides.

        Overrides layer additively onto ``accumulated`` (the effective overrides
        from prior attempts) rather than rebuilding from base, so a token bump for
        a later ``truncation`` does not discard an earlier ``mojibake`` DPI bump.
        Each dimension is clamped to its ceiling; an unknown diagnostic is identity.
        Returns a dict with optional ``decoding``/``render_options`` keys suitable
        for :meth:`with_overrides`.
        """

        prior = copy.deepcopy(accumulated) if accumulated else {}
        decoding = dict(prior.get("decoding") or {})
        render = dict(prior.get("render_options") or {})

        if diagnostic in _TOKEN_DIAGNOSTICS:
            current = int(decoding.get("max_tokens", self.decoding.get("max_tokens", 4096)))
            decoding["max_tokens"] = min(current * 2, self.max_decode_tokens)
        elif diagnostic in _RENDER_DIAGNOSTICS:
            current = int(
                render.get("target_longest_side", self.render_options.get("target_longest_side", 1280))
            )
            bumped = max(current + 1, int(current * 1.25))
            render["target_longest_side"] = min(bumped, self.max_render_longest_side)
            current_dpi = int(render.get("dpi", self.render_options.get("dpi", self.base_render_dpi)))
            render["dpi"] = min(max(current_dpi + 1, int(current_dpi * 1.25)), self.max_render_dpi)
        elif diagnostic in _REPETITION_DIAGNOSTICS:
            # Step by 0.15: empirically the first bump to ~1.15 already breaks the
            # decode loops VLMs fall into on dense pages, so the ladder converges fast.
            current = float(decoding.get("repetition_penalty", 1.0))
            decoding["repetition_penalty"] = min(round(current + 0.15, 3), self.max_repetition_penalty)
        # else: unknown diagnostic -> identity (carry accumulated forward unchanged)

        result: dict[str, Any] = {}
        if decoding:
            result["decoding"] = decoding
        if render:
            result["render_options"] = render
        return result


# Diagnostic -> remediation dimension. Diagnostics come from
# ``ProfileValidationResult.diagnostic`` and ``VerificationFinding.kind``.
_TOKEN_DIAGNOSTICS = frozenset({"truncation_indicator", "length_anomaly"})
# empty_output re-renders at higher DPI: an empty completion on a content-bearing
# page is often a too-small/low-detail image, so escalate resolution on retry rather
# than re-rolling the identical request.
_RENDER_DIAGNOSTICS = frozenset({"mojibake", "control_characters", "empty_output"})
_REPETITION_DIAGNOSTICS = frozenset({"repeated_ngram", "repeated_character"})


class ModelOcrProfileProtocol(Protocol):
    def build_request(
        self, page_id: str, image_bytes: bytes, image_media_type: str, *, model: str | None = None
    ) -> PageOcrRequest: ...

    def parse_and_validate(self, output: str) -> ProfileValidationResult: ...


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _has_repeated_ngram(text: str, *, min_repeats: int = 4) -> bool:
    words = [word.casefold() for word in text.split()]
    if len(words) < min_repeats:
        return False
    run_word = None
    run_count = 0
    for word in words:
        if word == run_word:
            run_count += 1
        else:
            run_word = word
            run_count = 1
        if run_count >= min_repeats:
            return True
    for size in (2, 3):
        if len(words) < size * min_repeats:
            continue
        for start in range(0, len(words) - size * min_repeats + 1):
            phrase = words[start : start + size]
            if all(
                words[start + size * offset : start + size * (offset + 1)] == phrase
                for offset in range(min_repeats)
            ):
                return True
    return False
