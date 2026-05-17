from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol


JsonMapping = dict[str, Any]


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_fingerprint(parts: JsonMapping) -> str:
    payload = _stable_json(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PageOcrRequest:
    """Provider-neutral request for one page image -> Markdown OCR."""

    page_id: str
    provider: str
    model: str
    profile_name: str
    profile_version: str
    prompt_hash: str
    parser_version: str
    image_hash: str
    prompt: str
    image_media_type: str
    image_bytes: bytes = field(repr=False, compare=False)
    decoding: JsonMapping = field(default_factory=dict)
    render_options: JsonMapping = field(default_factory=dict)
    provider_options: JsonMapping = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Request key sensitive to all replay-relevant provider/profile inputs."""

        return stable_fingerprint(
            {
                "schema_version": 1,
                "page_id": self.page_id,
                "provider": self.provider,
                "model": self.model,
                "profile_name": self.profile_name,
                "profile_version": self.profile_version,
                "prompt_hash": self.prompt_hash,
                "parser_version": self.parser_version,
                "image_hash": self.image_hash,
                "decoding": self.decoding,
                "render_options": self.render_options,
                "provider_options": self.provider_options,
            }
        )


@dataclass(frozen=True)
class PageOcrResponse:
    """Provider-neutral response after transport, before durable page commit."""

    markdown: str
    provider_request_id: str
    raw: object | None = None
    metadata: JsonMapping = field(default_factory=dict)


class ProviderError(RuntimeError):
    """Raised when provider transport fails before profile parsing/validation."""


class PageOcrProvider(Protocol):
    name: str

    def send(self, request: PageOcrRequest) -> PageOcrResponse:
        """Send a provider-neutral request and return provider-neutral output."""
