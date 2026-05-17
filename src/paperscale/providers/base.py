"""Provider-neutral OCR request and response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import json
from typing import Any


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class PageOcrRequest:
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
    image_bytes: bytes
    decoding: dict[str, Any] = field(default_factory=dict)
    render_options: dict[str, Any] = field(default_factory=dict)

    @property
    def image_data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.image_media_type};base64,{encoded}"

    @property
    def fingerprint(self) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "prompt_hash": self.prompt_hash,
            "parser_version": self.parser_version,
            "image_hash": self.image_hash,
            "decoding": self.decoding,
            "render_options": self.render_options,
        }
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProviderOcrResponse:
    markdown: str
    provider_request_id: str
    raw: Any | None = None
