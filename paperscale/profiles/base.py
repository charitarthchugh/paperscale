"""Model OCR profile contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from paperscale.providers.base import PageOcrRequest


@dataclass(frozen=True, slots=True)
class ParsedOcrResult:
    ok: bool
    markdown: str
    retry_classification: str = "none"
    metadata: dict[str, Any] | None = None
    issues: tuple[str, ...] = ()


class ModelOcrProfile(Protocol):
    name: str
    version: str
    task: str
    prompt_template: str
    public_modes: tuple[str, ...]

    def build_request(self, page_id: str, image_bytes: bytes, image_media_type: str) -> PageOcrRequest: ...

    def parse_and_validate(self, output: str) -> ParsedOcrResult: ...
