from __future__ import annotations

import base64
from typing import Any

from paperscale.providers.base import PageOcrRequest, PageOcrResponse, ProviderError


class OpenAIChatProvider:
    """OpenAI-compatible chat/Responses adapter behind provider-neutral requests."""

    name = "openai-compatible-chat"

    def __init__(self, *, client: Any) -> None:
        self._client = client

    def send(self, request: PageOcrRequest) -> PageOcrResponse:
        image_url = _data_url(request.image_media_type, request.image_bytes)
        payload = {
            "model": request.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            **request.decoding,
        }
        try:
            response = self._client.responses.create(**payload)
        except Exception as exc:  # pragma: no cover - client-specific exception taxonomy later
            raise ProviderError(f"OpenAI-compatible chat request failed: {exc}") from exc

        markdown = _extract_output_text(response)
        return PageOcrResponse(
            markdown=markdown,
            provider_request_id=request.fingerprint,
            raw=response,
            metadata={"provider": self.name, "model": request.model},
        )


def _data_url(media_type: str, image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _extract_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(response, dict) and isinstance(response.get("output_text"), str):
        return response["output_text"]
    raise ProviderError("provider response did not include text output")
