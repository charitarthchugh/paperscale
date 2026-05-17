"""OpenAI-compatible chat/Responses provider adapter."""

from __future__ import annotations

from paperscale.providers.base import PageOcrRequest, ProviderOcrResponse


class OpenAIChatProvider:
    """Send provider-neutral page OCR requests through an OpenAI-like client."""

    provider_name = "openai-compatible-chat"

    def __init__(self, *, client) -> None:
        self._client = client

    def send(self, request: PageOcrRequest) -> ProviderOcrResponse:
        response = self._client.responses.create(
            model=request.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.prompt},
                        {"type": "input_image", "image_url": request.image_data_url},
                    ],
                }
            ],
            temperature=request.decoding.get("temperature", 0.0),
            top_p=request.decoding.get("top_p"),
            max_output_tokens=request.decoding.get("max_tokens"),
        )
        markdown = getattr(response, "output_text", "")
        return ProviderOcrResponse(
            markdown=markdown,
            provider_request_id=request.fingerprint,
            raw=response,
        )
