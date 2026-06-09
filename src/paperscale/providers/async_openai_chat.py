"""Async OpenAI-compatible chat/completions provider for the asyncio worker pool.

vLLM serves ``/v1/chat/completions`` synchronously (request/response); workers
``await`` their own responses, so there is no server to poll. The HTTP connection
pool MUST be sized to match ``max_in_flight_requests`` (see
``concurrency-and-queuing.md``), which :func:`build_async_chat_provider` does via
``httpx.Limits``.
"""

from __future__ import annotations

import base64
from typing import Any

from paperscale.providers.base import PageOcrRequest, PageOcrResponse, ProviderError

# Decoding keys the OpenAI chat schema accepts directly; everything else (e.g.
# vLLM's ``repetition_penalty``) is forwarded via ``extra_body``.
_STANDARD_DECODING = frozenset(
    {"temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty", "stop", "seed"}
)


class AsyncOpenAIChatProvider:
    """OpenAI-compatible chat adapter behind provider-neutral requests (async)."""

    name = "openai-compatible-chat"

    def __init__(self, *, client: Any) -> None:
        self._client = client

    async def send(self, request: PageOcrRequest) -> PageOcrResponse:
        image_url = _data_url(request.image_media_type, request.image_bytes)
        standard, extra = _split_decoding(request.decoding)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            **standard,
        }
        if extra:
            payload["extra_body"] = extra
        try:
            response = await self._client.chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - mapped to provider-neutral error
            raise ProviderError(f"OpenAI-compatible chat request failed: {exc}") from exc

        markdown = _extract_chat_text(response)
        return PageOcrResponse(
            markdown=markdown,
            provider_request_id=_response_id(response, request.fingerprint),
            raw=response,
            metadata={"provider": self.name, "model": request.model},
        )


def build_async_chat_provider(
    base_url: str,
    *,
    max_connections: int,
    api_key: str = "paperscale-local",
    timeout: float = 120.0,
) -> AsyncOpenAIChatProvider:
    """Build an async provider whose HTTP pool matches the in-flight ceiling."""

    import httpx
    from openai import AsyncOpenAI

    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    return AsyncOpenAIChatProvider(client=client)


def _split_decoding(decoding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    standard: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in decoding.items():
        (standard if key in _STANDARD_DECODING else extra)[key] = value
    return standard, extra


def _data_url(media_type: str, image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _extract_chat_text(response: object) -> str:
    """Extract chat content as a string.

    Returns the content even when it is the empty string, and maps a ``None``
    completion to ``""`` so an empty completion is classified downstream as an
    ``empty_output`` *content* failure (retryable, never throttling) rather than a
    transport error that would wrongly trip the circuit breaker. Only a structurally
    malformed response (no choices / no message) is a transport-level ``ProviderError``.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        raise ProviderError("provider response had no choices")
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    if message is None:
        raise ProviderError("provider response had no message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""  # empty completion -> empty_output content failure, not transport error
    raise ProviderError("provider response chat content was not text")


def _response_id(response: object, fallback: str) -> str:
    rid = getattr(response, "id", None)
    if isinstance(rid, str) and rid:
        return rid
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    return fallback
