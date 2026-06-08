from __future__ import annotations

import asyncio
import base64
from binascii import Error as Base64Error
from dataclasses import dataclass, field
import hashlib
import json
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


JsonMapping = dict[str, Any]

SCENARIOS = frozenset(
    {
        "ok_markdown",
        "json_layout",
        "empty_output",
        "refusal",
        "repeated_ngram",
        "malformed_frontmatter",
        "truncated",
        "rate_limit",
        "rate_limit_then_ok",
        "server_error",
        "slow",
    }
)


@dataclass(frozen=True)
class MockApiConfig:
    served_model: str = "mock-vlm"
    scenario: str = "ok_markdown"
    max_image_bytes: int = 10 * 1024 * 1024
    max_in_flight: int = 8
    latency_ms: int = 0
    bearer_token: str | None = None
    allowed_media_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

    def __post_init__(self) -> None:
        if not self.served_model:
            raise ValueError("served_model is required")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown mock API scenario: {self.scenario}")
        if self.max_image_bytes < 0:
            raise ValueError("max_image_bytes must be non-negative")
        if self.max_in_flight < 0:
            raise ValueError("max_in_flight must be non-negative")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")


@dataclass
class MockApiState:
    scenario: str = "ok_markdown"
    requests: list[JsonMapping] = field(default_factory=list)
    _active_requests: int = field(default=0, init=False, repr=False)
    _scenario_hits: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown mock API scenario: {self.scenario}")

    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIOS:
            raise MockApiError(400, f"unknown scenario: {scenario}", "invalid_request_error", "invalid_scenario")
        with self._lock:
            self.scenario = scenario
            self._scenario_hits = 0

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self._scenario_hits = 0

    def try_enter(self, max_in_flight: int) -> bool:
        with self._lock:
            if self._active_requests >= max_in_flight:
                return False
            self._active_requests += 1
            return True

    def leave(self) -> None:
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1

    def current_scenario(self) -> str:
        with self._lock:
            return self.scenario

    def should_rate_limit_then_ok(self) -> bool:
        with self._lock:
            if self.scenario != "rate_limit_then_ok":
                return False
            self._scenario_hits += 1
            return self._scenario_hits == 1

    def record(self, entry: JsonMapping) -> None:
        with self._lock:
            self.requests.append(entry)

    def snapshot_requests(self) -> list[JsonMapping]:
        with self._lock:
            return [dict(entry) for entry in self.requests]


@dataclass(frozen=True)
class ValidatedRequest:
    endpoint: str
    model: str
    prompt: str
    prompt_sha256: str
    image_sha256: str
    image_media_type: str
    image_bytes: bytes = field(repr=False)
    request_id: str


class MockApiError(Exception):
    def __init__(self, status_code: int, message: str, error_type: str, code: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code


def create_app(config: MockApiConfig | None = None, *, state: MockApiState | None = None) -> FastAPI:
    config = config or MockApiConfig()
    state = state or MockApiState(scenario=config.scenario)
    app = FastAPI(title="Paperscale Mock Inference API", version="0.1.0")

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        auth_error = _auth_error(request, config)
        if auth_error is not None:
            return auth_error
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": config.served_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "paperscale-mock-api",
                    }
                ],
            }
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> JSONResponse:
        return await _handle_inference(request, endpoint="responses", config=config, state=state)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await _handle_inference(request, endpoint="chat.completions", config=config, state=state)

    @app.get("/__paperscale/requests")
    async def request_log(request: Request) -> JSONResponse:
        auth_error = _auth_error(request, config)
        if auth_error is not None:
            return auth_error
        return JSONResponse({"requests": state.snapshot_requests()})

    @app.post("/__paperscale/reset")
    async def reset(request: Request) -> JSONResponse:
        auth_error = _auth_error(request, config)
        if auth_error is not None:
            return auth_error
        state.reset()
        return JSONResponse({"ok": True, "requests": []})

    @app.post("/__paperscale/scenario")
    async def scenario(request: Request) -> JSONResponse:
        auth_error = _auth_error(request, config)
        if auth_error is not None:
            return auth_error
        try:
            payload = await _read_json_body(request)
            scenario_name = payload.get("scenario") if isinstance(payload, dict) else None
            if not isinstance(scenario_name, str):
                raise MockApiError(400, "scenario must be a string", "invalid_request_error", "invalid_scenario")
            state.set_scenario(scenario_name)
            return JSONResponse({"ok": True, "scenario": scenario_name})
        except MockApiError as exc:
            return _openai_error(exc)

    return app


async def _handle_inference(
    request: Request,
    *,
    endpoint: str,
    config: MockApiConfig,
    state: MockApiState,
) -> JSONResponse:
    auth_error = _auth_error(request, config)
    if auth_error is not None:
        return auth_error
    if not state.try_enter(config.max_in_flight):
        return _openai_error(
            MockApiError(429, "mock inference server is overloaded", "rate_limit_error", "server_overloaded")
        )
    try:
        try:
            payload = await _read_json_body(request)
            validated = _validate_payload(payload, endpoint=endpoint, config=config)
            scenario = state.current_scenario()
            if scenario == "rate_limit" or state.should_rate_limit_then_ok():
                return _openai_error(
                    MockApiError(429, "mock scenario rate limit", "rate_limit_error", "rate_limit_exceeded")
                )
            if scenario == "server_error":
                return _openai_error(
                    MockApiError(500, "mock scenario server error", "server_error", "server_error")
                )
            delay_ms = config.latency_ms
            if scenario == "slow" and delay_ms == 0:
                delay_ms = 250
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)
            markdown = _scenario_output(scenario, validated)
            state.record(_request_log_entry(validated, scenario=scenario))
            if endpoint == "responses":
                return JSONResponse(_responses_payload(validated, markdown))
            return JSONResponse(_chat_payload(validated, markdown))
        except MockApiError as exc:
            return _openai_error(exc)
    finally:
        state.leave()


async def _read_json_body(request: Request) -> JsonMapping:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise MockApiError(400, "request body must be valid JSON", "invalid_request_error", "invalid_json") from exc
    if not isinstance(payload, dict):
        raise MockApiError(400, "request body must be a JSON object", "invalid_request_error", "invalid_json")
    return payload


def _validate_payload(payload: JsonMapping, *, endpoint: str, config: MockApiConfig) -> ValidatedRequest:
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise MockApiError(400, "model is required", "invalid_request_error", "missing_model")
    if model != config.served_model:
        raise MockApiError(
            404,
            f"model {model!r} is not served by this mock API",
            "invalid_request_error",
            "model_not_found",
        )
    _validate_decoding(payload)
    if endpoint == "responses":
        prompt, image_url = _extract_responses_inputs(payload)
    else:
        prompt, image_url = _extract_chat_inputs(payload)
    image_media_type, image_bytes = _decode_image(image_url, config=config)
    prompt_sha256 = _sha256_text(prompt)
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    request_id = _request_id(
        {
            "endpoint": endpoint,
            "model": model,
            "prompt_sha256": prompt_sha256,
            "image_sha256": image_sha256,
        }
    )
    return ValidatedRequest(
        endpoint=endpoint,
        model=model,
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        image_sha256=image_sha256,
        image_media_type=image_media_type,
        image_bytes=image_bytes,
        request_id=request_id,
    )


def _validate_decoding(payload: JsonMapping) -> None:
    temperature = payload.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, int | float) or temperature < 0 or temperature > 2
    ):
        raise MockApiError(400, "temperature must be between 0 and 2", "invalid_request_error", "invalid_temperature")
    top_p = payload.get("top_p")
    if top_p is not None and (not isinstance(top_p, int | float) or top_p <= 0 or top_p > 1):
        raise MockApiError(400, "top_p must be greater than 0 and at most 1", "invalid_request_error", "invalid_top_p")
    for token_field in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        token_limit = payload.get(token_field)
        if token_limit is not None and (not isinstance(token_limit, int) or token_limit < 1):
            raise MockApiError(400, f"{token_field} must be a positive integer", "invalid_request_error", f"invalid_{token_field}")


def _extract_responses_inputs(payload: JsonMapping) -> tuple[str, str]:
    input_items = payload.get("input")
    if not isinstance(input_items, list):
        raise MockApiError(400, "input must be a list", "invalid_request_error", "invalid_input")
    text_parts: list[str] = []
    image_url = ""
    for item in input_items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "input_text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            if part_type == "input_image" and isinstance(part.get("image_url"), str):
                image_url = part["image_url"]
    if not image_url:
        raise MockApiError(400, "request must include an input_image", "invalid_request_error", "missing_image")
    return "\n".join(text_parts), image_url


def _extract_chat_inputs(payload: JsonMapping) -> tuple[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise MockApiError(400, "messages must be a list", "invalid_request_error", "invalid_messages")
    text_parts: list[str] = []
    image_url = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
            if part_type == "image_url":
                image = part.get("image_url")
                if isinstance(image, str):
                    image_url = image
                elif isinstance(image, dict) and isinstance(image.get("url"), str):
                    image_url = image["url"]
    if not image_url:
        raise MockApiError(400, "request must include an image_url", "invalid_request_error", "missing_image")
    return "\n".join(text_parts), image_url


def _decode_image(image_url: str, *, config: MockApiConfig) -> tuple[str, bytes]:
    prefix = "data:"
    marker = ";base64,"
    if not image_url.startswith(prefix) or marker not in image_url:
        raise MockApiError(400, "image URL must be a base64 data URL", "invalid_request_error", "invalid_image_url")
    media_type, encoded = image_url[len(prefix) :].split(marker, 1)
    if media_type not in config.allowed_media_types:
        raise MockApiError(400, f"unsupported image media type: {media_type}", "invalid_request_error", "unsupported_image_media_type")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (Base64Error, ValueError) as exc:
        raise MockApiError(400, "image data URL is not valid base64", "invalid_request_error", "invalid_image_base64") from exc
    if len(image_bytes) > config.max_image_bytes:
        raise MockApiError(413, "image exceeds configured byte limit", "invalid_request_error", "image_too_large")
    return media_type, image_bytes


def _scenario_output(scenario: str, request: ValidatedRequest) -> str:
    base = _stable_markdown(request)
    if scenario in {"ok_markdown", "slow", "rate_limit_then_ok"}:
        return base
    if scenario == "json_layout":
        return json.dumps(
            {
                "markdown": base,
                "regions": [
                    {"kind": "heading", "bbox": [0, 0, 100, 24]},
                    {"kind": "paragraph", "bbox": [0, 30, 320, 120]},
                ],
                "image_sha256": request.image_sha256,
            },
            sort_keys=True,
        )
    if scenario == "empty_output":
        return ""
    if scenario == "refusal":
        return "I'm sorry, I can't assist with OCR for this image."
    if scenario == "repeated_ngram":
        return "same same same same"
    if scenario == "malformed_frontmatter":
        return "---\ntitle: [mock ocr\n---\n# Mock OCR\n\nMalformed frontmatter scenario."
    if scenario == "truncated":
        return base[:96]
    return base


def _stable_markdown(request: ValidatedRequest) -> str:
    return (
        "# Mock OCR Page\n\n"
        f"- request_id: {request.request_id}\n"
        f"- model: {request.model}\n"
        f"- endpoint: {request.endpoint}\n"
        f"- image_sha256: {request.image_sha256[:16]}\n"
        f"- prompt_sha256: {request.prompt_sha256[:16]}\n"
    )


def _request_log_entry(request: ValidatedRequest, *, scenario: str) -> JsonMapping:
    return {
        "request_id": request.request_id,
        "endpoint": request.endpoint,
        "model": request.model,
        "scenario": scenario,
        "image_sha256": request.image_sha256,
        "prompt_sha256": request.prompt_sha256,
        "image_media_type": request.image_media_type,
        "image_bytes": len(request.image_bytes),
    }


def _responses_payload(request: ValidatedRequest, markdown: str) -> JsonMapping:
    created_at = int(time.time())
    return {
        "id": request.request_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": request.model,
        "output_text": markdown,
        "output": [
            {
                "id": f"msg_{request.request_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": markdown,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": max(1, len(markdown.split())), "total_tokens": max(2, len(markdown.split()) + 1)},
    }


def _chat_payload(request: ValidatedRequest, markdown: str) -> JsonMapping:
    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": markdown},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": max(1, len(markdown.split())), "total_tokens": max(2, len(markdown.split()) + 1)},
    }


def _auth_error(request: Request, config: MockApiConfig) -> JSONResponse | None:
    if config.bearer_token is None:
        return None
    expected = f"Bearer {config.bearer_token}"
    if request.headers.get("authorization") == expected:
        return None
    return _openai_error(
        MockApiError(401, "missing or invalid API key", "authentication_error", "invalid_api_key"),
        headers={"WWW-Authenticate": "Bearer"},
    )


def _openai_error(error: MockApiError, *, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"message": error.message, "type": error.error_type, "code": error.code}},
        headers=headers,
    )


def _request_id(parts: JsonMapping) -> str:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"mockreq-{hashlib.sha256(payload).hexdigest()[:16]}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
