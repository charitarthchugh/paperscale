"""Self-hosted OpenAI-compatible provider profiles and health checks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class InferenceServerProfile:
    endpoint: str
    served_model: str
    health_path: str = "/models"
    auth_mode: str = "none"
    timeout_seconds: float = 30.0
    request_format: str = "openai-compatible-chat"

    def health_url(self) -> str:
        return f"{self.endpoint.rstrip('/')}/{self.health_path.lstrip('/')}"


@dataclass(frozen=True, slots=True)
class ProviderCapacityProfile:
    name: str
    max_in_flight_requests: int
    max_provider_queue: int
    queue_size: int
    timeout_seconds: float
    backoff_initial_seconds: float
    backoff_max_seconds: float
    circuit_breaker_failure_threshold: int
    circuit_breaker_cooldown_seconds: float
    render_target_longest_side: int
    image_format: str = "png"

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    ok: bool
    served_model: str | None
    diagnostic: str


_BUILTIN_CAPACITY: dict[str, ProviderCapacityProfile] = {
    "local-vllm-small": ProviderCapacityProfile(
        name="local-vllm-small",
        max_in_flight_requests=2,
        max_provider_queue=4,
        queue_size=4,
        timeout_seconds=90.0,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=30.0,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_cooldown_seconds=30.0,
        render_target_longest_side=1280,
    ),
    "local-vllm-large": ProviderCapacityProfile(
        name="local-vllm-large",
        max_in_flight_requests=8,
        max_provider_queue=16,
        queue_size=16,
        timeout_seconds=120.0,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=20.0,
        circuit_breaker_failure_threshold=5,
        circuit_breaker_cooldown_seconds=15.0,
        render_target_longest_side=2048,
    ),
    "remote-openai-compatible": ProviderCapacityProfile(
        name="remote-openai-compatible",
        max_in_flight_requests=4,
        max_provider_queue=8,
        queue_size=8,
        timeout_seconds=180.0,
        backoff_initial_seconds=2.0,
        backoff_max_seconds=60.0,
        circuit_breaker_failure_threshold=4,
        circuit_breaker_cooldown_seconds=60.0,
        render_target_longest_side=1536,
    ),
}


def builtin_capacity_profile(name: str) -> ProviderCapacityProfile:
    try:
        return _BUILTIN_CAPACITY[name]
    except KeyError as exc:
        raise ValueError(f"unknown capacity profile {name!r}") from exc


class SelfHostedOpenAICompatibleProvider:
    def __init__(self, server: InferenceServerProfile, capacity: ProviderCapacityProfile, *, http_client) -> None:
        self.server = server
        self.capacity = capacity
        self.http_client = http_client

    def health_check(self) -> HealthCheckResult:
        response = self.http_client.get(self.server.health_url(), timeout=self.server.timeout_seconds)
        if getattr(response, "status_code", None) != 200:
            return HealthCheckResult(False, None, f"health check failed with status {response.status_code}")
        payload: dict[str, Any] = response.json()
        models = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        if self.server.served_model not in models:
            return HealthCheckResult(
                False,
                None,
                f"served model {self.server.served_model!r} not present in /models response: {models!r}",
            )
        return HealthCheckResult(True, self.server.served_model, "ok")
