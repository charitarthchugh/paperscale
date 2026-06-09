from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol
from urllib.parse import urljoin

from paperscale.providers.base import stable_fingerprint


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    retry_statuses: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)


@dataclass(frozen=True)
class InferenceServerProfile:
    """Self-hosted/OpenAI-compatible inference server configuration."""

    endpoint: str
    served_model: str
    request_format: str = "openai-chat-responses"
    health_path: str = "/models"
    auth_mode: str = "none"
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = RetryPolicy()
    overload_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("endpoint is required")
        if not self.served_model:
            raise ValueError("served_model is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @property
    def normalized_endpoint(self) -> str:
        return self.endpoint.rstrip("/") + "/"

    @property
    def health_url(self) -> str:
        return urljoin(self.normalized_endpoint, self.health_path.lstrip("/"))

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": 1,
                "endpoint": self.normalized_endpoint,
                "served_model": self.served_model,
                "request_format": self.request_format,
                "health_path": self.health_path,
                "auth_mode": self.auth_mode,
                "timeout_seconds": self.timeout_seconds,
                "retry_statuses": self.retry_policy.retry_statuses,
                "overload_statuses": self.overload_statuses,
            }
        )


@dataclass(frozen=True)
class ProviderCapacityProfile:
    """Scheduler-visible capacity limits for provider pressure control."""

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

    def __post_init__(self) -> None:
        positive_ints = {
            "max_in_flight_requests": self.max_in_flight_requests,
            "max_provider_queue": self.max_provider_queue,
            "queue_size": self.queue_size,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "render_target_longest_side": self.render_target_longest_side,
        }
        for field_name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.queue_size > self.max_provider_queue:
            raise ValueError("queue_size must not exceed max_provider_queue")
        if self.timeout_seconds <= 0 or self.backoff_initial_seconds <= 0:
            raise ValueError("timeout/backoff values must be positive")
        if self.backoff_max_seconds < self.backoff_initial_seconds:
            raise ValueError("backoff_max_seconds must be >= backoff_initial_seconds")

    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": 1,
                "name": self.name,
                "max_in_flight_requests": self.max_in_flight_requests,
                "max_provider_queue": self.max_provider_queue,
                "queue_size": self.queue_size,
                "timeout_seconds": self.timeout_seconds,
                "backoff_initial_seconds": self.backoff_initial_seconds,
                "backoff_max_seconds": self.backoff_max_seconds,
                "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
                "circuit_breaker_cooldown_seconds": self.circuit_breaker_cooldown_seconds,
                "render_target_longest_side": self.render_target_longest_side,
                "image_format": self.image_format,
            }
        )

    def with_timeout_from_server(self, server: InferenceServerProfile) -> "ProviderCapacityProfile":
        return replace(self, timeout_seconds=max(self.timeout_seconds, server.timeout_seconds))


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
        render_target_longest_side=1920,
    ),
    "remote-openai-compatible": ProviderCapacityProfile(
        name="remote-openai-compatible",
        max_in_flight_requests=4,
        max_provider_queue=8,
        queue_size=8,
        timeout_seconds=60.0,
        backoff_initial_seconds=1.0,
        backoff_max_seconds=60.0,
        circuit_breaker_failure_threshold=4,
        circuit_breaker_cooldown_seconds=60.0,
        render_target_longest_side=1600,
    ),
}


def builtin_capacity_profile(name: str) -> ProviderCapacityProfile:
    try:
        return _BUILTIN_CAPACITY[name]
    except KeyError as exc:
        raise ValueError(f"unknown capacity profile: {name}") from exc


def builtin_capacity_profile_names() -> tuple[str, ...]:
    return tuple(_BUILTIN_CAPACITY)


@dataclass(frozen=True)
class BackoffDecision:
    should_retry: bool
    backoff_seconds: float
    circuit_open: bool
    diagnostic: str


class ProviderOverloadController:
    """Small deterministic retry/circuit helper for scheduler-visible provider pressure."""

    def __init__(self, capacity: ProviderCapacityProfile) -> None:
        self.capacity = capacity
        self._consecutive_failures = 0
        self._queued_requests = 0

    @property
    def queued_requests(self) -> int:
        return self._queued_requests

    @property
    def circuit_open(self) -> bool:
        return self._consecutive_failures >= self.capacity.circuit_breaker_failure_threshold

    def try_enqueue(self) -> bool:
        if self._queued_requests >= self.capacity.queue_size:
            return False
        self._queued_requests += 1
        return True

    def mark_dequeued(self) -> None:
        if self._queued_requests > 0:
            self._queued_requests -= 1

    def record_success(self) -> BackoffDecision:
        self._consecutive_failures = 0
        return BackoffDecision(False, 0.0, False, "success")

    def record_status(self, status_code: int) -> BackoffDecision:
        if status_code in (429, 500, 502, 503, 504):
            self._consecutive_failures += 1
            backoff = min(
                self.capacity.backoff_initial_seconds * (2 ** (self._consecutive_failures - 1)),
                self.capacity.backoff_max_seconds,
            )
            return BackoffDecision(
                should_retry=not self.circuit_open,
                backoff_seconds=backoff,
                circuit_open=self.circuit_open,
                diagnostic=f"provider overload HTTP {status_code}",
            )
        self._consecutive_failures = 0
        return BackoffDecision(False, 0.0, False, f"non-retryable HTTP {status_code}")

    def record_failure(self, *, retryable: bool) -> BackoffDecision:
        """Account for a provider transport failure surfaced as an exception.

        Transport-agnostic counterpart to record_status: callers that only see
        exceptions (no HTTP status) use this to drive backoff and the circuit breaker.
        """
        if not retryable:
            self._consecutive_failures = 0
            return BackoffDecision(False, 0.0, False, "non-retryable provider error")
        self._consecutive_failures += 1
        backoff = min(
            self.capacity.backoff_initial_seconds * (2 ** (self._consecutive_failures - 1)),
            self.capacity.backoff_max_seconds,
        )
        return BackoffDecision(
            should_retry=not self.circuit_open,
            backoff_seconds=backoff,
            circuit_open=self.circuit_open,
            diagnostic="provider transport error",
        )


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float) -> Any: ...


@dataclass(frozen=True)
class HealthCheckResult:
    ok: bool
    endpoint: str
    served_model: str
    diagnostic: str
    observed_models: tuple[str, ...] = ()


class SelfHostedOpenAICompatibleProvider:
    """Self-hosted vLLM/OpenAI-compatible profile helper with fakeable health checks."""

    name = "self-hosted-openai-compatible"

    def __init__(
        self,
        server: InferenceServerProfile,
        capacity: ProviderCapacityProfile,
        *,
        http_client: HttpClient,
    ) -> None:
        self.server = server
        self.capacity = capacity.with_timeout_from_server(server)
        self._http_client = http_client

    def health_check(self) -> HealthCheckResult:
        try:
            response = self._http_client.get(
                self.server.health_url, timeout=self.server.timeout_seconds
            )
        except Exception as exc:  # pragma: no cover - exact HTTP client exceptions vary
            return HealthCheckResult(
                ok=False,
                endpoint=self.server.normalized_endpoint,
                served_model=self.server.served_model,
                diagnostic=f"health check failed: {exc}",
            )

        status_code = getattr(response, "status_code", 0)
        if status_code != 200:
            return HealthCheckResult(
                ok=False,
                endpoint=self.server.normalized_endpoint,
                served_model=self.server.served_model,
                diagnostic=f"health check returned HTTP {status_code}",
            )

        payload = response.json()
        observed_models = _extract_model_ids(payload)
        if self.server.served_model not in observed_models:
            return HealthCheckResult(
                ok=False,
                endpoint=self.server.normalized_endpoint,
                served_model=self.server.served_model,
                diagnostic=(
                    f"served model {self.server.served_model!r} not present in /models: "
                    f"{', '.join(observed_models) or '<none>'}"
                ),
                observed_models=observed_models,
            )
        return HealthCheckResult(
            ok=True,
            endpoint=self.server.normalized_endpoint,
            served_model=self.server.served_model,
            diagnostic="ok",
            observed_models=observed_models,
        )

    def profile_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": 1,
                "provider": self.name,
                "server": self.server.fingerprint(),
                "capacity": self.capacity.fingerprint(),
            }
        )


def _extract_model_ids(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data", [])
    if not isinstance(data, list):
        return ()
    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_ids.append(item["id"])
    return tuple(model_ids)
