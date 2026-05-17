from __future__ import annotations

import unittest

from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    ProviderCapacityProfile,
    ProviderOverloadController,
    SelfHostedOpenAICompatibleProvider,
    builtin_capacity_profile,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class FakeHttpClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, str, float]] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.calls.append(("GET", url, timeout))
        return self.responses.pop(0)


class SelfHostedProviderProfileTests(unittest.TestCase):
    def test_vllm_models_health_check_verifies_served_model(self) -> None:
        SelfHostedOpenAICompatibleProvider = require_symbol(
            "paperscale.providers.self_hosted", "SelfHostedOpenAICompatibleProvider"
        )
        InferenceServerProfile = require_symbol("paperscale.providers.self_hosted", "InferenceServerProfile")
        profile = InferenceServerProfile(
            endpoint="http://localhost:8000/v1",
            health_check_path="/models",
            served_model="zai-org/GLM-OCR",
            auth_mode="none",
        )
        client = FakeHttpClient(models=["zai-org/GLM-OCR"])
        provider = SelfHostedOpenAICompatibleProvider(profile=profile, http_client=client)
        self.assertTrue(provider.health_check().ok)
        self.assertEqual(client.calls, ["/models"])

    def test_local_vllm_small_capacity_defaults_are_conservative(self) -> None:
        builtin_capacity_profile = require_symbol("paperscale.providers.self_hosted", "builtin_capacity_profile")
        profile = builtin_capacity_profile("local-vllm-small")
        self.assertLessEqual(profile.max_in_flight_requests, 2)
        self.assertLessEqual(profile.max_provider_side_queued_requests, 4)
        self.assertLessEqual(profile.render_target_longest_side, 1280)
        self.assertGreaterEqual(profile.timeout_seconds, 30)

    def test_overload_trips_circuit_breaker_without_growing_queue(self) -> None:
        ProviderCircuitBreaker = require_symbol("paperscale.providers.self_hosted", "ProviderCircuitBreaker")
        breaker = ProviderCircuitBreaker(queue_size=2, overload_threshold=3)
        for _ in range(3):
            breaker.record_overload(status_code=429)
        self.assertTrue(breaker.is_open)
        self.assertLessEqual(breaker.queued_requests, 2)

    def test_provider_overload_opens_circuit_without_growing_queue_beyond_capacity(self) -> None:
        capacity = ProviderCapacityProfile(
            name="tiny",
            max_in_flight_requests=1,
            max_provider_queue=2,
            queue_size=2,
            timeout_seconds=10.0,
            backoff_initial_seconds=0.5,
            backoff_max_seconds=3.0,
            circuit_breaker_failure_threshold=2,
            circuit_breaker_cooldown_seconds=10.0,
            render_target_longest_side=800,
        )
        controller = ProviderOverloadController(capacity)

        self.assertTrue(controller.try_enqueue())
        self.assertTrue(controller.try_enqueue())
        self.assertFalse(controller.try_enqueue())
        first = controller.record_status(429)
        second = controller.record_status(503)

        self.assertTrue(first.should_retry)
        self.assertEqual(first.backoff_seconds, 0.5)
        self.assertFalse(first.circuit_open)
        self.assertFalse(second.should_retry)
        self.assertEqual(second.backoff_seconds, 1.0)
        self.assertTrue(second.circuit_open)
        self.assertEqual(controller.queued_requests, 2)


if __name__ == "__main__":
    unittest.main()
