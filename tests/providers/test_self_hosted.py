from __future__ import annotations

import unittest

from paperscale.providers.self_hosted import (
    InferenceServerProfile,
    ProviderCapacityProfile,
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
        server = InferenceServerProfile(
            endpoint="http://localhost:8000/v1",
            served_model="lightonai/LightOnOCR-2-1B",
            health_path="/models",
            timeout_seconds=3.0,
        )
        capacity = builtin_capacity_profile("local-vllm-small")
        client = FakeHttpClient(
            [FakeResponse(200, {"data": [{"id": "lightonai/LightOnOCR-2-1B"}]})]
        )

        provider = SelfHostedOpenAICompatibleProvider(server, capacity, http_client=client)
        result = provider.health_check()

        self.assertTrue(result.ok)
        self.assertEqual(result.served_model, "lightonai/LightOnOCR-2-1B")
        self.assertEqual(client.calls, [("GET", "http://localhost:8000/v1/models", 3.0)])

    def test_vllm_models_health_check_fails_for_wrong_model_without_network_retry(self) -> None:
        server = InferenceServerProfile(
            endpoint="http://localhost:8000/v1/",
            served_model="deepseek-ai/DeepSeek-OCR-2",
        )
        client = FakeHttpClient([FakeResponse(200, {"data": [{"id": "other-model"}]})])

        provider = SelfHostedOpenAICompatibleProvider(
            server, builtin_capacity_profile("local-vllm-small"), http_client=client
        )
        result = provider.health_check()

        self.assertFalse(result.ok)
        self.assertIn("deepseek-ai/DeepSeek-OCR-2", result.diagnostic)
        self.assertEqual(len(client.calls), 1)

    def test_local_vllm_small_capacity_uses_conservative_defaults(self) -> None:
        capacity = builtin_capacity_profile("local-vllm-small")

        self.assertEqual(capacity.max_in_flight_requests, 2)
        self.assertEqual(capacity.max_provider_queue, 4)
        self.assertEqual(capacity.queue_size, 4)
        self.assertGreaterEqual(capacity.timeout_seconds, 60.0)
        self.assertLessEqual(capacity.render_target_longest_side, 1280)

    def test_capacity_fingerprint_changes_when_constrained_hardware_tuning_changes(self) -> None:
        base = builtin_capacity_profile("local-vllm-small")
        tuned = ProviderCapacityProfile(
            name=base.name,
            max_in_flight_requests=base.max_in_flight_requests,
            max_provider_queue=base.max_provider_queue + 1,
            queue_size=base.queue_size,
            timeout_seconds=base.timeout_seconds,
            backoff_initial_seconds=base.backoff_initial_seconds,
            backoff_max_seconds=base.backoff_max_seconds,
            circuit_breaker_failure_threshold=base.circuit_breaker_failure_threshold,
            circuit_breaker_cooldown_seconds=base.circuit_breaker_cooldown_seconds,
            render_target_longest_side=base.render_target_longest_side,
            image_format=base.image_format,
        )

        self.assertNotEqual(base.fingerprint(), tuned.fingerprint())


if __name__ == "__main__":
    unittest.main()
