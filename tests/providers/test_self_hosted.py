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


def _capacity(threshold: int = 3, *, initial: float = 1.0, maximum: float = 8.0):
    from paperscale.providers.self_hosted import ProviderCapacityProfile

    return ProviderCapacityProfile(
        name="test",
        max_in_flight_requests=1,
        max_provider_queue=1,
        queue_size=1,
        timeout_seconds=1.0,
        backoff_initial_seconds=initial,
        backoff_max_seconds=maximum,
        circuit_breaker_failure_threshold=threshold,
        circuit_breaker_cooldown_seconds=1.0,
        render_target_longest_side=10,
    )


class RecordFailureTests(unittest.TestCase):
    def test_retryable_failures_open_circuit_after_threshold(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=3))
        first = controller.record_failure(retryable=True)
        second = controller.record_failure(retryable=True)
        third = controller.record_failure(retryable=True)
        self.assertTrue(first.should_retry)
        self.assertFalse(first.circuit_open)
        self.assertTrue(second.should_retry)
        self.assertTrue(third.circuit_open)
        self.assertFalse(third.should_retry)

    def test_backoff_grows_exponentially_then_caps(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=10, initial=1.0, maximum=8.0))
        backoffs = [controller.record_failure(retryable=True).backoff_seconds for _ in range(5)]
        self.assertEqual(backoffs, [1.0, 2.0, 4.0, 8.0, 8.0])

    def test_non_retryable_failure_resets_consecutive_count(self) -> None:
        from paperscale.providers.self_hosted import ProviderOverloadController

        controller = ProviderOverloadController(_capacity(threshold=2))
        controller.record_failure(retryable=True)
        decision = controller.record_failure(retryable=False)
        self.assertFalse(decision.should_retry)
        self.assertFalse(controller.circuit_open)


if __name__ == "__main__":
    unittest.main()
