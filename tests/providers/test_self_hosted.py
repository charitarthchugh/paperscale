from __future__ import annotations

import unittest

from tests.harness.fakes import FakeHttpClient
from tests.harness.imports import require_symbol


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


if __name__ == "__main__":
    unittest.main()
