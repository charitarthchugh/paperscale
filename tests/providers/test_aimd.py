from __future__ import annotations

import unittest

from paperscale.providers.self_hosted import ProviderCapacityProfile, ProviderOverloadController


def _capacity(max_in_flight: int) -> ProviderCapacityProfile:
    return ProviderCapacityProfile(
        name="aimd",
        max_in_flight_requests=max_in_flight,
        max_provider_queue=max_in_flight * 2,
        queue_size=max_in_flight * 2,
        timeout_seconds=10.0,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=8.0,
        circuit_breaker_failure_threshold=100,  # keep circuit out of the way
        circuit_breaker_cooldown_seconds=10.0,
        render_target_longest_side=800,
    )


class AimdConcurrencyTests(unittest.TestCase):
    def test_starts_fed_at_ceiling(self) -> None:
        controller = ProviderOverloadController(_capacity(16), floor=4, ceiling=16)
        self.assertEqual(controller.concurrency_limit, 16)

    def test_overload_multiplicatively_decreases_toward_floor(self) -> None:
        controller = ProviderOverloadController(_capacity(16), floor=4, ceiling=16, decrease_factor=0.5)
        controller.record_status(429)
        self.assertEqual(controller.concurrency_limit, 8)
        controller.record_status(503)
        self.assertEqual(controller.concurrency_limit, 4)
        controller.record_status(500)  # already at floor; never drops below
        self.assertEqual(controller.concurrency_limit, 4)

    def test_transport_failure_also_shrinks(self) -> None:
        controller = ProviderOverloadController(_capacity(16), floor=4, ceiling=16, decrease_factor=0.5)
        controller.record_failure(retryable=True)
        self.assertEqual(controller.concurrency_limit, 8)

    def test_sustained_success_additively_increases_toward_ceiling(self) -> None:
        controller = ProviderOverloadController(
            _capacity(16), floor=4, ceiling=16, decrease_factor=0.5, success_threshold=3, additive_step=1
        )
        controller.record_status(429)  # 16 -> 8
        controller.record_status(429)  # 8 -> 4
        self.assertEqual(controller.concurrency_limit, 4)
        # additive increase only after sustained success
        for _ in range(3):
            controller.record_success()
        self.assertEqual(controller.concurrency_limit, 5)
        for _ in range(3):
            controller.record_success()
        self.assertEqual(controller.concurrency_limit, 6)

    def test_never_exceeds_ceiling(self) -> None:
        controller = ProviderOverloadController(
            _capacity(8), floor=2, ceiling=8, success_threshold=1, additive_step=4
        )
        for _ in range(20):
            controller.record_success()
        self.assertEqual(controller.concurrency_limit, 8)

    def test_content_failure_never_throttles(self) -> None:
        controller = ProviderOverloadController(_capacity(16), floor=4, ceiling=16)
        controller.note_content_failure()
        controller.note_content_failure()
        self.assertEqual(controller.concurrency_limit, 16)
        self.assertFalse(controller.circuit_open)


if __name__ == "__main__":
    unittest.main()
