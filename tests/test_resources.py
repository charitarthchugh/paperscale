from __future__ import annotations

import unittest

from tests.harness.imports import require_symbol


class ResourceGovernorTests(unittest.TestCase):
    def test_acquisition_order_violation_raises(self) -> None:
        ResourceGovernor = require_symbol("paperscale.resources", "ResourceGovernor")
        ResourceOrderViolation = require_symbol("paperscale.resources", "ResourceOrderViolation")
        governor = ResourceGovernor()
        with self.assertRaises(ResourceOrderViolation):
            with governor.acquire("provider_concurrency"):
                with governor.acquire("file_descriptor"):
                    pass

    def test_managed_page_ocr_releases_in_reverse_global_order(self) -> None:
        ResourceGovernor = require_symbol("paperscale.resources", "ResourceGovernor")
        governor = ResourceGovernor()
        with governor.acquire_many(
            [
                "cancellation_token",
                "scheduler_slot",
                "pdf_render_slot",
                "file_descriptor",
                "provider_concurrency",
                "page_lease",
                "state_store_lock",
            ]
        ):
            self.assertEqual(
                governor.debug_active_order,
                [
                    "cancellation_token",
                    "scheduler_slot",
                    "pdf_render_slot",
                    "file_descriptor",
                    "provider_concurrency",
                    "page_lease",
                    "state_store_lock",
                ],
            )
        self.assertEqual(governor.debug_active_order, [])
        self.assertEqual(governor.debug_release_order[-3:], ["pdf_render_slot", "scheduler_slot", "cancellation_token"])

    def test_file_open_factory_requires_governor_token(self) -> None:
        ResourceGovernor = require_symbol("paperscale.resources", "ResourceGovernor")
        UnauthorizedResourceError = require_symbol("paperscale.resources", "UnauthorizedResourceError")
        governor = ResourceGovernor()
        with self.assertRaises(UnauthorizedResourceError):
            governor.open_file("/tmp/paperscale-forbidden", "rb")


if __name__ == "__main__":
    unittest.main()
