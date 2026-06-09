from __future__ import annotations

import unittest

from paperscale.profiles.builtin import get_builtin_profile


class RemediationLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        # deepseek_ocr_2 has the richest base decoding/render options.
        self.profile = get_builtin_profile("deepseek_ocr_2")
        self.base_tokens = int(self.profile.decoding["max_tokens"])
        self.base_side = int(self.profile.render_options["target_longest_side"])

    def test_unknown_diagnostic_is_identity(self) -> None:
        acc = self.profile.remediation_for("totally_unknown", accumulated=None)
        applied = self.profile.with_overrides(
            decoding=acc.get("decoding"), render_options=acc.get("render_options")
        )
        self.assertEqual(applied.decoding["max_tokens"], self.base_tokens)
        self.assertEqual(applied.render_options["target_longest_side"], self.base_side)

    def test_truncation_increases_token_budget(self) -> None:
        acc = self.profile.remediation_for("truncation_indicator", accumulated=None)
        self.assertGreater(acc["decoding"]["max_tokens"], self.base_tokens)

    def test_length_anomaly_increases_token_budget(self) -> None:
        acc = self.profile.remediation_for("length_anomaly", accumulated=None)
        self.assertGreater(acc["decoding"]["max_tokens"], self.base_tokens)

    def test_mojibake_increases_render_resolution(self) -> None:
        acc = self.profile.remediation_for("mojibake", accumulated=None)
        self.assertGreater(acc["render_options"]["target_longest_side"], self.base_side)

    def test_control_characters_increase_render_resolution(self) -> None:
        acc = self.profile.remediation_for("control_characters", accumulated=None)
        self.assertGreater(acc["render_options"]["target_longest_side"], self.base_side)

    def test_repeated_ngram_raises_repetition_penalty(self) -> None:
        acc = self.profile.remediation_for("repeated_ngram", accumulated=None)
        self.assertGreater(acc["decoding"].get("repetition_penalty", 1.0), 1.0)

    def test_remediations_accumulate_across_dimensions(self) -> None:
        # A mojibake DPI bump must survive a subsequent truncation token bump.
        acc = self.profile.remediation_for("mojibake", accumulated=None)
        acc = self.profile.remediation_for("truncation_indicator", accumulated=acc)
        self.assertGreater(acc["render_options"]["target_longest_side"], self.base_side)
        self.assertGreater(acc["decoding"]["max_tokens"], self.base_tokens)

    def test_repeated_application_monotonic_then_caps_at_ceiling(self) -> None:
        acc = None
        seen: list[int] = []
        for _ in range(12):
            acc = self.profile.remediation_for("truncation_indicator", accumulated=acc)
            seen.append(acc["decoding"]["max_tokens"])
        # monotonically non-decreasing
        self.assertEqual(seen, sorted(seen))
        # never exceeds the profile's declared ceiling
        self.assertLessEqual(max(seen), self.profile.max_decode_tokens)
        # actually reaches the ceiling under sustained pressure
        self.assertEqual(seen[-1], self.profile.max_decode_tokens)


if __name__ == "__main__":
    unittest.main()
