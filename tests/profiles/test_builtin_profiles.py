from __future__ import annotations

import unittest

from tests.harness.imports import require_symbol


class BuiltinModelOcrProfileTests(unittest.TestCase):
    def test_builtin_profiles_are_document_to_markdown_only(self) -> None:
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        for name in ["lighton_ocr_2_1b", "deepseek_ocr_2", "glm_ocr", "generic_vlm_markdown", "strict_json_ocr"]:
            profile = get_builtin_profile(name)
            self.assertEqual(profile.public_task, "document_to_markdown")
            self.assertNotIn("free_ocr", profile.supported_public_modes)
            self.assertNotIn("visual_qa", profile.supported_public_modes)
            self.assertNotIn("arbitrary_prompt", profile.supported_public_modes)

    def test_each_first_class_profile_builds_provider_neutral_request_from_same_page_task(self) -> None:
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        for name in ["lighton_ocr_2_1b", "deepseek_ocr_2", "glm_ocr"]:
            request = get_builtin_profile(name).build_request(task, image_bytes=b"fake")
            self.assertEqual(request.page_id, "doc:1")
            self.assertIn("markdown", request.prompt.lower())
            self.assertFalse(hasattr(request, "openai_messages"), "scheduler-facing request must stay provider-neutral")

    def test_profile_specific_fingerprints_change_for_prompt_parser_decoding_and_render(self) -> None:
        PageTask = require_symbol("paperscale.contracts", "PageTask")
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        task = PageTask(document_id="doc", page_number=1, image_hash="sha256:image")
        profile = get_builtin_profile("deepseek_ocr_2")
        original = profile.request_fingerprint(task)
        self.assertNotEqual(original, profile.with_prompt_version("v2").request_fingerprint(task))
        self.assertNotEqual(original, profile.with_parser_schema_version(2).request_fingerprint(task))
        self.assertNotEqual(original, profile.with_decoding_option("temperature", 0.2).request_fingerprint(task))
        self.assertNotEqual(original, profile.with_render_option("crop_mode", "none").request_fingerprint(task))

    def test_deepseek_profile_tracks_dynamic_resolution_and_repetition_quality_policy(self) -> None:
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        profile = get_builtin_profile("deepseek_ocr_2")
        self.assertEqual(profile.render_options["crop_mode"], "dynamic")
        result = profile.parse_and_validate("word " * 200)
        self.assertFalse(result.accepted)
        self.assertEqual(result.failure_kind, "repetition")

    def test_glm_profile_preserves_markdown_and_optional_layout_metadata(self) -> None:
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        result = get_builtin_profile("glm_ocr").parse_and_validate(
            "---\nlayout: {blocks: 2}\n---\n# Heading\n\nBody"
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.markdown, "# Heading\n\nBody")
        self.assertEqual(result.metadata["layout"]["blocks"], 2)

    def test_lighton_profile_covers_clean_markdown_and_render_preprocessing_settings(self) -> None:
        get_builtin_profile = require_symbol("paperscale.profiles.builtin", "get_builtin_profile")
        profile = get_builtin_profile("lighton_ocr_2_1b")
        self.assertIn("natural reading order", profile.prompt_template.lower())
        self.assertLessEqual(profile.render_options["target_longest_side"], 1536)
        self.assertTrue(profile.parse_and_validate("# Title\n\nClean text").accepted)


if __name__ == "__main__":
    unittest.main()
