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

    def test_first_class_profiles_build_provider_neutral_requests_from_same_page_input(self) -> None:
        for name in ("lighton_ocr_2_1b", "deepseek_ocr_2", "glm_ocr"):
            with self.subTest(profile=name):
                profile = get_builtin_profile(name)
                request = profile.build_request("doc-7:page-3", b"same-image", "image/png")

                self.assertEqual(request.page_id, "doc-7:page-3")
                self.assertEqual(request.provider, "openai-compatible-chat")
                self.assertEqual(request.profile_name, name)
                self.assertEqual(request.image_media_type, "image/png")
                self.assertNotIn("scheduler", request.provider_options)
                self.assertTrue(request.fingerprint)

    def test_builtin_profiles_classify_empty_and_malformed_outputs(self) -> None:
        for name in ("lighton_ocr_2_1b", "deepseek_ocr_2", "glm_ocr"):
            with self.subTest(profile=name):
                result = get_builtin_profile(name).parse_and_validate("   ")
                self.assertFalse(result.ok)
                self.assertEqual(result.retry_classification, "retryable")

    def test_profile_request_fingerprint_changes_with_prompt_decoding_render_and_parser(self) -> None:
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
