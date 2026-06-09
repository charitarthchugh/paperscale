from __future__ import annotations

import unittest

from paperscale.profiles.builtin import builtin_profile_names, get_builtin_profile


class BuiltinProfileTests(unittest.TestCase):
    def test_first_class_profiles_are_document_to_markdown_only(self) -> None:
        expected = {"lighton_ocr_2_1b", "deepseek_ocr_2", "glm_ocr"}

        self.assertTrue(expected.issubset(set(builtin_profile_names())))
        for name in expected:
            profile = get_builtin_profile(name)
            self.assertEqual(profile.task, "document_to_markdown")
            self.assertNotIn("free_ocr", profile.public_modes)
            self.assertNotIn("visual_qa", profile.public_modes)
            self.assertIn("markdown", profile.prompt_template.lower())

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
        baseline = profile.build_request("page-1", b"image", "image/png")

        changed_prompt = profile.with_overrides(prompt_version="v2").build_request(
            "page-1", b"image", "image/png"
        )
        changed_decoding = profile.with_overrides(decoding={"temperature": 0.1}).build_request(
            "page-1", b"image", "image/png"
        )
        changed_render = profile.with_overrides(render_options={"dynamic_resolution": False}).build_request(
            "page-1", b"image", "image/png"
        )
        changed_parser = profile.with_overrides(parser_version="parser-v2").build_request(
            "page-1", b"image", "image/png"
        )

        fingerprints = {
            baseline.fingerprint,
            changed_prompt.fingerprint,
            changed_decoding.fingerprint,
            changed_render.fingerprint,
            changed_parser.fingerprint,
        }
        self.assertEqual(len(fingerprints), 5)

    def test_deepseek_profile_rejects_repetition_and_keeps_crop_settings_in_request(self) -> None:
        profile = get_builtin_profile("deepseek_ocr_2")
        request = profile.build_request("page-1", b"image", "image/png")

        self.assertTrue(request.render_options["dynamic_resolution"])
        self.assertEqual(request.render_options["crop_mode"], "dynamic")
        result = profile.parse_and_validate("same same same same same same")
        self.assertFalse(result.ok)
        self.assertEqual(result.retry_classification, "retryable")

    def test_glm_profile_preserves_markdown_and_optional_layout_metadata(self) -> None:
        profile = get_builtin_profile("glm_ocr")
        result = profile.parse_and_validate(
            '{"markdown":"# Title\\n\\nBody","layout":{"blocks":2}}'
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.markdown, "# Title\n\nBody")
        self.assertEqual(result.metadata["layout"], {"blocks": 2})

    def test_lighton_profile_accepts_clean_markdown_and_has_conservative_rendering(self) -> None:
        profile = get_builtin_profile("lighton_ocr_2_1b")
        request = profile.build_request("page-1", b"image", "image/png")
        result = profile.parse_and_validate("# Heading\n\nA clean paragraph with | table | text |.")

        self.assertTrue(result.ok)
        self.assertEqual(result.markdown, "# Heading\n\nA clean paragraph with | table | text |.")
        self.assertLessEqual(request.render_options["target_longest_side"], 1280)
        self.assertIn("naturally ordered", request.prompt)


if __name__ == "__main__":
    unittest.main()
