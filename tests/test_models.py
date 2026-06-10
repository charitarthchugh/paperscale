"""Tests for the decoupled OCR-model adapters."""

import unittest

from paperscale.models import DEFAULT_MODEL, MODEL_REGISTRY, MarkdownModel, OlmOCRModel, build_ocr_model
from paperscale.models.markdown import _strip_code_fence
from paperscale.prompts import PageResponse


class RegistryTests(unittest.TestCase):
    def test_default_is_markdown(self):
        self.assertEqual(DEFAULT_MODEL, "markdown")

    def test_build_known_models(self):
        self.assertIsInstance(build_ocr_model("markdown"), MarkdownModel)
        self.assertIsInstance(build_ocr_model("olmocr"), OlmOCRModel)

    def test_registry_contents(self):
        self.assertEqual(set(MODEL_REGISTRY), {"markdown", "olmocr"})

    def test_unknown_model_raises_with_choices(self):
        with self.assertRaises(ValueError) as ctx:
            build_ocr_model("nope")
        self.assertIn("markdown", str(ctx.exception))
        self.assertIn("olmocr", str(ctx.exception))


class MarkdownModelTests(unittest.TestCase):
    def setUp(self):
        self.model = MarkdownModel()

    def test_build_messages_embeds_image(self):
        messages = self.model.build_messages("BASE64DATA")
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        kinds = {part["type"] for part in content}
        self.assertEqual(kinds, {"text", "image_url"})
        image_part = next(p for p in content if p["type"] == "image_url")
        self.assertEqual(image_part["image_url"]["url"], "data:image/png;base64,BASE64DATA")

    def test_no_guided_regex(self):
        self.assertIsNone(self.model.guided_regex())

    def test_parse_returns_markdown_as_natural_text(self):
        result = self.model.parse("# Heading\n\nbody")
        self.assertIsInstance(result, PageResponse)
        self.assertEqual(result.natural_text, "# Heading\n\nbody")
        # Markdown models always report a valid, unrotated page.
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)

    def test_parse_empty_is_none(self):
        self.assertIsNone(self.model.parse("   \n  ").natural_text)

    def test_parse_strips_wrapping_code_fence(self):
        wrapped = "```markdown\n# Title\n\ntext\n```"
        self.assertEqual(self.model.parse(wrapped).natural_text, "# Title\n\ntext")

    def test_custom_prompt(self):
        custom = MarkdownModel(prompt="just the text please")
        text_part = custom.build_messages("x")[0]["content"][0]["text"]
        self.assertEqual(text_part, "just the text please")


class StripCodeFenceTests(unittest.TestCase):
    def test_plain_text_untouched(self):
        self.assertEqual(_strip_code_fence("# hi"), "# hi")

    def test_bare_fence(self):
        self.assertEqual(_strip_code_fence("```\nhi\n```"), "hi")

    def test_md_fence(self):
        self.assertEqual(_strip_code_fence("```md\nhi\n```"), "hi")

    def test_inner_fences_preserved(self):
        # An inner code block must not be stripped.
        text = "para\n\n```python\nprint(1)\n```\n\nmore"
        self.assertEqual(_strip_code_fence(text), text)


class OlmOCRModelTests(unittest.TestCase):
    def setUp(self):
        self.model = OlmOCRModel()

    def test_has_guided_regex(self):
        self.assertIsNotNone(self.model.guided_regex())

    def test_parse_front_matter(self):
        content = (
            "---\n"
            "primary_language: en\n"
            "is_rotation_valid: True\n"
            "rotation_correction: 0\n"
            "is_table: False\n"
            "is_diagram: False\n"
            "---\n"
            "Hello world"
        )
        result = self.model.parse(content)
        self.assertEqual(result.natural_text, "Hello world")
        self.assertEqual(result.primary_language, "en")
        self.assertTrue(result.is_rotation_valid)

    def test_parse_rotation_flag(self):
        content = (
            "---\n"
            "primary_language: null\n"
            "is_rotation_valid: False\n"
            "rotation_correction: 90\n"
            "is_table: False\n"
            "is_diagram: False\n"
            "---\n"
        )
        result = self.model.parse(content)
        self.assertFalse(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 90)
        self.assertIsNone(result.natural_text)


if __name__ == "__main__":
    unittest.main()
