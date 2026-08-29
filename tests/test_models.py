"""Tests for the decoupled OCR-model adapters."""

import unittest

from paperscale.models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    GLMOCRModel,
    InfinityParser2FlashModel,
    LightOnOCRModel,
    LightOnOCRSoupModel,
    MarkdownModel,
    OlmOCRModel,
    OvisOCR2Model,
    QianfanOCRModel,
    Surya2Model,
    UnlimitedOCRModel,
    build_ocr_model,
)
from paperscale.models.glmocr import GLM_OCR_PROMPT
from paperscale.models.markdown import _strip_code_fence
from paperscale.models.ovisocr2 import OVIS_OCR2_PROMPT
from paperscale.prompts import PageResponse


class RegistryTests(unittest.TestCase):
    def test_default_is_markdown(self):
        self.assertEqual(DEFAULT_MODEL, "markdown")

    def test_build_known_models(self):
        self.assertIsInstance(build_ocr_model("markdown"), MarkdownModel)
        self.assertIsInstance(build_ocr_model("olmocr"), OlmOCRModel)
        self.assertIsInstance(build_ocr_model("lightonocr2"), LightOnOCRModel)
        self.assertIsInstance(build_ocr_model("lightonocr2-soup"), LightOnOCRSoupModel)
        self.assertIsInstance(build_ocr_model("glm-ocr"), GLMOCRModel)
        self.assertIsInstance(build_ocr_model("qianfan-ocr"), QianfanOCRModel)
        self.assertIsInstance(build_ocr_model("infinity-parser2-flash"), InfinityParser2FlashModel)
        self.assertIsInstance(build_ocr_model("surya2"), Surya2Model)
        self.assertIsInstance(build_ocr_model("unlimited-ocr"), UnlimitedOCRModel)
        self.assertIsInstance(build_ocr_model("ovisocr2"), OvisOCR2Model)

    def test_registry_contents(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            {
                "markdown",
                "olmocr",
                "lightonocr2",
                "lightonocr2-soup",
                "glm-ocr",
                "qianfan-ocr",
                "infinity-parser2-flash",
                "surya2",
                "unlimited-ocr",
                "ovisocr2",
            },
        )

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
        content = "---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\nis_table: False\nis_diagram: False\n---\nHello world"
        result = self.model.parse(content)
        self.assertEqual(result.natural_text, "Hello world")
        self.assertEqual(result.primary_language, "en")
        self.assertTrue(result.is_rotation_valid)

    def test_parse_rotation_flag(self):
        content = "---\nprimary_language: null\nis_rotation_valid: False\nrotation_correction: 90\nis_table: False\nis_diagram: False\n---\n"
        result = self.model.parse(content)
        self.assertFalse(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 90)
        self.assertIsNone(result.natural_text)


class LightOnOCRModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LightOnOCRModel()

    def test_build_messages_is_image_only(self):
        messages = self.model.build_messages("BASE64DATA")
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        kinds = {part["type"] for part in content}
        # LightOnOCR-2 takes a bare page image: no text instruction.
        self.assertEqual(kinds, {"image_url"})
        image_part = next(p for p in content if p["type"] == "image_url")
        self.assertEqual(image_part["image_url"]["url"], "data:image/png;base64,BASE64DATA")

    def test_no_guided_regex(self):
        self.assertIsNone(self.model.guided_regex())

    def test_parse_passes_through_markdown(self):
        # parse is inherited from MarkdownModel.
        self.assertEqual(self.model.parse("# Title\n\ntext").natural_text, "# Title\n\ntext")

    def test_parse_strips_wrapping_code_fence(self):
        self.assertEqual(self.model.parse("```md\n# T\n```").natural_text, "# T")

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "lightonai/LightOnOCR-2-1B")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        self.assertEqual(self.model.sampling_params(), {"top_p": 0.9})


class LightOnOCRSoupModelTests(unittest.TestCase):
    """The -ocr-soup variant only swaps the served checkpoint."""

    def setUp(self):
        self.model = LightOnOCRSoupModel()

    def test_is_lightonocr_subclass(self):
        self.assertIsInstance(self.model, LightOnOCRModel)

    def test_serves_soup_checkpoint(self):
        self.assertEqual(self.model.default_model_name, "lightonai/LightOnOCR-2-1B-ocr-soup")

    def test_inherits_image_only_recipe(self):
        content = self.model.build_messages("BASE64DATA")[0]["content"]
        self.assertEqual({part["type"] for part in content}, {"image_url"})
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        self.assertEqual(self.model.sampling_params(), {"top_p": 0.9})


class InterfaceDefaultsTests(unittest.TestCase):
    """The interface extensions keep the original adapters unchanged."""

    def test_markdown_defaults(self):
        model = MarkdownModel()
        self.assertEqual(model.sampling_params(), {})
        self.assertEqual(model.preferred_longest_image_dim, 1288)

    def test_olmocr_defaults(self):
        model = OlmOCRModel()
        self.assertEqual(model.sampling_params(), {})
        self.assertEqual(model.preferred_longest_image_dim, 1288)


class GLMOCRModelTests(unittest.TestCase):
    def setUp(self):
        self.model = GLMOCRModel()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "zai-org/GLM-OCR")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        # GLM-OCR uses default decoding; nothing merged into the request and
        # temperature stays pipeline-owned.
        self.assertEqual(self.model.sampling_params(), {})
        self.assertIsNone(self.model.guided_regex())

    def test_build_messages_has_prompt_and_image(self):
        messages = self.model.build_messages("QUJD")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        parts = messages[0]["content"]
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[0]["text"], GLM_OCR_PROMPT)
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_parse_passes_markdown_through(self):
        page = self.model.parse("# Heading\n\nBody text.")
        self.assertEqual(page.natural_text, "# Heading\n\nBody text.")
        self.assertIsNone(page.primary_language)
        self.assertTrue(page.is_rotation_valid)
        self.assertEqual(page.rotation_correction, 0)
        self.assertFalse(page.is_table)
        self.assertFalse(page.is_diagram)

    def test_parse_strips_wrapping_code_fence(self):
        page = self.model.parse("```markdown\n# Heading\n\nBody.\n```")
        self.assertEqual(page.natural_text, "# Heading\n\nBody.")

    def test_parse_empty_page_is_none(self):
        self.assertIsNone(self.model.parse("   ").natural_text)


class QianfanOCRModelTests(unittest.TestCase):
    def setUp(self):
        self.model = QianfanOCRModel()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "baidu/Qianfan-OCR")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        self.assertEqual(self.model.sampling_params(), {"top_p": 1.0, "max_tokens": 12000})
        self.assertIsNone(self.model.guided_regex())

    def test_build_messages_text_and_image(self):
        messages = self.model.build_messages("BASE64DATA")
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Parse this document to Markdown."})
        self.assertEqual(
            content[1],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64DATA"}},
        )

    def test_parse_passes_clean_markdown_through(self):
        result = self.model.parse("# Title\n\nHello world")
        self.assertEqual(result.natural_text, "# Title\n\nHello world")
        self.assertIsInstance(result, PageResponse)
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)
        self.assertFalse(result.is_table)
        self.assertFalse(result.is_diagram)
        self.assertIsNone(result.primary_language)

    def test_parse_strips_leading_think_block(self):
        result = self.model.parse("<think>layout...</think>\n# Title")
        self.assertEqual(result.natural_text, "# Title")

    def test_parse_keeps_content_before_stray_think_close(self):
        # A literal "</think>" in the page (no opening tag) must NOT truncate the
        # transcription — matched-pair stripping only.
        result = self.model.parse("real content </think> tail")
        self.assertEqual(result.natural_text, "real content </think> tail")

    def test_parse_detects_html_table(self):
        result = self.model.parse("Intro\n<table><tr><td>a</td></tr></table>")
        self.assertTrue(result.is_table)

    def test_parse_empty_page_returns_none(self):
        self.assertIsNone(self.model.parse("").natural_text)


class InfinityParser2FlashModelTests(unittest.TestCase):
    def setUp(self):
        self.model = InfinityParser2FlashModel()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "infly/Infinity-Parser2-Flash")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        # top-level chat_template_kwargs disables Qwen3 thinking (not extra_body)
        self.assertEqual(
            self.model.sampling_params(),
            {"top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_build_messages_doc2md_prompt_and_image(self):
        messages = self.model.build_messages("BASE64DATA")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        text_part, image_part = messages[0]["content"]
        self.assertEqual(
            text_part,
            {"type": "text", "text": "Please transform the document's contents into Markdown format."},
        )
        self.assertEqual(
            image_part,
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64DATA"}},
        )

    def test_parse_strips_think_block_and_code_fence(self):
        raw = "<think>plan the layout</think>```markdown\n# Heading\n\nBody text\n```"
        result = self.model.parse(raw)
        self.assertEqual(result.natural_text, "# Heading\n\nBody text")
        self.assertNotIn("<think>", result.natural_text)
        self.assertNotIn("```", result.natural_text)
        self.assertFalse(result.is_table)
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)

    def test_parse_detects_html_table(self):
        result = self.model.parse("<table><tr><td>cell</td></tr></table>")
        self.assertTrue(result.is_table)

    def test_parse_empty_after_think_block_is_none(self):
        result = self.model.parse("<think>only reasoning, no page text</think>")
        self.assertIsNone(result.natural_text)
        self.assertFalse(result.is_table)


# Captured Surya OCR 2 layout-HTML (real label names + the model's inner markup):
# a Section-Header wrapped in <b><u>, a Text block with an inline <math>, an HTML
# <table>, and a List-Group (<ol><li>). data-bbox is the model's "x0 y0 x1 y1" form.
_SAMPLE_LAYOUT_HTML = (
    '<div data-bbox="474 376 601 394" data-label="Section-Header">'
    "<p><b><u>Results and Discussion</u></b></p></div>"
    '<div data-bbox="151 186 889 271" data-label="Text">'
    "<p>The relation <math>E = mc^2</math> held across every trial.</p></div>"
    '<div data-bbox="151 439 919 600" data-label="Table">'
    "<table><tr><th>Name</th><th>Score</th></tr>"
    "<tr><td>Alpha</td><td>0.91</td></tr></table></div>"
    '<div data-bbox="151 610 900 700" data-label="List-Group">'
    "<ol><li>First finding</li></ol></div>"
)


class Surya2ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Surya2Model()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "datalab-to/surya-ocr-2")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        self.assertEqual(self.model.sampling_params(), {})
        self.assertIsNone(self.model.guided_regex())

    def test_build_messages(self):
        messages = self.model.build_messages("QUJD")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("OCR this image to HTML", content[0]["text"])
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_parse_layout_html(self):
        result = self.model.parse(_SAMPLE_LAYOUT_HTML)
        self.assertIsInstance(result, PageResponse)
        text = result.natural_text
        self.assertIsNotNone(text)
        # SectionHeader -> Markdown heading.
        self.assertIn("# Results and Discussion", text)
        # <math> -> paperscale inline LaTeX delimiters.
        self.assertIn("\\( E = mc^2 \\)", text)
        # HTML table -> Markdown table (header cells preserved, pipe delimiters).
        self.assertIn("Name", text)
        self.assertIn("Score", text)
        self.assertIn("|", text)
        # List-Group (<ol><li>) -> Markdown list item.
        self.assertIn("First finding", text)
        # Flags from data-label markers.
        self.assertTrue(result.is_table)
        self.assertFalse(result.is_diagram)
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)

    def test_parse_empty(self):
        result = self.model.parse("")
        self.assertIsNone(result.natural_text)
        self.assertFalse(result.is_table)
        self.assertFalse(result.is_diagram)

    def test_parse_pathological_nesting_does_not_raise(self):
        # A repetition-loop response can nest <div>s thousands deep and overflow
        # markdownify's recursive walk; parse() must degrade, not raise.
        evil = "<div>" * 2000 + "x" + "</div>" * 2000
        result = self.model.parse(evil)  # must not raise RecursionError
        self.assertIsInstance(result, PageResponse)

    def test_parse_flags_tolerate_attribute_quoting(self):
        # is_table/is_diagram must not depend on the model's quoting style.
        single = "<div data-label='Table'><table><tr><td>x</td></tr></table></div>"
        self.assertTrue(self.model.parse(single).is_table)
        spaced = '<div data-label = "Figure">fig</div>'
        self.assertTrue(self.model.parse(spaced).is_diagram)


class UnlimitedOCRModelTests(unittest.TestCase):
    def setUp(self):
        self.model = UnlimitedOCRModel()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "baidu/Unlimited-OCR")
        self.assertEqual(self.model.preferred_longest_image_dim, 1024)
        # skip_special_tokens=false and the ngram anti-loop processor args are
        # required by the vLLM recipe; temperature stays pipeline-owned.
        self.assertEqual(
            self.model.sampling_params(),
            {"skip_special_tokens": False, "vllm_xargs": {"ngram_size": 35, "window_size": 128}},
        )
        self.assertIsNone(self.model.guided_regex())

    def test_build_messages_has_image_prefixed_prompt(self):
        messages = self.model.build_messages("QUJD")
        self.assertEqual(len(messages), 1)
        text_part, image_part = messages[0]["content"]
        # The literal <image> prefix is mandatory; without it output is empty.
        self.assertEqual(text_part, {"type": "text", "text": "<image>document parsing."})
        self.assertEqual(image_part["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_parse_unwraps_ref_and_drops_det(self):
        raw = "<|ref|># Heading<|/ref|><|det|>[[10,20,900,60]]<|/det|>\n\nBody text with <|ref|>a grounded span<|/ref|><|det|>[[1,2,3,4]]<|/det|> inline."
        result = self.model.parse(raw)
        self.assertEqual(
            result.natural_text,
            "# Heading\n\nBody text with a grounded span inline.",
        )
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)

    def test_parse_strips_leftover_special_tokens(self):
        # skip_special_tokens=false can leave structural specials in the text.
        result = self.model.parse("Hello world<｜end▁of▁sentence｜>")
        self.assertEqual(result.natural_text, "Hello world")

    def test_parse_plain_markdown_passes_through(self):
        self.assertEqual(self.model.parse("# Title\n\ntext").natural_text, "# Title\n\ntext")

    def test_parse_empty_page_is_none(self):
        self.assertIsNone(self.model.parse("").natural_text)
        self.assertIsNone(self.model.parse("<|ref|><|/ref|><|det|>[[0,0,0,0]]<|/det|>").natural_text)


class OvisOCR2ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = OvisOCR2Model()

    def test_recipe(self):
        self.assertEqual(self.model.default_model_name, "ATH-MaaS/OvisOCR2")
        self.assertEqual(self.model.preferred_longest_image_dim, 1540)
        # Vendor decodes at 16384 tokens; temperature stays pipeline-owned.
        self.assertEqual(self.model.sampling_params(), {"max_tokens": 16384})
        self.assertNotIn("temperature", self.model.sampling_params())
        self.assertIsNone(self.model.guided_regex())

    def test_prompt_is_vendor_verbatim(self):
        # The leading newline separates the image from the instruction once the
        # chat template renders the turn; the braces are literal, not a format spec.
        self.assertTrue(OVIS_OCR2_PROMPT.startswith("\nExtract all readable content from the image"))
        self.assertIn('<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />', OVIS_OCR2_PROMPT)
        self.assertIn("Format tables as HTML: <table>...</table>.", OVIS_OCR2_PROMPT)
        self.assertTrue(OVIS_OCR2_PROMPT.endswith("without translation or paraphrasing."))

    def test_build_messages_puts_image_before_prompt(self):
        # Order is load-bearing: the chat template trims the ends of the rendered
        # user turn, so a text-first layout would eat the prompt's leading newline.
        messages = self.model.build_messages("QUJD")
        self.assertEqual(len(messages), 1)
        image_part, text_part = messages[0]["content"]
        self.assertEqual(image_part["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(text_part, {"type": "text", "text": OVIS_OCR2_PROMPT})

    def test_parse_passes_markdown_through(self):
        result = self.model.parse("# Title\n\nBody text.")
        self.assertEqual(result.natural_text, "# Title\n\nBody text.")
        self.assertTrue(result.is_rotation_valid)
        self.assertEqual(result.rotation_correction, 0)
        self.assertFalse(result.is_table)
        self.assertFalse(result.is_diagram)

    def test_parse_drops_bbox_image_tags_and_flags_diagram(self):
        raw = '# Report\n\n<img src="images/bbox_100_200_800_600.jpg" />\n\nFigure 1 shows the trend.'
        result = self.model.parse(raw)
        self.assertEqual(result.natural_text, "# Report\n\nFigure 1 shows the trend.")
        self.assertNotIn("<img", result.natural_text)
        self.assertTrue(result.is_diagram)

    def test_parse_drops_inline_bbox_image_tag(self):
        # The vendor filters whole blocks; a dead reference mid-paragraph is just
        # as dangling here, since paperscale never writes the crop files.
        result = self.model.parse('See <img src="images/bbox_1_2_3_4.jpg" /> below.')
        self.assertEqual(result.natural_text, "See  below.")
        self.assertTrue(result.is_diagram)

    def test_parse_detects_html_table(self):
        result = self.model.parse("<table><tr><td>1</td></tr></table>")
        self.assertTrue(result.is_table)
        self.assertFalse(result.is_diagram)

    def test_parse_strips_think_block(self):
        result = self.model.parse("<think>\n\n</think>\n\n# Heading")
        self.assertEqual(result.natural_text, "# Heading")

    def test_parse_keeps_content_before_stray_think_close(self):
        # Matched-pair only: a transcribed "</think>" must not truncate the page.
        result = self.model.parse("Body mentioning </think> literally.")
        self.assertEqual(result.natural_text, "Body mentioning </think> literally.")

    def test_parse_empty_page_is_none(self):
        self.assertIsNone(self.model.parse("").natural_text)
        # A page whose only output was a figure placeholder is empty text, but is
        # still a diagram page.
        figure_only = self.model.parse('<img src="images/bbox_0_0_999_999.jpg" />')
        self.assertIsNone(figure_only.natural_text)
        self.assertTrue(figure_only.is_diagram)


if __name__ == "__main__":
    unittest.main()
