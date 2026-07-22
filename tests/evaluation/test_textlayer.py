"""Tests for text-layer agreement. Deterministic without poppler/real PDFs:
`get_anchor_text` and the on-disk existence check are monkeypatched.
"""

import shutil
import unittest

from paperscale.evaluation import textlayer
from paperscale.evaluation.runs import DocMeta, PageText

LAYER = "The quick brown fox jumps over the lazy dog every single morning today."


def _page(text="hello world this is a page of ocr output", doc="/x/a.pdf", page=1, model="m"):
    return PageText(model=model, doc=doc, page=page, text=text)


def _meta(fallback=0, doc="/x/a.pdf", model="m"):
    return DocMeta(model=model, doc=doc, total_pages=1, fallback_pages=fallback, source_file=doc)


class TextLayerTest(unittest.TestCase):
    def setUp(self):
        self._orig_exists = textlayer._pdf_exists
        self._orig_anchor = textlayer.get_anchor_text
        textlayer._pdf_exists = lambda p: True
        textlayer.get_anchor_text = lambda doc, page, engine, target_length=4000: LAYER

    def tearDown(self):
        textlayer._pdf_exists = self._orig_exists
        textlayer.get_anchor_text = self._orig_anchor

    def test_fallback_doc_skipped(self):
        rows, rep = textlayer.compute_textlayer_agreement([_page()], [_meta(fallback=3)])
        self.assertEqual(rows, [])
        self.assertEqual(rep.docs_with_fallback, 1)

    def test_missing_pdf_skipped(self):
        textlayer._pdf_exists = lambda p: False
        rows, rep = textlayer.compute_textlayer_agreement([_page()], [_meta()])
        self.assertEqual(rows, [])
        self.assertEqual(rep.docs_missing_pdf, 1)

    def test_blank_layer_skipped(self):
        textlayer.get_anchor_text = lambda doc, page, engine, target_length=4000: "12"
        rows, rep = textlayer.compute_textlayer_agreement([_page()], [_meta()])
        self.assertEqual(rows, [])
        self.assertEqual(rep.pages_blank_layer, 1)

    def test_agreement_computed(self):
        rows, rep = textlayer.compute_textlayer_agreement([_page()], [_meta()])
        self.assertEqual(len(rows), 1)
        model, doc, page, f1, ned = rows[0]
        self.assertEqual((model, doc, page), ("m", "/x/a.pdf", 1))
        for v in (f1, ned):
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_identical_text_perfect(self):
        rows, _ = textlayer.compute_textlayer_agreement([_page(text=LAYER)], [_meta()])
        _, _, _, f1, ned = rows[0]
        self.assertEqual(f1, 1.0)
        self.assertEqual(ned, 1.0)

    def test_pdftotext_error_falls_back_to_pypdf(self):
        calls = []

        def fake(doc, page, engine, target_length=4000):
            calls.append(engine)
            if engine == "pdftotext":
                raise FileNotFoundError("no poppler")
            return LAYER

        textlayer.get_anchor_text = fake
        rows, _ = textlayer.compute_textlayer_agreement([_page()], [_meta()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(calls, ["pdftotext", "pypdf"])

    @unittest.skipUnless(shutil.which("pdftotext"), "pdftotext not installed")
    def test_real_pdftotext_smoke(self):
        import tempfile

        # Build a one-page PDF with a real text layer; skip if we can't cheaply.
        try:
            from reportlab.pdfgen import canvas
        except ImportError:
            self.skipTest("reportlab not installed")

        line = "The quick brown fox jumps over the lazy dog every morning."
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            path = fh.name
        c = canvas.Canvas(path)
        c.drawString(72, 720, line)
        c.save()

        textlayer.get_anchor_text = self._orig_anchor  # use the real extractor
        rows, _ = textlayer.compute_textlayer_agreement([_page(text=line, doc=path)], [_meta(doc=path)])
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0][3], 0.5)


if __name__ == "__main__":
    unittest.main()
