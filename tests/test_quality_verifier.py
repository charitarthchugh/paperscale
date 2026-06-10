"""Tests for the deterministic quality verifier that gates page acceptance."""

import unittest

from paperscale.quality.verifier import DeterministicQualityVerifier, assess_markdown_fragment


class AssessMarkdownFragmentTests(unittest.TestCase):
    def test_good_markdown_accepted(self):
        report = assess_markdown_fragment("# Title\n\nA normal paragraph of readable text.")
        self.assertTrue(report.accepted)
        self.assertEqual(report.issues, [])

    def test_empty_rejected(self):
        report = assess_markdown_fragment("   \n\t  ")
        self.assertFalse(report.accepted)
        self.assertEqual(report.issues[0].code, "empty_output")

    def test_mojibake_rejected(self):
        report = assess_markdown_fragment("Some text ��� with replacement chars.")
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "mojibake" for issue in report.issues))

    def test_repeated_ngram_rejected(self):
        report = assess_markdown_fragment(("buy now " * 30).strip())
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code in {"repeated_ngram", "repeated_character"} for issue in report.issues))

    def test_refusal_rejected(self):
        report = assess_markdown_fragment("I'm sorry, but I cannot help with that request.")
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "refusal_boilerplate" for issue in report.issues))


class VerifierClassifyTests(unittest.TestCase):
    def setUp(self):
        self.verifier = DeterministicQualityVerifier()

    def test_accepted_finding(self):
        finding = self.verifier.classify("# Heading\n\nReadable content here.")
        self.assertTrue(finding.accepted)
        self.assertEqual(finding.kind, "ok")
        self.assertEqual(finding.retry_class, "none")

    def test_refusal_is_terminal(self):
        finding = self.verifier.classify("As an AI, I cannot transcribe this document.")
        self.assertFalse(finding.accepted)
        self.assertEqual(finding.kind, "refusal")
        self.assertEqual(finding.retry_class, "terminal")

    def test_empty_is_retryable(self):
        finding = self.verifier.classify("")
        self.assertFalse(finding.accepted)
        self.assertEqual(finding.kind, "empty_output")
        self.assertEqual(finding.retry_class, "retryable")

    def test_mojibake_is_retryable(self):
        finding = self.verifier.classify("text ���� more")
        self.assertFalse(finding.accepted)
        self.assertEqual(finding.retry_class, "retryable")


if __name__ == "__main__":
    unittest.main()
