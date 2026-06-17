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

    def test_form_underscore_blanks_accepted(self):
        # Fill-in-the-blank lines on a court form are real content, not a loop.
        text = (
            "# Divorce Complaint (Dissolution of Marriage)\n\n"
            "Name change to: " + "_" * 40 + "\n\n"
            "Regarding parental decision-making responsibility, the court enters "
            "the relief requested after reviewing the evidence presented at hearing."
        )
        report = assess_markdown_fragment(text)
        self.assertTrue(report.accepted, report.issues)

    def test_genuine_repeated_letter_run_still_rejected(self):
        # A long run of an ordinary character is still degenerate.
        report = assess_markdown_fragment("Some readable lead-in text " + "a" * 40 + " and a tail.")
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "repeated_character" for issue in report.issues))

    def test_em_dash_line_loop_still_rejected(self):
        # Excluding "—" from the char-run gate must not hide a real em-dash loop;
        # the space-separated loop is caught by the n-gram gate instead.
        text = "Fairfield " + "— " * 60 + "Connecticut 06824"
        report = assess_markdown_fragment(text)
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "repeated_ngram" for issue in report.issues))

    def test_polite_phrase_in_long_transcript_accepted(self):
        # "I'm sorry" quoted deep in a deposition page is speech, not a refusal.
        text = (
            "Q. Please state your name for the record.\n"
            "A. Mark D. Lieberman.\n"
            "Q. And what is your current occupation, sir?\n"
            "A. I install center consoles in the marine industry.\n"
            "Q. Do you recall the delivery of the vessel in question?\n"
            "A. I don't remember the exact date, I'm sorry.\n"
            "Q. That is quite all right. Please review Exhibit 6, dated July.\n"
            "A. Yes, this reflects the work order as I described it earlier.\n"
            "Q. Were the repairs you described a normal part of the process?\n"
            "A. Yes, that was a normal part of the industry at the time.\n"
            "Q. Did you keep any photographs of the completed installation?\n"
            "A. I may have, but I cannot locate them in my records today.\n"
            "Q. Were you present when the customer signed the delivery form?\n"
            "A. I was, and I witnessed the signature on the second page.\n"
            "Q. Thank you for your patience answering these questions.\n"
            "A. Of course, happy to help clarify whatever I am able to.\n"
        )
        self.assertGreater(len(text), 600)
        report = assess_markdown_fragment(text)
        self.assertTrue(report.accepted, report.issues)

    def test_short_polite_refusal_still_rejected(self):
        report = assess_markdown_fragment("I'm sorry, but I can't transcribe this image.")
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "refusal_boilerplate" for issue in report.issues))

    def test_third_person_cannot_provide_accepted(self):
        # "cannot provide" is ordinary legal prose, not an assistant refusal.
        text = (
            "The Supreme Court repeated this holding, but added that the manner by which "
            "it refused to do so cannot provide the basis for claims of bad faith or a "
            "violation of the covenant of good faith and fair dealing under the contract."
        )
        report = assess_markdown_fragment(text)
        self.assertTrue(report.accepted, report.issues)

    def test_first_person_cannot_provide_still_rejected(self):
        report = assess_markdown_fragment("I cannot provide a transcription of this page.")
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
