import unittest

from paperscale.quality.verifier import assess_markdown_fragment


class QualityVerifierTests(unittest.TestCase):
    def test_accepts_clean_markdown_fragment(self) -> None:
        report = assess_markdown_fragment("# Invoice\n\nTotal: **$12.00**\n")

        self.assertTrue(report.accepted)
        self.assertEqual(report.severity, "ok")
        self.assertEqual(report.issues, [])

    def test_rejects_empty_or_incoherent_fragments_without_provider_calls(self) -> None:
        empty = assess_markdown_fragment("   \n\t")
        repeated = assess_markdown_fragment("Total due. " * 80)
        mojibake = assess_markdown_fragment("# Page\n\nAmount: \ufffd\ufffd\ufffd\ufffd\ufffd\ufffd")
        refusal = assess_markdown_fragment("I'm sorry, but I can't help with that request.")
        frontmatter = assess_markdown_fragment("---\ntitle: Example\n")
        truncated = assess_markdown_fragment("# Page\n\nThis sentence was truncated...")
        short = assess_markdown_fragment("ok")

        self.assertFalse(empty.accepted)
        self.assertEqual(empty.issues[0].code, "empty_output")
        self.assertFalse(repeated.accepted)
        self.assertIn("repeated_ngram", {issue.code for issue in repeated.issues})
        self.assertFalse(mojibake.accepted)
        self.assertIn("mojibake", {issue.code for issue in mojibake.issues})
        self.assertFalse(refusal.accepted)
        self.assertIn("refusal_boilerplate", {issue.code for issue in refusal.issues})
        self.assertFalse(frontmatter.accepted)
        self.assertIn("malformed_frontmatter", {issue.code for issue in frontmatter.issues})
        self.assertFalse(truncated.accepted)
        self.assertIn("truncation_indicator", {issue.code for issue in truncated.issues})
        self.assertFalse(short.accepted)
        self.assertIn("length_anomaly", {issue.code for issue in short.issues})


if __name__ == "__main__":
    unittest.main()
