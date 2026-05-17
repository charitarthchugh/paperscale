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

        self.assertFalse(empty.accepted)
        self.assertEqual(empty.issues[0].code, "empty_output")
        self.assertFalse(repeated.accepted)
        self.assertIn("repeated_ngram", {issue.code for issue in repeated.issues})
        self.assertFalse(mojibake.accepted)
        self.assertIn("mojibake", {issue.code for issue in mojibake.issues})


if __name__ == "__main__":
    unittest.main()
