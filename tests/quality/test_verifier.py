from __future__ import annotations

import unittest

from tests.harness.imports import require_symbol


class DeterministicQualityVerifierTests(unittest.TestCase):
    def test_empty_refusal_repeated_ngram_and_malformed_frontmatter_are_classified_before_commit(self) -> None:
        DeterministicQualityVerifier = require_symbol("paperscale.quality.verifier", "DeterministicQualityVerifier")
        verifier = DeterministicQualityVerifier()
        cases = {
            "": "empty_output",
            "I am unable to help with that document.": "refusal",
            "total total total total total total total total total total total": "repeated_ngram",
            "---\ntitle: [unterminated\n---\n# Body": "malformed_frontmatter",
        }
        for text, expected in cases.items():
            with self.subTest(expected=expected):
                finding = verifier.classify(text)
                self.assertFalse(finding.accepted)
                self.assertEqual(finding.kind, expected)
                self.assertIn(finding.retry_class, {"retryable", "terminal"})

    def test_verifier_metadata_is_persisted_with_page_artifact(self) -> None:
        PageArtifact = require_symbol("paperscale.contracts", "PageArtifact")
        VerificationFinding = require_symbol("paperscale.quality.verifier", "VerificationFinding")
        artifact = PageArtifact(
            page_id="doc:1",
            markdown="# Page",
            result_pointer="artifacts/doc/1.md",
            verifier_metadata=[VerificationFinding(accepted=True, kind="ok", retry_class="none", warnings=["short_page"])],
        )
        self.assertEqual(artifact.verifier_metadata[0].warnings, ["short_page"])

    def test_optional_verifier_slm_is_extension_point_not_required_dependency(self) -> None:
        DeterministicQualityVerifier = require_symbol("paperscale.quality.verifier", "DeterministicQualityVerifier")
        verifier = DeterministicQualityVerifier(optional_slm=None)
        self.assertTrue(verifier.classify("# Valid page\n\nText").accepted)


if __name__ == "__main__":
    unittest.main()
