"""Tests for the shared symspell dictionary + correction-counting."""

import unittest

from paperscale.evaluation.spell import build_dictionary, correct_text, correction_counts


class CorrectionCountsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sym = build_dictionary()

    def test_clean_text_needs_no_corrections(self):
        n, corrected, uncorrectable = correction_counts("the quick brown fox", self.sym)
        self.assertEqual((corrected, uncorrectable), (0, 0))
        self.assertEqual(n, 4)

    def test_typos_are_correctable(self):
        # "quikc"/"brwn" are edit-distance<=2 from real words -> correctable.
        n, corrected, uncorrectable = correction_counts("quikc brwn", self.sym)
        self.assertEqual(corrected, 2)
        self.assertEqual(uncorrectable, 0)

    def test_unrecoverable_garbage_is_uncorrectable(self):
        # random consonant soup has no near dictionary word.
        _, corrected, uncorrectable = correction_counts("xkcdqz zzqwvbf", self.sym)
        self.assertEqual(corrected, 0)
        self.assertEqual(uncorrectable, 2)

    def test_none_when_no_alpha_tokens(self):
        self.assertIsNone(correction_counts("123 456 !!! ---", self.sym))

    def test_extra_words_treated_as_known(self):
        sym = build_dictionary(frozenset({"zzqxword"}))
        n, corrected, uncorrectable = correction_counts("zzqxword", sym)
        self.assertEqual((corrected, uncorrectable), (0, 0))


class CorrectTextTest(unittest.TestCase):
    def test_fixes_word_keeps_numbers_and_punctuation(self):
        sym = build_dictionary()
        out = correct_text("wrold cat 42, ok!", sym)
        self.assertIn("world", out)   # wrold -> world
        self.assertIn("42,", out)     # number + punctuation untouched
        self.assertTrue(out.endswith("!"))


if __name__ == "__main__":
    unittest.main()
