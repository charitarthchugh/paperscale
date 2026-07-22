"""Tests for pure per-page metrics."""

import unittest

from paperscale.evaluation.metrics import (
    bow_f1,
    garbage_token_fraction,
    normalize_markdown,
    one_minus_ned,
)

CLEAN = "The quick brown fox jumps over the lazy dog every morning."
GARBLED = "Teh qckuz brwn fxo jmps ovr teh lzy dg3 xkcdqz mrnng thzzzz."
REPEAT = "error error error error error error error error error error"


class GarbageTest(unittest.TestCase):
    def test_clean_low_garbled_high(self):
        self.assertLess(garbage_token_fraction(CLEAN), garbage_token_fraction(GARBLED))

    def test_repeat_chars_flagged(self):
        self.assertGreater(garbage_token_fraction("thzzzz aaaaaa"), 0.5)

    def test_none_on_empty(self):
        self.assertIsNone(garbage_token_fraction("   "))

    def test_vowelless_run_flagged_but_not_acronyms(self):
        from paperscale.evaluation.metrics import _is_garbage

        self.assertTrue(_is_garbage("brwn"))       # lowercase vowel-less OCR garble
        self.assertTrue(_is_garbage("brwnjmps"))
        self.assertFalse(_is_garbage("PDF"))       # all-caps acronym preserved
        self.assertFalse(_is_garbage("XML"))
        self.assertFalse(_is_garbage("cat"))       # has a vowel


class AgreementTest(unittest.TestCase):
    def test_identical_perfect(self):
        self.assertEqual(bow_f1(CLEAN, CLEAN), 1.0)
        self.assertEqual(one_minus_ned(CLEAN, CLEAN), 1.0)

    def test_symmetric(self):
        self.assertAlmostEqual(bow_f1(CLEAN, GARBLED), bow_f1(GARBLED, CLEAN))
        self.assertAlmostEqual(one_minus_ned(CLEAN, GARBLED), one_minus_ned(GARBLED, CLEAN))

    def test_reorder_high_bow_low_ned(self):
        a = "alpha beta gamma delta epsilon"
        b = "epsilon delta gamma beta alpha"
        self.assertEqual(bow_f1(a, b), 1.0)  # same bag of words
        self.assertLess(one_minus_ned(a, b), 0.6)  # but very different sequence

    def test_disjoint_zero_bow(self):
        self.assertEqual(bow_f1("one two three", "four five six"), 0.0)

    def test_normalize_strips_markdown(self):
        self.assertEqual(normalize_markdown("# **Hello** [world](http://x)  `code`"), "Hello world code")


if __name__ == "__main__":
    unittest.main()
