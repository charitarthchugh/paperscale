"""Tests for pure per-page metrics."""

import unittest

from paperscale.evaluation.metrics import (
    missing_peer_pairs,
    bow_f1,
    garbage_token_fraction,
    normalize_markdown,
    one_minus_ned,
    peer_rows_for_page,
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


class PeerRowsTest(unittest.TestCase):
    def test_emits_both_directions_with_equal_scores(self):
        rows = peer_rows_for_page((
            ("d1", 3),
            {"a": CLEAN, "b": GARBLED, "c": CLEAN},
            [("a", "b"), ("a", "c"), ("b", "c")],
        ))
        # 3 unordered pairs -> 6 directed rows.
        self.assertEqual(len(rows), 6)
        by_pair = {(r[0], r[1]): r for r in rows}
        for m, peer in [("a", "b"), ("a", "c"), ("b", "c")]:
            fwd, rev = by_pair[(m, peer)], by_pair[(peer, m)]
            self.assertEqual(fwd[2:], rev[2:])  # doc, page, f1, ned identical
            self.assertEqual(fwd[2:4], ("d1", 3))
        self.assertEqual(by_pair[("a", "c")][4], 1.0)  # identical texts -> perfect f1

    def test_scores_only_the_requested_pairs(self):
        # Resume: a-b was already on disk, so only a-c and b-c are asked for.
        rows = peer_rows_for_page((
            ("d1", 3),
            {"a": CLEAN, "b": GARBLED, "c": CLEAN},
            [("a", "c"), ("b", "c")],
        ))
        self.assertEqual(len(rows), 4)  # 2 pairs, both directions
        self.assertEqual(
            {(r[0], r[1]) for r in rows},
            {("a", "c"), ("c", "a"), ("b", "c"), ("c", "b")},
        )

    def test_no_requested_pairs_yields_no_rows(self):
        rows = peer_rows_for_page((("d1", 3), {"a": CLEAN, "b": GARBLED}, []))
        self.assertEqual(rows, [])


class MissingPeerPairsTest(unittest.TestCase):
    """Peer agreement resumes per model pair, so adding a run scores only new pairs."""

    def test_all_pairs_when_nothing_is_stored(self):
        self.assertEqual(
            missing_peer_pairs(["a", "b", "c"], set()),
            [("a", "b"), ("a", "c"), ("b", "c")],
        )

    def test_no_pairs_when_everything_is_stored(self):
        self.assertEqual(missing_peer_pairs(["a", "b"], {("a", "b"), ("b", "a")}), [])

    def test_adding_a_third_model_scores_only_its_new_pairs(self):
        # a-b was scored on an earlier run and must NOT be recomputed.
        self.assertEqual(
            missing_peer_pairs(["a", "b", "c"], {("a", "b"), ("b", "a")}),
            [("a", "c"), ("b", "c")],
        )

    def test_a_single_model_has_no_pairs(self):
        self.assertEqual(missing_peer_pairs(["a"], set()), [])

    def test_a_half_written_pair_is_recomputed(self):
        # Only one direction on disk (interrupted write, or a pre-resume DB). Trusting
        # it would leave b with no row against a, skewing b's leaderboard mean.
        self.assertEqual(missing_peer_pairs(["a", "b"], {("a", "b")}), [("a", "b")])

    def test_input_order_does_not_matter(self):
        self.assertEqual(
            missing_peer_pairs(["c", "a", "b"], {("b", "a"), ("a", "b")}),
            [("a", "c"), ("b", "c")],
        )


if __name__ == "__main__":
    unittest.main()
