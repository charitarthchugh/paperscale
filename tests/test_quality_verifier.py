"""Tests for the deterministic quality verifier that gates page acceptance."""

import inspect
import re
import time
import unittest

from paperscale.quality.verifier import (
    QUALITY_CHECK_CODES,
    DeterministicQualityVerifier,
    _has_repeated_tail_loop,
    assess_markdown_fragment,
    expand_disabled_checks,
)


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


GOOD_SENTENCE = "The Court finds that the respondent failed to establish a prima facie case under the applicable provisions of the statute. "
# A sentence-length loop unit: 13 tokens, longer than the n-gram check can score.
LOOP_SENTENCE = "Payment shall be made within thirty (30) days of invoice. "


class RepeatedTailLoopTests(unittest.TestCase):
    """The tail-loop check covers what the n-gram check structurally cannot.

    ``_has_repeated_ngram_loop`` scores ``count * ngram_size / len(tokens)`` with
    ngram_size capped at 5, so it tops out at ``5 / period``: a looped sentence
    never reaches the 0.35 threshold however much of the page it consumes.
    """

    def test_sentence_length_loop_rejected(self):
        # 86% of this page is one repeated sentence; the n-gram check scores it
        # at 0.35 and lets it through, so the tail check has to catch it.
        text = (GOOD_SENTENCE * 20) + "\n\n" + (LOOP_SENTENCE * 260)
        report = assess_markdown_fragment(text)
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "repeated_tail" for issue in report.issues))

    def test_loop_followed_by_a_trailer_is_still_rejected(self):
        # The loop rarely runs to the final character. OvisOCR2 emits HTML tables,
        # so a model looping table rows typically closes the tag afterwards; an
        # end-anchored check would accept every one of these pages.
        looping = (GOOD_SENTENCE * 20) + "\n\n" + (LOOP_SENTENCE * 260)
        for trailer in (
            "</table>",
            "END OF DOCUMENT.",
            " The auditor then reviewed the remaining ledger entries in detail.",
            " The auditor reviewed further entries." * 10,
        ):
            with self.subTest(trailer=trailer[:24]):
                report = assess_markdown_fragment(looping + trailer)
                self.assertTrue(any(issue.code == "repeated_tail" for issue in report.issues))

    def test_long_period_loop_rejected(self):
        # A 22-token unit: the n-gram check cannot score above 5/22 = 0.23.
        unit = "Payment of the sum shall be made to the clerk within thirty (30) calendar days of the invoice date. "
        report = assess_markdown_fragment((GOOD_SENTENCE * 12) + "\n\n" + (unit * 120))
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "repeated_tail" for issue in report.issues))

    def test_clean_page_accepted(self):
        # Varied prose, not one sentence repeated: a page that really is the same
        # sentence 80 times is degenerate, and the gate is right to reject it.
        page = "# Memorandum of Decision\n\n" + "".join(
            f"Paragraph {n} records that the witness described the events of the {n}th of March and produced exhibit {n} in support. " for n in range(1, 60)
        )
        report = assess_markdown_fragment(page)
        self.assertTrue(report.accepted, [issue.code for issue in report.issues])

    def test_repetitive_form_rows_are_not_a_loop(self):
        # Legal forms legitimately end in identical blank rows. They stay well
        # under the 0.35 share, so the page must survive even though its final
        # characters do repeat at a fixed period.
        preamble = (
            "# Schedule B — Statement of Financial Affairs\n\n"
            "The debtor certifies that the following schedule is complete and accurate "
            "as of the petition date, and incorporates by reference the exhibits attached "
            "hereto in accordance with the applicable local rules of this district. "
        ) * 30
        row = "<tr><td>&nbsp;</td><td>N/A</td><td>0.00</td><td>None</td></tr>\n"
        form = preamble + "\n\n<table>\n" + (row * 40)
        self.assertTrue(assess_markdown_fragment(form).accepted)
        # Also with the closing tag, which is what the trailer ladder probes for.
        report = assess_markdown_fragment(form + "</table>")
        self.assertTrue(report.accepted, [issue.code for issue in report.issues])

    def test_sub_threshold_repeating_tail_is_kept(self):
        # A tail that repeats but stays under the 0.35 share is not a loop. This
        # is also the shape that decides the check's cost: it never short-circuits,
        # so it must stay a slice comparison rather than a per-period character walk.
        text = (GOOD_SENTENCE * 130) + ("x" * 6_000)
        self.assertLess(6_000 / len(text), 0.35)
        report = assess_markdown_fragment(text)
        self.assertFalse(any(issue.code == "repeated_tail" for issue in report.issues))

    def test_tail_check_is_cheap_on_a_large_page(self):
        # Guards against reintroducing the O(max_period * n) walk, which cost
        # ~300ms per page on this input.
        text = (GOOD_SENTENCE * 130) + ("x" * 6_000)
        start = time.perf_counter()
        for _ in range(10):
            _has_repeated_tail_loop(text)
        self.assertLess((time.perf_counter() - start) / 10, 0.05)

    def test_short_text_does_not_crash(self):
        for text in ("", " ", "a", "ab", "# Title"):
            assess_markdown_fragment(text)


class DisabledChecksTests(unittest.TestCase):
    def test_disabling_tail_check_accepts_the_loop(self):
        text = (GOOD_SENTENCE * 20) + "\n\n" + (LOOP_SENTENCE * 260)
        self.assertFalse(assess_markdown_fragment(text).accepted)
        report = assess_markdown_fragment(text, disabled_checks=frozenset({"repeated_tail"}))
        self.assertTrue(report.accepted, [issue.code for issue in report.issues])

    def test_disabling_one_check_leaves_the_others_armed(self):
        report = assess_markdown_fragment(
            "I'm sorry, but I cannot help with that request.",
            disabled_checks=frozenset({"repeated_tail"}),
        )
        self.assertFalse(report.accepted)
        self.assertTrue(any(issue.code == "refusal_boilerplate" for issue in report.issues))

    def test_disabling_every_check_accepts_anything(self):
        every = frozenset(QUALITY_CHECK_CODES)
        for text in ("", "buy now " * 40, "I cannot provide that.", "��� bad"):
            self.assertTrue(assess_markdown_fragment(text, disabled_checks=every).accepted)

    def test_verifier_threads_disabled_checks(self):
        text = (GOOD_SENTENCE * 20) + "\n\n" + (LOOP_SENTENCE * 260)
        self.assertFalse(DeterministicQualityVerifier().classify(text).accepted)
        relaxed = DeterministicQualityVerifier(disabled_checks=frozenset({"repeated_tail"}))
        self.assertTrue(relaxed.classify(text).accepted)

    def test_all_codes_are_real_checks(self):
        # Guards the --disable-quality-check vocabulary against drift in both
        # directions: every advertised code must be honoured by a real `enabled(...)`
        # guard, and every code the gate raises must be advertised.
        self.assertEqual(len(set(QUALITY_CHECK_CODES)), len(QUALITY_CHECK_CODES))
        source = inspect.getsource(assess_markdown_fragment)
        for code in QUALITY_CHECK_CODES:
            with self.subTest(code=code):
                self.assertIn(f'enabled("{code}")', source, f"{code} is advertised but nothing checks it")
        for raised in re.findall(r'QualityIssue\("(\w+)"', source):
            with self.subTest(raised=raised):
                self.assertIn(raised, QUALITY_CHECK_CODES, f"{raised} is raised but not disableable")

    def test_every_code_can_actually_be_disabled(self):
        # Each code must be accepted by expand_disabled_checks and land in the set
        # the verifier consults; "all" must expand to the whole vocabulary.
        for code in QUALITY_CHECK_CODES:
            with self.subTest(code=code):
                self.assertEqual(expand_disabled_checks([code]), frozenset({code}))
        self.assertEqual(expand_disabled_checks(["all"]), frozenset(QUALITY_CHECK_CODES))
        self.assertEqual(expand_disabled_checks([]), frozenset())


if __name__ == "__main__":
    unittest.main()
