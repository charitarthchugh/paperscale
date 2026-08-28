"""Tests for derived Resume state: the intersection, the layout guard, and what --no-resume does not do.

Design 11, test obligations 31-34. The properties under test are the ones that are invisible when
they break: an intersection computed wrongly does not raise, it silently skips work that was never
done or repeats a day of it.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from paperscale.embed.resume import (
    LAYOUT_BARE,
    LAYOUT_LABELLED,
    RE_OCR_WARNING,
    LayoutChangeError,
    check_layout,
    derive_resume_state,
    invocation_layout,
    sink_set_warning,
)

LOGGER = "paperscale.embed.resume"

A = ("run-a", "case.pdf")
B = ("run-a", "brief.pdf")
C = ("run-b", "case.pdf")


class FakeSink:
    """A Sink reduced to the two operations Resume's rules are stated over.

    `known()` returns a fresh set every call, the way both real Sinks do (an `os.walk` and a
    `SELECT`), so a test that mutates the returned set cannot accidentally pass.
    """

    def __init__(self, *held: tuple[str, str]) -> None:
        self._held = set(held)
        self.known_calls = 0
        self.writes: list[tuple[str, str]] = []

    def known(self) -> set[tuple[str, str]]:
        self.known_calls += 1
        return set(self._held)

    def write(self, key: tuple[str, str]) -> None:
        self.writes.append(key)
        self._held.add(key)


class NpzSink(FakeSink):
    """Named so the per-Sink log line is asserted against something realistic."""


class LanceSink(FakeSink):
    pass


def embed_pass(corpus: list[tuple[str, str]], sinks: list[FakeSink], *, no_resume: bool = False) -> list[tuple[str, str]]:
    """One Invocation, reduced to the part Resume governs: skip what the state holds, write both Sinks."""
    state = derive_resume_state(list(sinks), no_resume=no_resume)
    embedded = [key for key in corpus if key not in state]
    for key in embedded:
        for sink in sinks:
            sink.write(key)
    return embedded


class IntersectionTest(unittest.TestCase):
    def test_one_sink_needs_no_special_case(self):
        self.assertEqual(derive_resume_state([FakeSink(A, B)], no_resume=False), {A, B})

    def test_a_document_is_done_only_when_every_sink_holds_it(self):
        # Obligation 31. B is in the .npz tree but not yet committed to LanceDB, so it is not done.
        npz = NpzSink(A, B)
        lance = LanceSink(A)
        self.assertEqual(derive_resume_state([npz, lance], no_resume=False), {A})

    def test_the_gap_between_two_sinks_heals_itself(self):
        # The crash case: the .npz pair landed, the LanceDB batch had not flushed. The next
        # Invocation re-embeds B, one write is a no-op and the other completes, and nothing had to
        # detect that anything was wrong.
        npz = NpzSink(A, B)
        lance = LanceSink(A)

        embedded = embed_pass([A, B], [npz, lance])

        self.assertEqual(embedded, [B])
        self.assertEqual(npz.writes, [B], "the .npz Sink rewrites B harmlessly through rename")
        self.assertEqual(lance.writes, [B])
        self.assertEqual(npz.known(), lance.known())

    def test_a_healed_invocation_repeats_nothing_next_time(self):
        npz = NpzSink(A, B)
        lance = LanceSink(A)
        embed_pass([A, B], [npz, lance])

        self.assertEqual(embed_pass([A, B], [npz, lance]), [])

    def test_the_batched_sink_sets_the_pace(self):
        # LanceDB lags the .npz tree by up to one batch of 64, so the intersection is the batched
        # Sink's set whenever the two disagree. That is the price of batching, paid in GPU time.
        npz = NpzSink(*[("run-a", f"doc-{i}.pdf") for i in range(100)])
        lance = LanceSink(*[("run-a", f"doc-{i}.pdf") for i in range(64)])

        self.assertEqual(len(derive_resume_state([npz, lance], no_resume=False)), 64)

    def test_every_sink_is_read_exactly_once(self):
        # Read once at startup, never per Document -- and no short-circuit when the first Sink
        # already knows nothing, or the number of reads would depend on the data.
        npz = NpzSink()
        lance = LanceSink(A, B)

        derive_resume_state([npz, lance], no_resume=False)

        self.assertEqual((npz.known_calls, lance.known_calls), (1, 1))

    def test_the_returned_set_does_not_alias_a_sinks_own_set(self):
        npz = NpzSink(A, B)
        state = derive_resume_state([npz], no_resume=False)
        state.add(C)

        self.assertEqual(npz.known(), {A, B})

    def test_no_sinks_is_the_empty_set_not_everything(self):
        # The CLI rejects this (--no-npz without --lancedb), but the mathematical answer for an
        # intersection over no sets is "everything", which would skip the whole corpus and report a
        # successful no-op.
        self.assertEqual(derive_resume_state([], no_resume=False), set())

    def test_the_skip_count_and_the_per_sink_counts_are_logged(self):
        with self.assertLogs(LOGGER, level="INFO") as caught:
            derive_resume_state([NpzSink(A, B), LanceSink(A)], no_resume=False)
        text = "\n".join(caught.output)
        self.assertIn("NpzSink knows 2", text)
        self.assertIn("LanceSink knows 1", text)
        self.assertIn("1 Document(s)", text)


class EnablingASinkLaterTest(unittest.TestCase):
    def warning_for(self, recorded: list[str], enabled: list[str], **kwargs) -> str:
        """The warning text, having first pinned that there *is* one -- keeps the assertions below bare-assert free."""
        message = sink_set_warning(recorded, enabled, **kwargs)
        self.assertIsNotNone(message, "a changed Sink set must warn")
        return str(message)

    def test_a_new_sink_empties_the_intersection(self):
        # Obligation 32. Correct and expensive: the whole corpus is embedded again.
        npz = NpzSink(A, B, C)
        lance = LanceSink()

        self.assertEqual(derive_resume_state([npz, lance], no_resume=False), set())

    def test_the_invocation_says_so_before_starting(self):
        message = self.warning_for(["npz"], ["npz", "lancedb"], corpus_size=47000)
        self.assertIn("--lancedb", message)
        self.assertIn("47,000 Documents", message)
        self.assertIn("re-embed", message)

    def test_an_unchanged_sink_set_is_silent(self):
        self.assertIsNone(sink_set_warning(["npz"], ["npz"]))

    def test_order_is_not_a_change(self):
        # `sinks` is a set on the wire and a list only because JSON has no set.
        self.assertIsNone(sink_set_warning(["npz", "lancedb"], ["lancedb", "npz"]))

    def test_the_scale_is_named_even_without_a_corpus_size(self):
        message = self.warning_for(["npz"], ["npz", "lancedb"])
        self.assertIn("the whole corpus", message)

    def test_a_dropped_sink_warns_without_threatening_a_re_embed(self):
        message = self.warning_for(["npz", "lancedb"], ["npz"], corpus_size=47000)
        self.assertIn("--lancedb", message)
        self.assertIn("no longer enabled", message)
        self.assertNotIn("re-embed", message)


class LayoutGuardTest(unittest.TestCase):
    def test_one_run_is_bare_and_two_are_labelled(self):
        self.assertEqual(invocation_layout(1), LAYOUT_BARE)
        self.assertEqual(invocation_layout(2), LAYOUT_LABELLED)

    def test_an_unchanged_layout_passes(self):
        self.assertIsNone(check_layout(out=Path("/out"), recorded=LAYOUT_BARE, current=LAYOUT_BARE))

    def test_a_layout_change_stops_the_invocation_reporting_both_values(self):
        # Obligation 33. The failure it prevents is duplication, not mixing: nothing would match,
        # so the corpus would be embedded again into a parallel subtree.
        with self.assertRaises(LayoutChangeError) as caught:
            check_layout(out=Path("/data/vectors"), recorded=LAYOUT_BARE, current=LAYOUT_LABELLED)
        message = str(caught.exception)
        self.assertIn(LAYOUT_BARE, message)
        self.assertIn(LAYOUT_LABELLED, message)
        self.assertIn("/data/vectors", message)

    def test_the_message_names_both_ways_out(self):
        with self.assertRaises(LayoutChangeError) as caught:
            check_layout(out="/data/vectors", recorded=LAYOUT_LABELLED, current=LAYOUT_BARE)
        message = str(caught.exception)
        self.assertIn("--run", message)
        self.assertIn("--out", message)


class NoResumeTest(unittest.TestCase):
    def test_no_resume_returns_the_empty_set(self):
        self.assertEqual(derive_resume_state([NpzSink(A, B)], no_resume=True), set())

    def test_no_resume_re_embeds_everything_and_overwrites(self):
        npz = NpzSink(A, B)
        lance = LanceSink(A, B)

        embedded = embed_pass([A, B], [npz, lance], no_resume=True)

        self.assertEqual(embedded, [A, B])
        self.assertEqual(npz.writes, [A, B])
        self.assertEqual(lance.writes, [A, B])

    def test_no_resume_deletes_nothing(self):
        # Obligation 34, and the deliberate divergence from the OCR pipeline's --no-resume, which
        # rmtree's results/, done_flags/ and worker_locks/. An embed output is the deliverable and
        # may already be open in a Consumer, so nothing here is allowed to remove a file.
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "run-a").mkdir()
            kept = out / "run-a" / "case.pdf.npz"
            kept.write_bytes(b"vectors")
            (out / "paperscale-embed.json").write_text("{}")

            with (
                mock.patch.object(shutil, "rmtree", side_effect=AssertionError("--no-resume must not rmtree an embed output")),
                mock.patch.object(Path, "unlink", side_effect=AssertionError("--no-resume must not remove an output file")),
            ):
                self.assertEqual(derive_resume_state([NpzSink(A, B)], no_resume=True), set())

            self.assertTrue(kept.exists())
            self.assertTrue((out / "paperscale-embed.json").exists())

    def test_no_resume_asks_no_sink_what_it_holds(self):
        # Ignoring prior state is sufficient because both Sinks are idempotent, so the os.walk and
        # the SELECT are not paid for either. A Sink that needs its own set (LanceDB, to choose
        # add() over merge_insert) reads it for itself.
        npz = NpzSink(A, B)

        derive_resume_state([npz], no_resume=True)

        self.assertEqual(npz.known_calls, 0)

    def test_no_resume_says_out_loud_that_it_deletes_nothing(self):
        with self.assertLogs(LOGGER, level="INFO") as caught:
            derive_resume_state([NpzSink(A)], no_resume=True)
        text = "\n".join(caught.output)
        self.assertIn("nothing on disk is deleted", text)


class ReOcrWarningTest(unittest.TestCase):
    """The README reproduces this verbatim (design 19), so its content is pinned here."""

    def test_it_states_the_failure_and_both_ways_out(self):
        self.assertIn("stale vectors", RE_OCR_WARNING)
        self.assertIn("--no-resume", RE_OCR_WARNING)
        self.assertIn("fresh output directory", RE_OCR_WARNING)

    def test_it_admits_the_two_checks_do_not_cover_text(self):
        # Without this paragraph the model and layout checks read as protection they are not.
        self.assertIn("Neither of those notices changed *text*", RE_OCR_WARNING)

    def test_it_is_ascii_so_the_readme_can_carry_it_verbatim(self):
        RE_OCR_WARNING.encode("ascii")
