"""Tests for the context-length rule, the Chunk budget, and the request floor."""

from __future__ import annotations

import unittest

from paperscale.embed.budget import SAFETY_MARGIN, chunk_budget, request_budget, resolve_context_length

LOGGER = "paperscale.embed.budget"

# Both pinned families state 32768 on their cards. The server numbers are what a default
# serve of each actually advertises: Qwen3 reports its max_position_embeddings, Nemotron
# reports the YaRN-scaled 16384 * 16.
CARD = 32768
QWEN3_SERVER = 40960
NEMOTRON_SERVER = 262144


class ResolveContextLengthTest(unittest.TestCase):
    def test_default_is_the_minimum_of_card_and_server(self):
        self.assertEqual(resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=None), CARD)

    def test_yarn_advertisement_never_raises_the_window(self):
        # 262144 is the checkpoint's real YaRN-scaled base, not a bug -- but the card is what
        # is authoritative about the model, so the advertisement must not pull the window up.
        self.assertEqual(resolve_context_length(card_context_length=CARD, server_max_model_len=NEMOTRON_SERVER, override=None), CARD)

    def test_short_deployment_pulls_the_window_down(self):
        # The protection min() adds over trusting the card alone: --max-model-len 8192 against
        # a 32768 card would otherwise hard-fail every long Document.
        self.assertEqual(resolve_context_length(card_context_length=CARD, server_max_model_len=8192, override=None), 8192)

    def test_override_above_the_server_is_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=QWEN3_SERVER + 1)
        message = str(caught.exception)
        self.assertIn("40961", message)
        self.assertIn(str(QWEN3_SERVER), message)

    def test_override_equal_to_the_server_is_accepted(self):
        # The correctness boundary is "above", not "at" -- the server agreed to serve this one.
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(
                resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=QWEN3_SERVER),
                QWEN3_SERVER,
            )

    def test_override_above_the_card_is_allowed_and_warns_naming_both_numbers(self):
        with self.assertLogs(LOGGER, level="WARNING") as caught:
            value = resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=40000)
        self.assertEqual(value, 40000)
        text = "\n".join(caught.output)
        self.assertIn("40000", text)
        self.assertIn(str(CARD), text)

    def test_the_warning_says_quality_is_unmeasured_not_reduced(self):
        # "may reduce quality" implies a measurement exists above the card. None does; the risk
        # is an absence of evidence, and the operator has to be told that specifically.
        with self.assertLogs(LOGGER, level="WARNING") as caught:
            resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=40000)
        text = "\n".join(caught.output).lower()
        self.assertIn("unmeasured", text)
        self.assertNotIn("may reduce quality", text)

    def test_override_equal_to_the_card_is_silent(self):
        with self.assertNoLogs(LOGGER):
            self.assertEqual(resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=CARD), CARD)

    def test_override_below_both_is_silent(self):
        with self.assertNoLogs(LOGGER):
            self.assertEqual(resolve_context_length(card_context_length=CARD, server_max_model_len=QWEN3_SERVER, override=4096), 4096)

    def test_default_never_warns(self):
        with self.assertNoLogs(LOGGER):
            resolve_context_length(card_context_length=CARD, server_max_model_len=NEMOTRON_SERVER, override=None)


class ChunkBudgetTest(unittest.TestCase):
    def test_margin_is_sixty_four(self):
        self.assertEqual(SAFETY_MARGIN, 64)

    def test_qwen3_on_a_default_serve(self):
        # Empty document-side Instruction, so the whole margin is the only deduction.
        self.assertEqual(chunk_budget(CARD, 0), 32704)

    def test_nemotron_on_a_default_serve(self):
        # "passage: " is about three tokens.
        self.assertEqual(chunk_budget(CARD, 3), 32701)

    def test_the_stale_868_margin_is_not_in_use(self):
        self.assertNotEqual(chunk_budget(CARD, 0), 31900)

    def test_a_short_deployment_shrinks_the_budget_with_it(self):
        self.assertEqual(chunk_budget(8192, 3), 8125)


class RequestBudgetTest(unittest.TestCase):
    def test_the_default_flag_is_raised_to_the_floor(self):
        # The default configuration of both pinned families lands here: 32000 sits ~704 tokens
        # below one full-size Chunk, which is why the floor is enforced by raising.
        with self.assertLogs(LOGGER, level="INFO") as caught:
            self.assertEqual(request_budget(32000, 32704), 32704)
        text = "\n".join(caught.output)
        self.assertIn("32000", text)
        self.assertIn("32704", text)

    def test_a_flag_below_the_floor_is_never_rejected(self):
        with self.assertLogs(LOGGER, level="INFO"):
            self.assertEqual(request_budget(1, 32704), 32704)

    def test_a_flag_on_the_floor_is_left_alone_and_silent(self):
        with self.assertNoLogs(LOGGER):
            self.assertEqual(request_budget(32704, 32704), 32704)

    def test_a_flag_above_the_floor_is_left_alone_and_silent(self):
        with self.assertNoLogs(LOGGER):
            self.assertEqual(request_budget(64000, 32704), 64000)


if __name__ == "__main__":
    unittest.main()
