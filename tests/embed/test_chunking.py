"""Tests for the greedy page packer.

Every test drives `chunk_document` with a fake tokenizer, which is what makes the call
*count* observable -- the design's central affordability claim is that the common case costs
exactly one `/v1/tokenize` call per Document, and nothing else can prove it.

Documents are built by `_document`, which reproduces `build_dolma_document`'s span layout
byte for byte: the `\\n` joiner folded into the preceding page, and a zero-width span for a
page whose `natural_text` was None. Hand-written spans would test a layout the pipeline does
not produce.
"""

from __future__ import annotations

import math
import unittest

from paperscale.embed.chunking import Chunk, chunk_document

_CHARS_PER_TOKEN = 4


class _FakeTokenizer:
    """Stands in for `EmbedClient.tokenize`, recording every string it is asked about.

    `ceil(len(s) / 4)` is subadditive the same way a BPE count is -- splitting a string can
    only raise the total -- so a packer that respects the budget under this counter respects
    it under a real one. The empty string answers 1, not 0, to imitate the BOS token
    /v1/tokenize prepends by default: if the packer ever asked the server about a zero-width
    page, that 1 would show up as a token an empty page had cost.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, text: str) -> int:
        self.calls.append(text)
        return 1 if not text else math.ceil(len(text) / _CHARS_PER_TOKEN)


def _count(text: str) -> int:
    """`_FakeTokenizer`'s answer, without recording a call."""
    return 1 if not text else math.ceil(len(text) / _CHARS_PER_TOKEN)


def _document(pages: list[str | None]) -> tuple[str, list[list[int]]]:
    """Build (text, spans) exactly as `build_dolma_document` does, joiner rules included."""
    text = ""
    spans: list[list[int]] = []
    for index, page in enumerate(pages):
        content = "" if page is None else page + ("\n" if index < len(pages) - 1 else "")
        start = len(text)
        text += content
        spans.append([start, len(text), index + 1])
    return text, spans


class TilingMixin:
    """The property every other assertion rests on: the Chunks partition the Document."""

    def assert_tiles(self, chunks: list[Chunk], text: str) -> None:
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text))
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier.end_char, later.start_char)
        self.assertEqual("".join(text[c.start_char : c.end_char] for c in chunks), text)


class CommonCaseTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    async def test_document_under_budget_is_one_chunk_costing_one_call(self):
        text, spans = _document(["alpha page one", "beta page two", "gamma page three"])
        tok = _FakeTokenizer()

        chunks = await chunk_document(text, spans, 100, tok)

        self.assertEqual(tok.calls, [text])
        self.assertEqual(len(chunks), 1)
        self.assert_tiles(chunks, text)
        self.assertEqual(chunks[0].first_page, 1)
        self.assertEqual(chunks[0].last_page, 3)
        self.assertEqual(chunks[0].token_count, _count(text))
        self.assertFalse(chunks[0].is_partial_page)

    async def test_document_exactly_on_the_budget_still_fits_whole(self):
        text, spans = _document(["x" * 39])
        tok = _FakeTokenizer()

        chunks = await chunk_document(text, spans, _count(text), tok)

        self.assertEqual(len(tok.calls), 1)
        self.assertEqual(len(chunks), 1)

    async def test_empty_text_is_zero_chunks_and_costs_nothing(self):
        tok = _FakeTokenizer()

        self.assertEqual(await chunk_document("", [], 100, tok), [])
        self.assertEqual(tok.calls, [])

    async def test_whitespace_only_text_is_zero_chunks(self):
        text, spans = _document(["   ", "  "])
        tok = _FakeTokenizer()

        self.assertEqual(await chunk_document(text, spans, 100, tok), [])
        self.assertEqual(tok.calls, [])


class PagePackingTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    def setUp(self):
        # Six 20-char pages: 6 tokens each with the joiner, 5 for the last. Three pages fill
        # a 20-token budget with 2 to spare and the fourth would overflow it.
        self.text, self.spans = _document(["abcde fghij klmno pq"[:20] for _ in range(6)])
        self.budget = 20

    async def test_chunks_tile_the_document_and_carry_whole_pages(self):
        chunks = await chunk_document(self.text, self.spans, self.budget, _FakeTokenizer())

        self.assert_tiles(chunks, self.text)
        self.assertEqual([(c.first_page, c.last_page) for c in chunks], [(1, 3), (4, 6)])
        self.assertFalse(any(c.is_partial_page for c in chunks))

    async def test_no_assembled_chunk_exceeds_the_budget_when_tokenized_whole(self):
        chunks = await chunk_document(self.text, self.spans, self.budget, _FakeTokenizer())

        for chunk in chunks:
            whole = _count(self.text[chunk.start_char : chunk.end_char])
            self.assertLessEqual(whole, self.budget)
            # The recorded count is the sum of the per-page counts, so it is an upper bound
            # on the whole-Chunk count -- that gap is exactly what buys the packer the right
            # to skip a re-check per Chunk.
            self.assertGreaterEqual(chunk.token_count, whole)

    async def test_overflow_costs_one_call_per_page_and_never_re_tokenizes_a_chunk(self):
        tok = _FakeTokenizer()

        chunks = await chunk_document(self.text, self.spans, self.budget, tok)

        self.assertEqual(tok.calls, [self.text] + [self.text[s:e] for s, e, _ in self.spans])
        for chunk in chunks:
            self.assertNotIn(self.text[chunk.start_char : chunk.end_char], tok.calls[1:])


class ZeroWidthPageTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    async def test_empty_page_never_forces_a_break_and_is_never_sent(self):
        # Pages 1 and 3 cost 6 tokens each and page 4 costs 5, so a 12-token budget holds
        # pages 1-3 and breaks before page 4. If the empty page 2 cost even one token the
        # break would move a page earlier, which is the failure this guards.
        text, spans = _document(["a" * 20, None, "b" * 20, "c" * 20])
        tok = _FakeTokenizer()

        chunks = await chunk_document(text, spans, 12, tok)

        self.assertNotIn("", tok.calls)
        self.assert_tiles(chunks, text)
        self.assertEqual([(c.first_page, c.last_page) for c in chunks], [(1, 3), (4, 4)])

    async def test_trailing_empty_pages_do_not_produce_an_empty_chunk(self):
        # The last two pages sit at `len(text)` with zero width. Emitting a Chunk for them
        # would put a zero-width span in the Sink and break the tiling the offsets promise.
        text, spans = _document(["a" * 20, "b" * 20, "c" * 20, None, None])
        tok = _FakeTokenizer()

        chunks = await chunk_document(text, spans, 6, tok)

        self.assert_tiles(chunks, text)
        self.assertTrue(all(c.end_char > c.start_char for c in chunks))
        self.assertEqual(chunks[-1].last_page, 5)

    async def test_leading_empty_pages_are_named_by_the_first_real_chunk(self):
        text, spans = _document([None, None, "a" * 20, "b" * 20, "c" * 20])
        tok = _FakeTokenizer()

        chunks = await chunk_document(text, spans, 12, tok)

        self.assertNotIn("", tok.calls)
        self.assert_tiles(chunks, text)
        self.assertEqual(chunks[0].first_page, 1)


class OversizedPageTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    async def test_page_over_budget_is_cut_at_newlines_and_marked_partial(self):
        text, spans = _document(["\n".join(["y" * 29] * 10)])

        chunks = await chunk_document(text, spans, 20, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.is_partial_page for c in chunks))
        for chunk in chunks[:-1]:
            self.assertEqual(text[chunk.end_char - 1], "\n")

    async def test_page_without_a_newline_takes_a_hard_cut_and_loses_nothing(self):
        text, spans = _document(["z" * 200])

        chunks = await chunk_document(text, spans, 10, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.is_partial_page for c in chunks))
        self.assertTrue(all(c.first_page == 1 and c.last_page == 1 for c in chunks))

    async def test_a_budget_no_page_can_meet_still_never_fails_the_document(self):
        text, spans = _document(["z" * 200])

        chunks = await chunk_document(text, spans, 1, _FakeTokenizer())

        self.assert_tiles(chunks, text)

    async def test_cut_pieces_stay_inside_the_budget(self):
        text, spans = _document(["\n".join(["w" * 17] * 12)])
        budget = 15

        chunks = await chunk_document(text, spans, budget, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        for chunk in chunks:
            self.assertLessEqual(_count(text[chunk.start_char : chunk.end_char]), budget)

    async def test_the_tail_of_a_cut_page_packs_with_the_following_page(self):
        # The remainder of a cut page is usually far short of the budget; leaving it open is
        # what stops the Document that could least afford it from wasting most of a context.
        text, spans = _document(["z" * 100, "w" * 20])

        chunks = await chunk_document(text, spans, 12, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        self.assertEqual((chunks[-1].first_page, chunks[-1].last_page), (1, 2))
        self.assertTrue(chunks[-1].is_partial_page)

    async def test_a_whole_page_chunk_after_a_cut_page_is_not_marked_partial(self):
        # The flag has to describe the Chunk, not the Document: a Chunk that begins and ends
        # on page boundaries is quotable as whole pages even when a neighbour was cut.
        text, spans = _document(["z" * 400, "w" * 20, "v" * 20])

        chunks = await chunk_document(text, spans, 12, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        self.assertFalse(chunks[-1].is_partial_page)
        self.assertEqual((chunks[-1].first_page, chunks[-1].last_page), (2, 3))
        self.assertTrue(chunks[-2].is_partial_page)


class SpanlessRecordTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    async def test_text_without_spans_is_kept_under_an_unnumbered_page(self):
        tok = _FakeTokenizer()

        chunks = await chunk_document("some text with no page spans", [], 100, tok)

        self.assert_tiles(chunks, "some text with no page spans")
        self.assertEqual((chunks[0].first_page, chunks[0].last_page), (0, 0))


if __name__ == "__main__":
    unittest.main()
