"""Tests for the greedy page packer.

Every test drives `chunk_document` with a fake tokenizer, which is what makes the call
*count* observable -- the design's central affordability claim is that the common case costs
exactly one `/tokenize` call per Document, and nothing else can prove it.

Documents are built by `_document`, which reproduces `build_dolma_document`'s span layout
byte for byte: the `\\n` joiner folded into the preceding page, and a zero-width span for a
page whose `natural_text` was None. Hand-written spans would test a layout the pipeline does
not produce.

Two fake tokenizers, and the second one is the point. `_FakeTokenizer` charges four
characters per token *everywhere*, which makes any chars-per-token ratio measured on a whole
Document exactly right for every piece of it -- so under it the packer's arithmetic cannot be
caught being wrong about a page, and design obligations 9 and 11 (section 18.3) proved only
that the packer works on text of uniform density. `_VariableDensityTokenizer` charges one
token per dense character and sixteen sparse characters per token, a sixteen-fold spread,
which is the range that separates CJK or formula-heavy text from Latin prose. Every property
those obligations name is asserted under it as well.
"""

from __future__ import annotations

import math
import unittest

from paperscale.embed.chunking import Chunk, chunk_document

_CHARS_PER_TOKEN = 4

# One token per occurrence, against sixteen characters per token for everything else.
_DENSE = "漢"
_SPARSE_CHARS_PER_TOKEN = 16


class _FakeTokenizer:
    """Stands in for `EmbedClient.tokenize`, recording every string it is asked about.

    `ceil(len(s) / 4)` is subadditive the same way a BPE count is -- splitting a string can
    only raise the total -- so a packer that respects the budget under this counter respects
    it under a real one. The empty string answers 1, not 0, to imitate the BOS token
    /tokenize prepends by default: if the packer ever asked the server about a zero-width
    page, that 1 would show up as a token an empty page had cost.

    Uniform density is this counter's blind spot, not a feature: see
    `_VariableDensityTokenizer`, which is what the budget properties are also asserted under.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, text: str) -> int:
        self.calls.append(text)
        return _count(text)


class _VariableDensityTokenizer:
    """A tokenizer whose density depends on *which* characters it is given.

    Real tokenizers are wildly uneven: a CJK ideograph or a run of formula markup can cost a
    token per character where Latin prose costs several characters per token. That unevenness
    is the whole hazard on the path that cuts inside a page, and `_FakeTokenizer` cannot
    express it -- under a flat four-characters-per-token rule, a ratio measured on a whole
    Document is exactly right for every page of it, so an estimate can never be caught out.

    Still subadditive, which is what the packer's no-re-tokenize proof rests on: dense
    characters are counted one by one and so add exactly across a cut, and
    `ceil(x / 16) + ceil(y / 16) >= ceil((x + y) / 16)` covers the rest. The empty string
    answers 1 for the same BOS reason as above.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, text: str) -> int:
        self.calls.append(text)
        return _variable_count(text)


def _count(text: str) -> int:
    """`_FakeTokenizer`'s answer, without recording a call."""
    return 1 if not text else math.ceil(len(text) / _CHARS_PER_TOKEN)


def _variable_count(text: str) -> int:
    """`_VariableDensityTokenizer`'s answer, without recording a call."""
    if not text:
        return 1
    dense = text.count(_DENSE)
    return dense + math.ceil((len(text) - dense) / _SPARSE_CHARS_PER_TOKEN)


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
        #
        # Which page it lands on is pinned, and the pin is doing the work. Measured cuts
        # leave page 3 alone in the last Chunk; the estimated cuts this commit removed left
        # pages 2 and 3 in it together. Both land on a page boundary and both come out
        # whole, so "opens on some boundary and is not partial" is satisfied by the
        # behaviour that was wrong -- asserted that way, this test passes unchanged on the
        # previous commit and witnesses nothing.
        text, spans = _document(["z" * 400, "w" * 20, "v" * 20])

        chunks = await chunk_document(text, spans, 12, _FakeTokenizer())

        self.assert_tiles(chunks, text)
        self.assertFalse(chunks[-1].is_partial_page)
        self.assertEqual((chunks[-1].first_page, chunks[-1].last_page), (3, 3))
        self.assertTrue(chunks[-2].is_partial_page)


class VariableDensityTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    """Obligations 9 and 11 (section 18.3) under a tokenizer whose density is not uniform.

    Under `_FakeTokenizer` every one of these passes on a packer that sizes its cuts by a
    Document-wide chars-per-token average, because that average is exactly right for every
    page. Under this tokenizer it is right for no page, which is the situation a real corpus
    presents and the situation the budget guarantee has to survive.
    """

    def assert_within_budget(self, chunks: list[Chunk], text: str, budget: int) -> None:
        """The guarantee itself: nothing leaves the packer that the server would 400."""
        for chunk in chunks:
            whole = _variable_count(text[chunk.start_char : chunk.end_char])
            self.assertLessEqual(whole, budget, f"chunk [{chunk.start_char}, {chunk.end_char}) is {whole} tokens")
            # The recorded count is `vectors.py`'s pooling weight. Exact for a piece cut out
            # of a page, an upper bound for one packed out of whole pages, never below.
            self.assertGreaterEqual(chunk.token_count, whole)
            self.assertLessEqual(chunk.token_count, budget)

    async def test_a_page_denser_than_its_document_is_never_shipped_whole(self):
        # The reproduction. 4000 sparse characters cost 250 tokens and 800 dense ones cost
        # 800, so the Document averages ~4.6 characters per token while page 2 spends one per
        # character. Sizing page 2's cut by that average put the budget position (~1900
        # characters) past the page's own end, the cut collapsed onto that end, and the whole
        # 800-token page shipped as a single Chunk of four times the budget -- recorded as
        # 176 tokens, which is also the weight it would have been pooled at.
        text, spans = _document(["a" * 4000, _DENSE * 800])
        budget = 200

        chunks = await chunk_document(text, spans, budget, _VariableDensityTokenizer())

        self.assert_tiles(chunks, text)
        self.assert_within_budget(chunks, text, budget)

    async def test_a_chunk_covering_whole_pages_is_never_called_partial(self):
        # `is_partial_page` was asserted rather than derived on the cutting path, so the page
        # that escaped whole above was mislabelled too: a Consumer was told it could not quote
        # "page 2" for a Chunk that was exactly page 2, beginning and ending on its edges.
        text, spans = _document(["a" * 4000, _DENSE * 800])
        page_starts = {start for start, _end, _page in spans}
        page_ends = {end for _start, end, _page in spans}

        chunks = await chunk_document(text, spans, 200, _VariableDensityTokenizer())

        for chunk in chunks:
            whole_pages = chunk.start_char in page_starts and chunk.end_char in page_ends
            self.assertEqual(chunk.is_partial_page, not whole_pages)

    async def test_density_that_drifts_inside_one_page_still_yields_chunks_that_fit(self):
        # The same hazard one level in. Page 1 is sparse for 1600 characters and dense for the
        # next 300, so even a ratio measured on the page as a whole misjudges its second half.
        # The remainder left open after a cut is a packing input, so getting it wrong put the
        # following page onto a Chunk with no room: 376 tokens against a budget of 200,
        # recorded as 168.
        text, spans = _document(["a" * 1600 + _DENSE * 300, "a" * 800])
        budget = 200

        chunks = await chunk_document(text, spans, budget, _VariableDensityTokenizer())

        self.assert_tiles(chunks, text)
        self.assert_within_budget(chunks, text, budget)

    async def test_a_dense_page_is_cut_at_its_newlines_and_marked_partial(self):
        # Obligation 9 with the density taken out of the packer's favour. Page 1 fits, so the
        # only Chunk boundaries inside a page belong to page 2; the Document average would put
        # them ~600 characters apart, which is 361 tokens of this page.
        text, spans = _document(["a" * 1500, "\n".join([_DENSE * 40] * 10)])
        budget = 100

        chunks = await chunk_document(text, spans, budget, _VariableDensityTokenizer())

        self.assert_tiles(chunks, text)
        self.assert_within_budget(chunks, text, budget)
        self.assertFalse(chunks[0].is_partial_page)
        self.assertTrue(all(chunk.is_partial_page for chunk in chunks[1:]))
        for chunk in chunks[:-1]:
            self.assertEqual(text[chunk.end_char - 1], "\n")

    async def test_assembled_chunks_hold_when_pages_differ_in_density(self):
        # Obligation 11 where it already held, now proved rather than assumed: alternating
        # pages cost the same 101 tokens from 101 and 1601 characters respectively, so a
        # packer reasoning about characters anywhere would mis-pack. No page exceeds the
        # budget alone, so the call profile stays at one per Document plus one per page and no
        # assembled Chunk is ever sent back to the server.
        text, spans = _document([_DENSE * 100 if index % 2 else "a" * 1600 for index in range(8)])
        budget = 250
        tok = _VariableDensityTokenizer()

        chunks = await chunk_document(text, spans, budget, tok)

        self.assert_tiles(chunks, text)
        self.assert_within_budget(chunks, text, budget)
        self.assertEqual(tok.calls, [text] + [text[start:end] for start, end, _page in spans])
        self.assertFalse(any(chunk.is_partial_page for chunk in chunks))

    async def test_cutting_a_dense_page_costs_calls_only_on_that_page(self):
        # What the exactness is paid for in. The Document call and the per-page calls are
        # unchanged; the extra ones are the pieces of the page that had to be cut and the
        # remainder after each, and they are asked about that page's characters and nothing
        # else. A Document with no such page pays none of them, which the test above asserts.
        text, spans = _document(["a" * 1600, _DENSE * 600])
        tok = _VariableDensityTokenizer()

        chunks = await chunk_document(text, spans, 200, tok)

        self.assert_tiles(chunks, text)
        page_two = text[spans[1][0] : spans[1][1]]
        extra = tok.calls[1 + len(spans) :]
        self.assertTrue(extra)
        self.assertTrue(all(call in page_two for call in extra))


class SpanlessRecordTests(unittest.IsolatedAsyncioTestCase, TilingMixin):
    async def test_text_without_spans_is_kept_under_an_unnumbered_page(self):
        tok = _FakeTokenizer()

        chunks = await chunk_document("some text with no page spans", [], 100, tok)

        self.assert_tiles(chunks, "some text with no page spans")
        self.assertEqual((chunks[0].first_page, chunks[0].last_page), (0, 0))


if __name__ == "__main__":
    unittest.main()
