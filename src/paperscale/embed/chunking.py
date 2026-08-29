r"""Greedy page packing: cut one Document's text into Chunks that tile it exactly once.

Two facts shaped all of this (design section 5.1).

**The page spans are already exact character ranges.** `build_dolma_document`
(`src/paperscale/pipeline.py`) writes `attributes.pdf_page_numbers` as
`[start_char, end_char, page_num]` triples that *tile* the text -- `span[i].end ==
span[i+1].start`, no gaps -- so page-respecting chunking costs no offset arithmetic: a run
of consecutive pages *is* a character range. Two consequences are load-bearing below. The
`\n` joiner between pages is folded into the *preceding* page's span, so a page slice
carries its own trailing newline (every page but the last), which is why a Chunk boundary
that lands on a page boundary needs no adjustment to keep the text reconstructible. And a
page whose `natural_text` was `None` emits a *zero-width* span: a real entry that costs zero
tokens and must never be able to close a Chunk that still had room.

**paperscale cannot count tokens.** There is no `transformers`, `tokenizers` or `tiktoken`
in `pyproject.toml`, so counting is a new capability rather than a call to something that
exists. The count is therefore asked of the server that holds the tokenizer actually in use
(`tokenize` here is `client.EmbedClient.tokenize`, injected). That is why the shape of the
algorithm is a *call budget*: one call for the whole Document, a call per page only for a
Document that overflows, and a call per piece only for the single page that overflows on its
own. The rarest path pays the most, which is the right way round.

**Assembled Chunks are never re-tokenized**, and the reason is a proof rather than a
measurement: a BPE merge cannot span a split boundary, so splitting a string can only raise
its token count -- `sum(tokenize(page_i)) >= tokenize(concat(page_i))`. Packing by summed
per-page counts errs in the safe direction, so a Chunk whose pages sum under the budget
cannot exceed it when tokenized whole. This is what keeps the Overflow path at N calls
instead of N plus one re-check per Chunk.

**A boundary inside a page is measured, never estimated.** That proof covers only Chunks
assembled out of whole pages; it says nothing about a boundary placed *inside* one. Such a
boundary used to be placed with the Document's average chars-per-token ratio, and an average
cannot vouch for a page denser than itself: a page of formulae or CJK is short in characters
and long in tokens, so the estimated budget position landed past the page's own end, the
cut collapsed onto that end, and the whole over-budget page shipped as a single Chunk. The
server answers such a Chunk with a 400 and fails the Document, and the token count recorded
for it -- the pooling weight `vectors.py` reads -- came out several times too small. So
every piece cut out of an oversized page is counted on the wire before it is closed, and so
is the remainder left open for the following pages to pack onto. Those counts are the only
thing that makes the size of a Chunk a fact rather than a hope, and they fall only on a page
that alone exceeds the budget.

**Chunks do not overlap.** Overlap would double-count the shared spans in the pooled
Document vector and stop the stored offsets describing a partition, so reconstruction from
the offsets would no longer round-trip. Overlap is a retrieval-side tactic and belongs to
the Consumer, which holds the exact offsets and can re-chunk without help.

Rejected: rebuilding character offsets from `/tokenize`'s `return_token_strs`.
`evaluation/pplx.py:201` already carries the warning -- decoded tokens carry marker glyphs
(SentencePiece, BPE) whose handling differs per tokenizer, so character attribution built on
them breaks silently when the model changes. Slicing by the offsets already in hand is exact
and model-independent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """One Chunk of a Document, recorded in both coordinate systems.

    `[start_char, end_char)` is the primitive: it is the only pair that can express a
    boundary *inside* a page, and it is what makes a Chunk reconstructible by slicing
    `record["text"]`. `[first_page, last_page]` is derived from it and is what a citation
    shows a human. Pages alone would lose the partial-page case; offsets alone would make
    every Consumer re-derive pages from `pdf_page_numbers`.

    `token_count` is exact when the Document fits in one Chunk and exact for every piece cut
    out of an oversized page; for a Chunk packed out of whole pages it is an upper bound (the
    sum of the per-page counts, by the subadditivity argument in the module docstring). It is
    never an estimate. `vectors.py` uses it as a pooling weight, so a count that is wrong by a
    factor is text weighted wrongly in the Document vector, which is a silent error rather
    than a loud one.

    `is_partial_page` marks the one path that puts a boundary inside a page, so a Consumer
    can tell a Chunk that is quotable as "pages 4-6" from one that is not.
    """

    start_char: int
    end_char: int
    first_page: int
    last_page: int
    token_count: int
    is_partial_page: bool


async def chunk_document(text: str, spans: list[list[int]], budget: int, tokenize: Callable[[str], Awaitable[int]]) -> list[Chunk]:
    """Pack `text` into Chunks of at most `budget` tokens, respecting page boundaries.

    `spans` is the Record's `attributes.pdf_page_numbers`. `tokenize` is an async
    `(str) -> int`; it is a parameter rather than a client attribute so a test can inject a
    fake and *count the calls*, which is the only way to prove the common case stays at one.

    The common case -- a Document that fits whole -- costs exactly one call and returns one
    Chunk. Only a Document that overflows pays one further call per non-empty page, and only
    a single page that overflows on its own produces a boundary inside a page -- and pays the
    further calls that placing those boundaries exactly costs. Text is never dropped and no
    input fails the Document; a Document with nothing to embed comes back as zero Chunks,
    which is a recorded outcome (an empty output) and not a failure.
    """
    if not text or not text.strip():
        return []
    if not spans:
        # `build_dolma_document` emits one span per page result and returns None when the
        # text came out empty, so text-without-spans is not something this pipeline can
        # produce. Treat it as one unnumbered page rather than drop the text or raise from
        # inside the packer: page 0 is a number the OCR side never assigns, so a Consumer
        # reading a citation of "page 0" learns the truth, which is that none was recorded.
        spans = [[0, len(text), 0]]

    total = await tokenize(text)
    if total <= budget:
        return [Chunk(start_char=0, end_char=len(text), first_page=spans[0][2], last_page=spans[-1][2], token_count=total, is_partial_page=False)]

    # Overflow path. Every size decision from here down rests on a count the server gave, not
    # on a ratio derived from the Document as a whole: whole pages are packed by their own
    # exact counts here, and a page that has to be cut is measured piece by piece below.
    counts: list[int] = []
    for start, end, _page in spans:
        # A zero-width page is answered here rather than on the wire. /tokenize prepends
        # the model's BOS by default, so the server answers "" with 1, and a page that cost
        # one token could close a Chunk that still had room -- exactly the break the design
        # says an empty page can never force. It also saves a round trip per empty page.
        counts.append(await tokenize(text[start:end]) if end > start else 0)

    chunks: list[Chunk] = []
    open_chunk = _OpenChunk(spans[0][0])

    for (page_start, page_end, page), count in zip(spans, counts):
        if count > budget:
            logger.debug("page %s spans %d chars and %d tokens, over the %d-token budget; cutting inside the page", page, page_end - page_start, count, budget)
            open_chunk.close(page_start, chunks, page)
            await _cut_oversized_page(text, page_start, page_end, page, count, budget, tokenize, open_chunk, chunks)
        elif open_chunk.token_count + count > budget:
            # Close at the page's first character: the previous page's span already carries
            # the `\n` joiner, so the two Chunks meet exactly and neither loses a byte.
            open_chunk.close(page_start, chunks, page)
            open_chunk.add(page, count)
        else:
            open_chunk.add(page, count)

    # Close on `len(text)` rather than on the last span's end, so the recorded Chunks tile
    # the whole Document even if a hand-built Record's spans stop short of its text.
    open_chunk.close(len(text), chunks, spans[-1][2])
    return chunks


class _OpenChunk:
    """The Chunk currently being filled.

    An object rather than five locals because the oversized-page path closes and reopens it
    from a second function, and because closing it is not always an emit: a close at a point
    that equals its start means only zero-width pages have accumulated there. Those pages
    stay pending instead of becoming an empty Chunk, so the next real Chunk still names them
    and no page number is lost.
    """

    def __init__(self, start: int) -> None:
        self.start = start
        self.first_page: int | None = None
        self.last_page: int | None = None
        self.token_count = 0
        self.is_partial_page = False

    def add(self, page: int, tokens: int, *, partial: bool = False) -> None:
        if self.first_page is None:
            self.first_page = page
        self.last_page = page
        self.token_count += tokens
        self.is_partial_page = self.is_partial_page or partial

    def close(self, end: int, out: list[Chunk], fallback_page: int) -> None:
        if end <= self.start:
            return
        # `fallback_page` covers only the spans-stop-short case above; a Chunk with width
        # always has at least one page in a Record this pipeline produced.
        first = self.first_page if self.first_page is not None else fallback_page
        last = self.last_page if self.last_page is not None else fallback_page
        out.append(
            Chunk(start_char=self.start, end_char=end, first_page=first, last_page=last, token_count=self.token_count, is_partial_page=self.is_partial_page)
        )
        self.start = end
        self.first_page = None
        self.last_page = None
        self.token_count = 0
        self.is_partial_page = False


async def _cut_oversized_page(
    text: str,
    page_start: int,
    page_end: int,
    page: int,
    page_tokens: int,
    budget: int,
    tokenize: Callable[[str], Awaitable[int]],
    open_chunk: _OpenChunk,
    chunks: list[Chunk],
) -> None:
    """Split a page that alone exceeds the budget, at newlines where there are any.

    Cut at the last `\\n` at or before the budget position; a page with no newline in reach
    takes a hard character cut. Never drop text, never fail the Document -- this is the only
    path that produces a boundary inside a page, and it is why `is_partial_page` exists as a
    recorded field.

    `remaining` is what makes this exact: it always holds the *server's* count of
    `text[pos:page_end]`, starting from the count the caller already paid for. Two things
    come out of it. The loop condition is a fact rather than an estimate, so the last piece
    is cut when the rest genuinely does not fit and not when a ratio suggests it might. And
    the ratio handed to `_fit_piece` is measured on exactly the text still to be cut, so
    density that drifts *within* a page -- prose, then a formula block -- is re-anchored
    after every cut rather than averaged away over a Document.

    The tail is left open rather than closed, so following pages pack onto it: the remainder
    of a cut page is usually far short of the budget, and closing it would waste most of a
    32K context on the Document that could least afford it. It carries its exact count out
    with it, which is what keeps the packing decisions that follow inside the budget too.
    """
    pos = page_start
    remaining = page_tokens
    while remaining > budget and pos < page_end:
        cut, count = await _fit_piece(text, pos, page_end, budget, (page_end - pos) / remaining, tokenize)
        # Derived rather than asserted. A piece is partial unless it runs from one edge of
        # the page to the other, and a Consumer reading the flag is being told whether it may
        # cite whole pages -- so it has to describe the span that was actually cut.
        open_chunk.add(page, count, partial=pos > page_start or cut < page_end)
        open_chunk.close(cut, chunks, page)
        pos = cut
        remaining = await tokenize(text[pos:page_end]) if pos < page_end else 0
    if pos < page_end:
        open_chunk.add(page, remaining, partial=pos > page_start)


async def _fit_piece(text: str, start: int, page_end: int, budget: int, chars_per_token: float, tokenize: Callable[[str], Awaitable[int]]) -> tuple[int, int]:
    """The next piece of an oversized page: its exclusive end, and its exact token count.

    `chars_per_token` is measured on the text still to be cut, so the first candidate is
    usually right and usually costs a single call. It is still only a ratio, so a candidate
    the server counts over the budget is shrunk by the density it just measured *on that
    candidate* and asked again -- one Newton-ish step, which lands inside the budget in one
    or two tries even when the page's own density is uneven.

    The loop terminates because each attempt is strictly shorter than the last (`cut - 1`
    caps the next limit) and never shorter than one character. Bottoming out at a single
    character returns it over budget rather than raising: a budget no character can meet is
    a broken budget, and failing the Document is the one thing this path may never do.
    """
    limit = min(start + max(1, int(budget * chars_per_token)), page_end)
    while True:
        cut = _cut_point(text, start, limit)
        count = await tokenize(text[start:cut])
        if count <= budget or cut - start <= 1:
            return cut, count
        limit = min(start + max(1, int((cut - start) * budget / count)), cut - 1)


def _cut_point(text: str, start: int, limit: int) -> int:
    """The exclusive end of the next piece: after the last `\\n` before `limit`, else `limit`.

    The newline stays with the piece it terminates, matching how a page span keeps its own
    joiner, so the pieces still tile. `limit` is exclusive in the search as well, so a piece
    never reaches past the candidate budget position. The result is always greater than
    `start`, which is what guarantees the caller's loops terminate.
    """
    newline = text.rfind("\n", start, limit)
    return limit if newline < 0 else newline + 1
