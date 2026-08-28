"""Turn a model card, a served deployment and two flags into three token numbers.

Four numbers describe one model's window, not two. For Nemotron-3-Embed-8B, vLLM
advertises 262144 on a default serve, `config.json`'s `max_position_embeddings`
agrees at 262144, the card *states* 32768, and the card *exercises* 4096 -- every
published score for that model was produced at 4096. The 262144 is neither
arbitrary nor a bug: the checkpoint's `rope_parameters` are yarn with
`factor: 16.0` over `original_max_position_embeddings: 16384`, and
16384 * 16 = 262144 exactly. Qwen3 carries `rope_scaling: null` and simply
advertises its 40960, which is why only one of the two families shows a wild
number.

Hence the default of `min(card, server)`: **the card is authoritative about the
model, the server is authoritative about the deployment, and neither is
authoritative about both.** `min()` takes the safe half of each. Stated as the
governing rule it sharpens -- *the server can be trusted as an upper bound and
not as a lower one*. That protects against the 262144 advertisement in one
direction and, in the other, against an operator who serves
`--max-model-len 8192` against a 32768 card and would otherwise hard-fail every
long Document.

The card-exercised numbers (Qwen3's 8192, Nemotron's 4096) are recorded here and
deliberately not used: choosing them would be paperscale making a quality call it
cannot measure, and it lands hardest on the classification Consumer -- at 4096
the smoke corpus goes from 52 Chunks to 97, so the Document vector becomes a mean
over roughly two Chunks instead of one. `--context-length` is what makes that
reversible the moment the Consumer has evidence.

Design authority: `docs/design/embed.md` sections 4 and 12.3. Read them before
changing any number here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# NOT a defence against packing -- subadditivity already covers packing, and covers the
# Instruction with it, since tokens("passage: ") + tokens(text) >= tokens("passage: " + text).
# What is left is a ROUTE-level risk: paperscale counts tokens on /v1/tokenize and sends text
# to /v1/embeddings, and if those two routes apply special tokens differently by even one
# token, a Chunk sitting exactly on the budget hard-fails -- the design deliberately relies on
# vLLM *erroring* on overflow rather than truncating. 64 tokens is 0.2% of the context against
# a visible hard failure. The 868 implied by an early manifest example is stale: it was never
# stated as a decision, and 2.6% of the context is a lot to spend on nothing recorded.
SAFETY_MARGIN = 64


def resolve_context_length(*, card_context_length: int, server_max_model_len: int, override: int | None) -> int:
    """Resolve the window this Invocation will treat as the model's, in tokens.

    Only `server_max_model_len` is a correctness boundary. Above it the request
    cannot succeed at all, so an override above it is rejected at startup rather
    than left to fail once per Document.

    The card's number is a *quality claim*, and the evidence behind it is thinner
    than the card implies -- Nemotron's "max sequence length is 32768" sits beside
    benchmark scores all produced at 4096. Refusing an operator that trade would
    be paperscale enforcing a quality opinion under cover of a safety check, and
    this design assigns embedding-quality judgement to the Consumer. So above the
    card is allowed and only warned about.

    Nothing about the override reaches a Sink. Whether it went above the card is
    fully derivable from `chunk_budget_tokens` against the `card_context_length`
    that `model_id` identifies, and storing a derived value invites the two copies
    to disagree.
    """
    if override is None:
        return min(card_context_length, server_max_model_len)

    if override > server_max_model_len:
        raise SystemExit(
            f"--context-length {override} is above the server's max_model_len of {server_max_model_len}; "
            f"every request that used it would be rejected by the server. Re-serve with a larger "
            f"--max-model-len, or ask for at most {server_max_model_len}."
        )

    if override > card_context_length:
        # Both numbers, and the actual risk. Never "may reduce quality": that phrasing
        # understates it by implying a measurement exists on the far side. There is none --
        # the risk is an absence of evidence, and the warning has to say so.
        logger.warning(
            "--context-length %d is above the model card's %d. The card's number is a quality claim, not a correctness "
            "boundary: above it, embedding quality is unmeasured by the vendor. The server accepts up to %d, so the "
            "requests will still succeed.",
            override,
            card_context_length,
            server_max_model_len,
        )

    return override


def chunk_budget(validated_context_length: int, instruction_tokens: int) -> int:
    """Largest Chunk, in tokens, that can carry the Instruction and still fit.

    `instruction_tokens` is the document-side Instruction's exact count -- one
    `/v1/tokenize` call at startup, or 0 for Qwen3's empty string. The Adapter
    applies the Instruction, so its tokens are spent out of the same window the
    Chunk text is measured against.

    On a default serve of either pinned family this lands at 32704 for Qwen3
    (empty Instruction) and about 32701 for Nemotron (`"passage: "`, about three
    tokens). An early manifest *example* showing 31900 is stale and must not be
    copied back in.
    """
    return validated_context_length - instruction_tokens - SAFETY_MARGIN


def request_budget(flag_value: int, chunk_budget_tokens: int) -> int:
    """Per-request token budget, floored at one full-size Chunk.

    A request may pack Chunks from several Documents, but it can never be allowed
    to hold fewer than one, or a Chunk at the full budget could not be sent at
    all.

    The floor is enforced by **raising**, never by rejecting, and the reason is
    arithmetic rather than taste: at `SAFETY_MARGIN = 64` the 32000 default sits
    about 704 tokens *below* the floor for both pinned families, so rejecting
    would reject the default configuration. Silently ignoring the floor is the
    other bad option -- it would leave a full-size Chunk unsendable, which is the
    real fault. Raising is always safe, because it only ever permits a larger
    request; a token budget bounds HTTP round trips, not engine work (vLLM fans an
    `input` array of N texts into N independent engine requests).

    The line is INFO rather than WARNING deliberately: it fires on every default
    Invocation of both pinned families, and a warning that is always on is a
    warning nobody reads.
    """
    if flag_value < chunk_budget_tokens:
        logger.info(
            "--request-tokens %d is below the Chunk budget of %d; raising it to %d so a full-size Chunk can be sent.",
            flag_value,
            chunk_budget_tokens,
            chunk_budget_tokens,
        )
        return chunk_budget_tokens
    return flag_value
