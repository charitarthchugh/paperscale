"""Embedding-model Adapters -- the three facts a serving engine cannot be trusted for.

The rule is "ask the server for everything it can be *trusted* on; the Adapter carries only the
rest". The first form of that rule -- "hardcode nothing but the model id" -- is not implementable:
an inventory of four serving engines (#23) found three facts that do not go away.

1. **MRL validity, as a range ``[min_dim, native_dim]`` and never a list.** Neither pinned family
   publishes an enumeration. Qwen3's cards state a continuous range, "32 to N", where N is that
   size's own native width; N = 2560 for the 4B also rules out the tempting "the valid points are
   powers of two" shortcut. Nemotron's cards say "for example, keeping the first 1024 or 512
   dimensions" -- examples of a range, not a list.
2. **The Instruction convention and its literal strings, for both the document and the query
   side.** No engine reports either one, and a serving flag can change what the model sees without
   appearing in any response (TEI accepts ``--default-prompt`` and ``/info`` does not report it).
3. **The card context length.** Undiscoverable, and every engine over-reports it *in the unsafe
   direction*: a default ``vllm serve`` advertises ``max_model_len`` 262144 for Nemotron and 40960
   for Qwen3-8B against a documented 32768. ``budget.py`` takes the ``min()`` of the two.

Everything else is asked of the server and recorded: the served model id from ``/v1/models``, exact
token counts from ``POST /v1/tokenize``, and the output width from the response itself -- a probe,
never an ask, because no surveyed engine reports it (huggingface/text-embeddings-inference#148 is
still true; it was closed COMPLETED by a comment that reframed the question, not by a fix).

These Adapters live here and **not** in ``paperscale/models/``: that package's ``__init__`` eagerly
imports all nine OCR Adapters, so an embed Adapter there would drag the embed package -- and with
it numpy -- into every OCR Run.
"""

from __future__ import annotations

import abc


class EmbedModel(abc.ABC):
    """The facts about one embedding model that paperscale cannot obtain at run time.

    Five typed class attributes and no methods. An OCR Adapter builds a prompt and parses a
    response; an embedding Adapter has neither job, so subclasses are pure fact tables and the
    shape matches ``default_model_name`` / ``preferred_longest_image_dim`` on
    :class:`paperscale.models.base.OCRModel`.
    """

    #: The model card's context length. Never the server's ``max_model_len``: every engine
    #: surveyed reports a larger number than the card, i.e. wrong in the direction that silently
    #: truncates a Chunk. ``budget.py`` min()s this against the server's answer.
    card_context_length: int

    #: The width the model actually emits, published per size. Doubles as the wrong-model
    #: assertion (#34): MRL slicing is client-side, so every response arrives at native width and
    #: a mismatch is one length comparison. Without it, pointing ``--embed-model`` at a server
    #: holding a different size of the same family produces a Sink that looks perfect and is
    #: wrong -- once every vector is cut to 768 the substitution leaves no trace, and Resume,
    #: which asks only about Document names, will not catch it either.
    native_dim: int

    #: The lowest width worth storing. Qwen3's published floor, adopted as the convention for both
    #: families. It is an Adapter constant because vLLM's own accepted range is ``[1,
    #: embedding_size]`` -- the server's floor is 1, so this quality claim is one no engine will
    #: ever enforce on our behalf.
    min_dim: int = 32

    #: Prefix paperscale applies to every Document before embedding, and records in both Sinks.
    #: Recording it is load-bearing rather than defensive: a Consumer that does not know Documents
    #: were prefixed will build queries that do not match them.
    document_instruction: str

    #: The query-side convention, recorded for the Consumer and **never applied here** -- queries
    #: are the Consumer's side of the seam. A ``{task_description}`` slot means the Consumer
    #: supplies its own text; no slot means use the string as it is.
    query_instruction: str


class _Qwen3Embedding(EmbedModel):
    """Shared facts for the Qwen3-Embedding family.

    Family base, one concrete Adapter per *size*, because ``native_dim`` is the fact that changes
    across a family and it is also the wrong-model assertion -- a single family-level Adapter
    could not hold one width and would have to guess.
    """

    card_context_length = 32768

    #: Empty string, **never None**. Qwen's card says "No need to add instruction for retrieval
    #: documents", so "" is a recorded decision; a missing value would say we never looked.
    #: Provenance has to keep those apart, or a later reader of the schema tidies one into the
    #: other and the distinction is gone.
    document_instruction = ""

    #: Copied character by character from Qwen's own ``get_detailed_instruct`` helper: a space
    #: after ``Instruct:``, **none** after ``Query:``. Qwen publishes that omitting it costs
    #: roughly 1-5% of retrieval performance -- paperscale never applies it, but the Consumer
    #: cannot make that trade without the string in front of it.
    query_instruction = "Instruct: {task_description}\nQuery:{query}"


class Qwen3Embedding0_6B(_Qwen3Embedding):
    """Qwen/Qwen3-Embedding-0.6B -- 1024 native, MRL down to 32."""

    native_dim = 1024


class Qwen3Embedding4B(_Qwen3Embedding):
    """Qwen/Qwen3-Embedding-4B -- 2560 native, the width that rules out any powers-of-two rule."""

    native_dim = 2560


class Qwen3Embedding8B(_Qwen3Embedding):
    """Qwen/Qwen3-Embedding-8B -- 4096 native, indistinguishable from the Nemotron 8B by width alone."""

    native_dim = 4096


class _Nemotron3Embed(EmbedModel):
    """Shared facts for the NVIDIA Nemotron-3-Embed family.

    The design first generalized from Qwen3 and assumed the Instruction convention was query-side
    only. It is not (#23): this family requires ``passage: `` on documents, which is why applying
    the document side is paperscale's job and recording it is not optional.
    """

    card_context_length = 32768

    document_instruction = "passage: "

    #: A fixed string, not a template -- no ``{task_description}`` slot, so the Consumer uses it
    #: verbatim. The shape difference from Qwen3 is exactly why this is recorded as a string
    #: rather than as a family name the Consumer would have to look up.
    query_instruction = "query: "


class Nemotron3Embed1B(_Nemotron3Embed):
    """nvidia/Nemotron-3-Embed-1B -- 2048 native."""

    native_dim = 2048


class Nemotron3Embed8B(_Nemotron3Embed):
    """nvidia/Nemotron-3-Embed-8B -- 4096 native."""

    native_dim = 4096


# Keys are lowercase-hyphenated, mirroring MODEL_REGISTRY so an engineer who knows one knows the
# other. There is deliberately **no DEFAULT_EMBED_MODEL**, the one place the mirror is not exact
# (#35): OCR models produce text that is roughly comparable across models, embedding models
# produce vectors that are meaningless across models, and both Sinks bake the model into the
# output's identity -- so a default would silently pick the semantics of an entire corpus.
EMBED_MODEL_REGISTRY: dict[str, type[EmbedModel]] = {
    "qwen3-embedding-0.6b": Qwen3Embedding0_6B,
    "qwen3-embedding-4b": Qwen3Embedding4B,
    "qwen3-embedding-8b": Qwen3Embedding8B,
    "nemotron-3-embed-1b": Nemotron3Embed1B,
    "nemotron-3-embed-8b": Nemotron3Embed8B,
}


def build_embed_model(name: str) -> EmbedModel:
    """Instantiate a registered embedding-model adapter by name."""
    try:
        factory = EMBED_MODEL_REGISTRY[name]
    except KeyError:
        choices = ", ".join(sorted(EMBED_MODEL_REGISTRY))
        raise ValueError(f"Unknown --embed-model {name!r}. Choose from: {choices}") from None
    return factory()


def validate_embed_dim(adapter: EmbedModel, embed_dim: int) -> int:
    """Check ``--embed-dim`` against this model's MRL range and return the stored width.

    Both ends stop the Invocation rather than adjusting the number. Below ``min_dim`` the server
    would happily comply -- vLLM accepts any width down to 1 -- and hand back vectors the model's
    own publisher does not claim are usable.

    Above ``native_dim`` the rejection is the interesting one: slicing only cuts, so the request is
    unsatisfiable, and clamping to ``native_dim`` would look like success. Someone asking a
    2048-wide model for 4096 has a wrong mental model (usually: they meant the 8B sibling), and
    every downstream artifact of that mistake -- a Sink recording ``stored_dim`` 2048, vectors that
    search perfectly well -- looks exactly like a correct Invocation.
    """
    if embed_dim < adapter.min_dim:
        raise SystemExit(
            f"--embed-dim {embed_dim} is below {adapter.min_dim}, the narrowest width this model is published as usable at. "
            f"Valid range for this model: {adapter.min_dim}-{adapter.native_dim}."
        )
    if embed_dim > adapter.native_dim:
        raise SystemExit(
            f"--embed-dim {embed_dim} is above this model's native width of {adapter.native_dim}. "
            f"Matryoshka slicing can only cut a vector down, so this is refused rather than quietly served at {adapter.native_dim}. "
            f"Valid range for this model: {adapter.min_dim}-{adapter.native_dim}."
        )
    return embed_dim


__all__ = [
    "EMBED_MODEL_REGISTRY",
    "EmbedModel",
    "Nemotron3Embed1B",
    "Nemotron3Embed8B",
    "Qwen3Embedding0_6B",
    "Qwen3Embedding4B",
    "Qwen3Embedding8B",
    "build_embed_model",
    "validate_embed_dim",
]
