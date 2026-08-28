"""MRL slicing, re-normalization, and the token-weighted pool.

The order is load-bearing: **slice each Chunk vector to `stored_dim`, re-normalize
it, and only then pool**. Pooling at native width and slicing the result was
rejected because the Document vector would then depend on coordinates that no Sink
ever writes -- a Consumer holding one Sink could not reproduce it. Under this order
it can, and gets the same answer. The slice itself commutes with averaging; only
the per-Chunk re-normalization makes the two orders differ at all.

Slicing here rather than asking the server for `dimensions` is not a numerical
choice, because the two routes are the same arithmetic: slicing is coordinate
selection, hence linear, and normalizing divides by a positive scalar that then
cancels, so `normalize(slice(raw))` and `normalize(slice(normalize(raw)))` are the
same unit vector. It is an operational choice. `dimensions` is a real vLLM
parameter, but all four pinned `config.json` files declare neither `is_matryoshka`
nor `matryoshka_dimensions`, so `PoolingParams` raises *before* inference on a
default launch -- a server flag paperscale cannot set, cannot verify, and that
fails at run time when an operator forgets it. The client-side slice has no
precondition to forget, and costs only bandwidth on a LAN.

numpy is imported inside each function on purpose: it ships in the optional
`embed` extra, and `import paperscale.embed` must keep working without it.

Design: `docs/design/embed.md` sections 6.1 and 6.2.
"""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from paperscale.embed.chunking import Chunk


@dataclasses.dataclass(frozen=True)
class EmbeddedDocument:
    """One Document's finished vectors plus the per-Document provenance.

    `chunks` and the rows of `chunk_vectors` are parallel: Chunk `i` is
    `chunk_vectors[i]`. That correspondence is why neither Sink stores a
    `chunk_index` array -- it is exactly `arange(len(chunks))`, and a stored copy
    of a derived value is a second source of truth waiting to disagree.

    `source_file` is the **raw, unnormalized** `Source-File` from the Record, not
    the derived Document name: the name is lossy by construction (a tarball member
    and a leading slash both collapse), so the only way a Consumer can get back to
    the PDF is the string the OCR Run recorded. `source_digest` is over that same
    string and never over the text -- the text is the thing being embedded, and
    hashing it would make the identity of a Document change whenever the OCR model
    did.

    `created` is timezone-aware UTC because both Sinks serialize it: the `.npz`
    sidecar writes ISO-8601 `Z` and LanceDB wants `timestamp[us, UTC]`, and a naive
    datetime silently means "whatever the writing machine's clock was set to".
    """

    document_name: str
    run_label: str
    source_file: str
    source_digest: str
    created: datetime.datetime
    chunks: list[Chunk]
    chunk_vectors: np.ndarray
    document_vector: np.ndarray


def slice_and_normalize(raw, stored_dim: int) -> np.ndarray:
    """Cut `(n, native_dim)` server vectors to `(n, stored_dim)` unit-length rows.

    The re-normalization is not cosmetic. The server returns vectors normalized at
    *native* width; a slice of a unit vector is shorter than unit, and by a
    different factor per row. Skipping this step is a real shipping bug rather than
    a hypothetical one -- vLLM's own long-text embedding path never re-normalizes
    its cross-chunk mean, and Vespa's MRL truncation defaults `normalize` to false,
    so dot-product scoring quietly stops being cosine similarity.

    A row whose slice has zero (or non-finite) norm is an error rather than a
    division: the whole point of the Sink invariant is that every stored vector is
    unit length, and `NaN` satisfies no check downstream while still writing,
    loading, and searching without complaint.
    """
    import numpy as np

    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D (n_chunks, native_dim) array of vectors, got shape {vectors.shape}")
    if stored_dim < 1:
        raise ValueError(f"stored_dim must be at least 1, got {stored_dim}")
    if vectors.shape[1] < stored_dim:
        # The Adapter's native_dim check in the client is the primary guard; this is
        # the backstop for the day a server default moves under it, because a silent
        # numpy slice would hand back a narrower array that every later shape check
        # -- both Sinks included -- would happily accept as correct.
        raise ValueError(f"server returned {vectors.shape[1]}-wide vectors, narrower than the requested stored_dim {stored_dim}")

    # float64 for the two reductions (this norm, and the pooled sum below) and one
    # rounding at the end: the store is float32 either way, and widening the
    # accumulate costs nothing at 4096 coordinates while keeping the divide from
    # rounding a second time in the narrow type.
    sliced = vectors[:, :stored_dim].astype(np.float64)
    norms = np.linalg.norm(sliced, axis=1)
    if not np.all(norms > 0):
        # `> 0` rather than `!= 0` so a NaN row -- which compares False against
        # everything -- is caught here too, instead of propagating into a Sink.
        bad = int(np.argmin(norms))
        raise ValueError(f"vector {bad} has no usable direction in its first {stored_dim} dimensions (norm {norms[bad]!r}); refusing to write NaN")
    return (sliced / norms[:, None]).astype(np.float32)


def pool_document_vector(chunk_vectors, token_counts) -> np.ndarray:
    """Pool Chunk vectors into one Document vector, weighted by token count.

        document_vector = normalize(sum(token_count[i] * chunk_vector[i]))

    **Note the missing division.** Dividing the weighted sum by `sum(w)` is a no-op:
    the `normalize` that follows discards any positive scale. Leaving it out removes
    a step and one source of float error. (LangChain hand-rolls this same algorithm,
    short-circuit included, and *does* divide by the total weight first -- which is
    exactly the no-op.)

    **Why token-weighted at all**: invariance to the cut. The Chunk budget is an
    implementation detail that moves when the validated context length moves, and
    the oversized-page path can move it again inside one Document. Greedy packing
    makes the alternative concrete -- a two-Chunk Document can be forty pages plus a
    one-page tail, and a uniform mean hands that tail page half the Document vector.
    The knowing trade is that unit-length Chunk vectors plus token weights assert
    that text volume equals importance, which for a one-page cover letter in front
    of a forty-page exhibit is a claim about the corpus, not a fact.

    **The single-Chunk case is a copy, not an average**, and that does not fall out
    of the arithmetic: a weighted mean of one element is still re-normalized, and
    the divisor is `1 +/- eps` rather than `1`, so the low bits move (measured: on
    64 sampled 768-wide vectors, all 64 came back bitwise different). The identity
    case is then exact by construction rather than by tolerance, which is what lets
    a test assert bit-equality.
    """
    import numpy as np

    vectors = np.asarray(chunk_vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D (n_chunks, stored_dim) array of Chunk vectors, got shape {vectors.shape}")

    n_chunks = vectors.shape[0]
    if n_chunks == 0:
        # A zero-Chunk Document is a recorded outcome, not a failure, and stored_dim
        # is not knowable from an empty array -- both Sinks key the empty layout off
        # this shape.
        return np.zeros((0,), dtype=np.float32)
    if n_chunks == 1:
        # `.copy()` and not the row itself: a view would alias the Chunk vector, and
        # both Sinks write the two arrays independently.
        return vectors[0].copy()

    weights = np.asarray(token_counts, dtype=np.float64)
    if weights.shape != (n_chunks,):
        raise ValueError(f"got {weights.size} token counts for {n_chunks} Chunk vectors; they are parallel arrays")
    if weights.sum() == 0:
        # Unreachable in practice -- a Document with more than one Chunk overflowed
        # the budget, so it has tokens -- but the alternative is a zero vector, and
        # then a zero-norm error, in a path nobody exercises.
        weights = np.ones(n_chunks, dtype=np.float64)

    pooled = (vectors.astype(np.float64) * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(pooled))
    if not norm > 0:
        # Requires the Chunk vectors to cancel exactly. Still an error and not a
        # `NaN`: a Sink has no way to represent "this Document has no direction",
        # and every Consumer would read the NaN as a vector.
        raise ValueError(f"the token-weighted sum of {n_chunks} Chunk vectors has norm {norm!r}; refusing to write NaN")
    return (pooled / norm).astype(np.float32)
