# paperscale

Large-scale OCR of PDF corpora against self-hosted inference servers, and the conversion of
that OCR output into vectors. paperscale's job ends when text or vectors exist and are
identified; searching, ranking and analysing them belongs to whatever consumes them.

> Seeded while charting the `paperscale embed` redesign
> ([#21](https://github.com/charitarthchugh/paperscale/issues/21)), now complete — the design it
> produced is `docs/design/embed.md`, which uses these terms exactly and is the authority on the
> decisions. This file stays the authority on the language. Terms outside that effort's scope are
> not yet captured.

## Language

### Corpus and documents

**Document**:
One source PDF and the text produced from it. The unit of work everywhere in paperscale.
_Avoid_: file, PDF, item

**Record**:
One line of a `results/*.jsonl` file — the OCR text of a single document plus its attributes.
_Avoid_: row, entry, JSONL line

**Run**:
One execution of paperscale over a corpus, identified by a **run label** supplied as
`label=path`. Two runs over the same corpus with different models are different runs.
_Avoid_: job, batch, pass

**Invocation**:
One execution of `paperscale embed`. It uses exactly one embedding model, and it can read more
than one **Run**, because `--run label=path` is repeatable. So a fact that varies between Runs
— the run label a Document came from — varies *within* an Invocation, while the model facts do
not.
_Avoid_: run, session, job

**Document name**:
The stable identifier for a document, derived from the `Source-File` path recorded in its
**Record**. Answers "which document is this?" and nothing else. There is no input directory at
embed time — `embed` reads JSONL, and the PDF tree root is recorded nowhere. Two documents may
never share one: the source extension is kept rather than replaced so the common collision cannot
arise, and any collision that survives that stops the **Invocation** before any work is done.
_Avoid_: doc_id, key, filename

### Embedding

**Adapter**:
The per-model seam that holds the facts about a model that no serving layer exposes. Selected
by name; one exists for OCR models and one for embedding models.
_Avoid_: backend, driver, provider, plugin

**Chunk**:
A contiguous span of one document's text, small enough for the embedding model to accept. A
document that fits entirely is a single chunk.
_Avoid_: segment, passage, window, split

**Chunk budget**:
The largest a **Chunk** may be, in tokens: the model's validated context length, less the
**Instruction** paperscale prepends, less a small margin. Derived per model, never chosen — an
operator can move the context length it is derived from, but cannot set the budget itself.
_Avoid_: max tokens, window size, context length

**Overflow**:
The condition where a document's text exceeds what the embedding model will accept, and must
therefore become more than one chunk. The minority path — but real, and large when it happens:
**3 of 49 documents (6%)** in a legal sample, against a ~32,700-token **chunk budget**.
_Avoid_: too long, truncation, oversized

**Chunk vector**:
The embedding of one chunk.
_Avoid_: embedding, chunk embedding

**Document vector**:
The single vector representing a whole document, reduced from its chunk vectors. Where a
document is one chunk, it equals that chunk's vector.
_Avoid_: pooled vector, doc embedding, centroid

**Stored dimension** / **Native dimension**:
The width a vector is *kept* at, and the width the model *produces*. paperscale asks for native
vectors and truncates them itself, so every vector exists at both widths at different moments and
confusing the two is the easiest error in this domain to make silently.
_Avoid_: dim, size, embedding size

**Instruction**:
The text an asymmetric embedding model expects in front of its input, differing between a
**query** and a document. Some models instruct the query only (Qwen3-Embedding); others require
a prefix on both sides (Nemotron-3-Embed wants `passage: ` on documents and `query: ` on
queries). paperscale applies the document side and records which convention it used, so a
consumer can match it on the query side.
_Avoid_: prompt, prefix, system message

**Sink**:
A destination that written vectors land in. An **Invocation** may enable more than one, and a
**Document** is not done until every enabled Sink holds it — which is why enabling a second Sink
re-embeds a corpus the first one already covered.
_Avoid_: output, store, target, writer

**Resume**:
Skipping documents a previous **Invocation** already embedded, so an interrupted one continues
rather than restarts.
_Avoid_: incremental, checkpoint, caching

**Consumer**:
The separate downstream project that reads paperscale's vectors. It owns retrieval, ranking
and quality measurement; paperscale owns none of them.
_Avoid_: client, user, downstream system
