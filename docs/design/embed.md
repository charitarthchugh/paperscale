# `paperscale embed` — design and implementation

**Status:** locked design, not yet implemented. This is the destination of
[#21, *Map: a from-scratch design for paperscale embed*](https://github.com/charitarthchugh/paperscale/issues/21),
synthesized from its twenty closed decision tickets. Nothing on that map remains open.

**How to read this.** It is written for someone who has never seen the map. Every section states
what was decided and why, and attributes it to the ticket that decided it (`#N`). Where a ticket's
resolution was later amended, this document carries the **amended** rule and says so; the map's
record contains superseded text that a reader arriving via a search engine will otherwise trust.
Section [17](#17-what-the-record-does-not-settle) collects the few things the record genuinely
leaves open, so an implementer probes rather than assumes. Section
[21](#21-appendix-the-design-in-simplified-technical-english) says the whole design again in
ASD-STE100 Simplified Technical English, for a reader who wants it in short sentences first.

**Vocabulary.** This document uses [`CONTEXT.md`](../../CONTEXT.md)'s terms exactly — **Document**,
**Record**, **Run**, **Invocation**, **Document name**, **Adapter**, **Chunk**, **Overflow**,
**Chunk vector**, **Document vector**, **Instruction**, **Sink**, **Resume**, **Consumer**. Where a
sentence would otherwise reach for "file", "row", "job", "doc_id", "backend", "passage",
"embedding", "prompt", "store", "checkpoint" or "downstream system", the glossary term is used
instead. `CONTEXT.md` is the authority on the language; this document is the authority on the
design.

---

## Table of contents

1. [What `embed` does, and the two Consumers it serves](#1-what-embed-does-and-the-two-consumers-it-serves)
2. [The pinned models and the one supported engine](#2-the-pinned-models-and-the-one-supported-engine)
3. [The Adapter seam](#3-the-adapter-seam)
4. [The context-length rule and the Chunk budget as numbers](#4-the-context-length-rule-and-the-chunk-budget-as-numbers)
5. [Chunking](#5-chunking)
6. [Dimensions and pooling](#6-dimensions-and-pooling)
7. [Identity: the Document name](#7-identity-the-document-name)
8. [The `.npz` Sink](#8-the-npz-sink)
9. [The LanceDB Sink](#9-the-lancedb-sink)
10. [Provenance: every recorded fact and where it lives](#10-provenance-every-recorded-fact-and-where-it-lives)
11. [Resume](#11-resume)
12. [The pipeline, end to end](#12-the-pipeline-end-to-end)
13. [The TUI panel](#13-the-tui-panel)
14. [The CLI surface](#14-the-cli-surface)
15. [Packaging](#15-packaging)
16. [Two probed claims](#16-two-probed-claims)
17. [What the record does not settle](#17-what-the-record-does-not-settle)
18. [Implementation plan](#18-implementation-plan)
19. [Documentation obligations](#19-documentation-obligations)
20. [Appendix: decision index](#20-appendix-decision-index)
21. [Appendix: the design in Simplified Technical English](#21-appendix-the-design-in-simplified-technical-english)

---

## 1. What `embed` does, and the two Consumers it serves

### 1.1 Scope

`paperscale embed` reads the `results/*.jsonl` that an OCR Run produced, turns each Document's text
into vectors against an external, OpenAI-compatible `/v1/embeddings` server, and writes those
vectors into one or both Sinks. paperscale's job ends when the vectors exist and are identified.
Retrieval, reranking, RAG and embedding-quality evaluation all belong to the Consumer.

One execution of `embed` is an **Invocation**. An Invocation uses exactly one embedding model and
may read more than one Run, because `--run LABEL=PATH` is repeatable (#26, which added *Invocation*
to `CONTEXT.md`). That distinction is load-bearing throughout: the model facts are Invocation-scoped
and therefore live in one manifest or one block of table metadata, while the run label varies per
Document and therefore lives on the Document.

`embed` never sees a directory of PDFs. It reads JSONL, and the PDF tree root is recorded nowhere
(#22). Every identity question is therefore answered from the `Source-File` string inside a Record.

### 1.2 Persisting the vectors is a deliberate premise

It is worth saying out loud, because it is not universal. Of the fourteen comparable pipelines read
at pinned commits for #36, `nomic-embed` keeps its vectors in RAM and a FAISS index, consumes them
for filtering, and discards them — its only artifact is a JSON id list. paperscale takes the
opposite position on purpose: **the vectors are the deliverable**, they outlive the Invocation that
made them, and a separate project reads them later without access to this repository. Every format
decision below follows from that premise, especially the insistence that a Sink be interpretable by
a stranger with a numpy import.

### 1.3 The uniform output shape — and the two Consumers

Every Document emits **Chunk vectors plus one pooled Document vector**, and **both Sinks carry
both** (standing decision 3).

paperscale serves two Consumers, and they read different artifacts. **Neither is primary; neither is
a convenience over the other.**

| Consumer | Reads | Why |
|---|---|---|
| **RAG** — retrieve text, hand it to a language model | **Chunk vectors** | Ranking is max-over-Chunks, which is query-*dependent*. A stored Document vector cannot be, so it is genuinely the weaker instrument here — at most a coarse-to-fine filter. |
| **Classification** — vectors as features for a separate model predicting a per-Document label | **Document vectors** | `sklearn` / `xgboost` / torch want `X` of shape `(n_documents, stored_dim)` against one label per Document. That *is* the Document vector array. There is no query, so the reduction cannot be deferred to read time — it has to happen somewhere, and the only question was where. |

**The divergence from practice is a different job, not a trade paperscale lost.** #36 found that zero
of fourteen comparable pipelines produce a Document vector as a first-class output. All fourteen are
*retrieval* pipelines that end at a search index — exactly the job that never needs one. The
unanimity was an artifact of a homogeneous sample. An earlier note on #36 instructed this document to
state that "Chunk vectors are the primary artifact and the Document vector is a convenience over
them"; **that instruction was retracted in full** and must not be written. #36's research file is
checked into the repository, so this paragraph is also what stops a future reader deleting the
Document vector on its authority.

The common case is `n_chunks == 1`, where the Document vector is a **bit-exact copy** of the sole
Chunk vector (#25). That duplication — 46 of 49 Documents in the smoke corpus — is the accepted price
of a Sink that is complete on its own.

### 1.4 Which Sink to reach for

Both Sinks carry both artifacts. Splitting them (Chunk vectors to LanceDB for RAG, Document vectors
to `.npz` for classification) was considered and **rejected**: it would put coarse-to-fine RAG across
a cross-Sink join, leave #27's `documents` table without a reason to exist (it holds the provenance
`chunks` deliberately does not repeat), spend the property #34 and #25 earned across two tickets —
that **one** Sink alone suffices to recompute and confirm the Document vector — and make both Sinks
mandatory for a Consumer doing both jobs, when `CONTEXT.md` says an Invocation *may* enable more than
one Sink.

The guidance is therefore about the *shape of the store*, not about the job:

| Sink | What it is for |
|---|---|
| **LanceDB** | The working store, for **both** jobs. Vector search and filters for RAG; one columnar scan of `documents` builds the classification feature matrix; both vectors in one place, so coarse-to-fine works. |
| **`.npz`** | Portability and archive. No database, plain files, numpy the only thing needed to read it. |

**Measured, because it is the one place the guidance bites.** Building `X` from the `.npz` Sink means
opening one ZIP per Document under the mirrored-tree layout. Warm cache, single-Chunk layout, all
eight arrays present:

```
2000 Documents, 768-dim
  per-Document .npz :  206.9 ms   (103 us/Document)
  single .npy matrix:    1.7 ms
  ratio: 124x
  extrapolated   100,000 Documents:  10.3 s  vs 0.08 s
  extrapolated 1,000,000 Documents: 103.4 s  vs 0.84 s
```

Real, but not disqualifying — seconds to a couple of minutes, once, and cacheable by the Consumer.
An Invocation-level `document_vectors.npy` roll-up was **rejected** on that number: it is an
aggregate written once at the end, so it fights #26's per-Document atomicity and #28's Resume story
— a resumed Invocation would have to rebuild it by reading every `.npz` anyway.

For the classification Consumer, so that #25 does not read as a constraint: the token-weighted mean
is a **default feature**, not the only one available. A Consumer wanting first-Chunk, max-pool, or
`mean ⊕ max` concat builds any of them from the Chunk vectors, and `n_chunks` and `token_count` are
themselves usable features.

### 1.5 Raw vector norms — considered and declined

Every stored vector is L2-normalized, so magnitude is destroyed, and #25 confirmed the raw norms are
neither stored nor recoverable. Reclaiming them is cheap in principle: request `use_activation:
false`, record `‖raw‖` as one `float32` per Chunk, then slice-and-normalize client-side, which
[§6.1](#61-mrl-slicing--client-side-on-by-default)'s algebra already proves equivalent.

**Declined.** For these models the raw norm tracks token count and frequency far more than meaning;
length is already available via `token_count`; and it would put a new server-side precondition on the
request path to buy a feature no Consumer has asked for. Recorded as *declined*, not *unexamined* —
a classification Consumer is exactly who would wonder.

### 1.6 Out of scope

- **Query-side embedding.** paperscale never embeds a query. It records the query-side Instruction
  convention instead, so the Consumer can construct matching queries (standing decision 9, #37).
- **Retrieval, reranking, RAG.**
- **Embedding-quality evaluation.** Measured downstream, by the Consumer. It is a second
  destination and cannot be designed against a format that was not yet locked.
- **Qdrant, or any second vector store.** LanceDB only.
- **safetensors** as an output format — replaced by `.npz`.
- **SGLang, TEI, Ollama, NIM** as serving engines — see [§2.2](#22-vllm-is-the-only-supported-engine).
- **Fixing the markdown export's extension collision** ([#32](https://github.com/charitarthchugh/paperscale/issues/32)).
  It is a pre-existing bug in the OCR-side export on `main`. #41 decided to leave it alone; the
  resulting divergence between `embed`'s name derivation and the export's is filed as
  [#42](https://github.com/charitarthchugh/paperscale/issues/42).

---

## 2. The pinned models and the one supported engine

### 2.1 Two pinned model families

**Qwen3-Embedding** and **Nemotron-3-Embed** (standing decision 2). Both document a 32,768-token
context.

"Long context" therefore means 32K, **not** "the Document always fits". At ~3.6 chars/token that is
roughly 45 dense pages. **Overflow is the minority path**, and this sentence is what a reader uses to
judge how much the chunking machinery is worth, so it is stated with its measurement: against the
49-Document smoke corpus, Overflow is **3 of 49 (~6%)**, stable across chars/token ratios of 3.0–4.0.
The median Document is 8,880 characters, then there is a cliff to 121k / 171k / 218k. The tail is
real and large — 89 pages at the top — which is exactly what makes #24's greedy packer and #25's
token weighting worth their complexity for a path that fires one time in sixteen.

*(The map's standing decision 2 originally asserted Overflow was "routine, not exceptional". The only
measurement available contradicts that, on a small sample, and the text was amended.)*

### 2.2 vLLM is the only supported engine

Standing decision 10, narrowed from "vLLM and SGLang" after #33.

**Why the others are out.** #23 inventoried vLLM, TEI, Ollama and NIM against every fact `embed`
needs. **TEI and Ollama truncate silently** on oversized input — TEI since v1.9.0, Ollama
uncontrollably — and their OpenAI-compatible `/v1/embeddings` route cannot be made truncation-safe.
Silent truncation is disqualifying here in a way it is not elsewhere: stored character offsets would
describe text that was never embedded, and nothing downstream could detect it. **NIM** cannot report
max input tokens over HTTP at all.

**Why SGLang is out, despite being the better engine for this workload.** #33 found SGLang errors on
oversized input by default, and — uniquely among the five engines surveyed — the setting is *askable*
via `/server_info`, so truncation safety could have been a startup assertion rather than an
assumption. Its `GET /v1/loads` needs no flag and carries `total_prefill_uncached_tokens` and
`total_prefill_busy_us`, monotonic counters incremented on embedding batches, which yield GPU-busy
prefill tokens/sec and a saturation ratio *directly*. vLLM has no equivalent. It was dropped anyway
because **SGLang does not serve Nemotron-3-Embed** (its `Ministral3Model` architecture is
unregistered, SGLang has no `…Model` → `…ForCausalLM` normalisation, and the Transformers fallback's
pooling ternary cannot produce the MEAN pooling Nemotron needs). Keeping both pinned model families
was worth more than keeping both engines.

**What that costs, knowingly.** There is no GPU-busy-time counter on vLLM, so the panel's throughput
row is derived from `vllm:prompt_tokens_total` over wall-clock. What is lost is the busy-time
*ratio*, not the numbers. vLLM's `/metrics` is mounted unconditionally and
`src/paperscale/vllm_stats.py` already parses it, so the panel is fed either way.

**#33's engine-detection scheme and error-body branching table must not be built.** They describe a
two-engine client for an engine paperscale no longer targets. #33 stays on record in case the trade
is ever revisited.

---

## 3. The Adapter seam

### 3.1 The rule

**Ask the server for everything it can be *trusted* on; the Adapter carries only what the server
cannot be trusted for** (standing decision 4, as revised by #23 and #37).

The original formulation — "hardcode nothing but the model id" — is not implementable. #23's
four-engine inventory established that three facts are irreducible.

### 3.2 The residue — exactly three facts

1. **MRL validity, as a *range* `[min_dim, native_dim]` — never a list.** Neither pinned family
   publishes an enumeration. Qwen3's cards state a *continuous* range, "32 to N", where N is the
   model's own native dimension (1024 / 2560 / 4096); 2560 also rules out any "valid points are
   powers of two" shortcut. Nemotron's cards describe slicing "for example, keeping the first 1024 or
   512 dimensions", which are examples of a range, not an enumeration. `native_dim` doubles as the
   assertion that the server has the right model loaded (#34).
2. **The Instruction convention and its literal strings, for both query and document sides** (#23,
   #37). See [§3.5](#35-the-instruction-is-two-plain-strings).
3. **The model's card context length.** Undiscoverable, and every engine over-reports it: a default
   `vllm serve` advertises `max_model_len` 262,144 for Nemotron and 40,960 for Qwen3-8B against a
   documented 32,768. Asking the server returns a number wrong **in the unsafe direction**. #37
   restated this as a *rule* rather than a constant — see [§4](#4-the-context-length-rule-and-the-chunk-budget-as-numbers).

**The output dimension is a *probe*, never an *ask*** — one cheap request — on every engine surveyed
(#23). TEI's [#148](https://github.com/huggingface/text-embeddings-inference/issues/148) is still
true; it was closed COMPLETED by a maintainer comment reframing the question, not by a fix.

**Everything else is asked of the server and recorded.** Served model id comes from `/v1/models`
(`.id` plus `.root`); exact token counts come from `POST /tokenize`.

### 3.3 The Adapter's complete contract

```python
class EmbedModel(abc.ABC):
    card_context_length: int      # the card's number; min()'d against the server at run time
    native_dim: int               # published per model; doubles as the wrong-model assertion (#34)
    min_dim: int = 32             # Qwen3's published floor, adopted as the convention
    document_instruction: str     # applied by paperscale
    query_instruction: str        # recorded for the Consumer; never applied
```

Five typed class attributes with defaults, matching the shape of `default_model_name` and
`preferred_longest_image_dim` on `src/paperscale/models/base.py`'s `OCRModel`. There are no methods:
unlike an OCR Adapter, an embedding Adapter has no prompt to build and no response to parse.

**One Adapter per model *size*, not per family.** `native_dim` differs per size, so a family-level
Adapter cannot hold one width. Siblings are short subclasses.

| Registry key | `native_dim` | `card_context_length` | `document_instruction` | `query_instruction` |
|---|---|---|---|---|
| `qwen3-embedding-0.6b` | 1024 | 32768 | `""` | `"Instruct: {task_description}\nQuery:{query}"` |
| `qwen3-embedding-4b` | 2560 | 32768 | `""` | *(as above)* |
| `qwen3-embedding-8b` | 4096 | 32768 | `""` | *(as above)* |
| `nemotron-3-embed-1b` | 2048 | 32768 | `"passage: "` | `"query: "` |
| `nemotron-3-embed-8b` | 4096 | 32768 | `"passage: "` | `"query: "` |

Keys are lowercase-hyphenated, matching `MODEL_REGISTRY` in `src/paperscale/models/__init__.py`.
`EMBED_MODEL_REGISTRY` and `build_embed_model(name)` mirror `MODEL_REGISTRY` and `build_ocr_model`,
so an engineer who knows one knows the other.

**There is no `DEFAULT_EMBED_MODEL`.** #35 made `--embed-model` required where `--ocr-model` is not,
so the registry deliberately has no default entry. This is the one place the mirror is not exact.

### 3.4 The Adapter applies the document-side Instruction, and it is recorded

Standing decision 9, revised after #23. The map originally generalized from Qwen3 and assumed the
convention was query-side only. It is not: **Nemotron-3-Embed requires `passage: ` on documents** and
`query: ` on queries. Both families emit L2-normalized vectors.

Applying the document side is document-side work and always was in scope; only the query side stays
out. **Recording it is therefore load-bearing rather than defensive** — a Consumer that does not know
Documents were prefixed will build mismatched queries.

Sharp edge worth carrying even though the engine is now vLLM-only: TEI accepts `--default-prompt` and
`/info` does not report it, so a server can silently double-prefix. The general lesson survives the
engine narrowing — a serving flag can change what the model sees without appearing in any response.

### 3.5 The Instruction is two plain strings

`instruction` was one provenance fact until #37 split it. It is now **two**, both invariant within an
Invocation, and **both Sinks must carry both**, since a Consumer matching query-side conventions reads
them from the Sink rather than from a model card.

The two families differ in *shape*, not only in content. Verified character by character from Qwen's
own helper:

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
```

Space after `Instruct:`, **no space** after `Query:`, and *"No need to add instruction for retrieval
documents"*. So Nemotron's query side is a fixed string and Qwen3's is a template with a
caller-supplied slot; the presence of a `{task_description}` placeholder means the Consumer supplies
its own, and its absence means use the string as-is.

**Two facts to write down as facts, not as background:**

- **`document_instruction` is `""` for Qwen3 — empty string, never null.** `""` says *we decided on
  none*; a missing value says *we did not record it*. Provenance has to keep those apart, and a
  reader of the schema will otherwise "tidy" one into the other.
- **Omitting Qwen3's query-side Instruction costs approximately 1–5% retrieval performance**,
  published by Qwen. paperscale never applies it, but the Consumer decides whether to, and cannot
  decide without the number.

### 3.6 The wrong-model assertion

paperscale compares the observed width of the server's response against `adapter.native_dim` and
**stops the Invocation** on mismatch, reporting both numbers (#34).

This is not a nicety. Point `--embed-model` at a server with a different size of the same family
loaded and, with every vector cut to 768, the stored data carries no trace of the substitution — the
Sink looks perfect and is wrong. Standing decision 7 guarantees Resume will not catch it either,
since Resume asks only "do I know this Document name?". Because MRL slicing is client-side, every
response arrives at native width, so the check costs one length comparison per response rather than a
special probe.

It does **not** cover two models of *equal* width, which is why #38 requires the served model id from
`/v1/models` to be asked before the reporter is built and shown in the panel header
([§13.3](#133-model-moves-to-the-header)).

---

## 4. The context-length rule and the Chunk budget as numbers

### 4.1 There are four numbers on the table, not two

Standing decision 4.3 framed this as *server-advertised* versus *documented*. #37 found a third
number on both cards that neither the map nor the ticket anticipated, and a later check of the
checkpoints themselves found a fourth on Nemotron:

| | Qwen3-Embedding-8B | Nemotron-3-Embed-8B |
|---|---|---|
| vLLM advertises on a default serve | 40,960 | 262,144 |
| `config.json` `max_position_embeddings` | 40,960 | 262,144 |
| Card states | "Context Length: 32k" | "The model's max sequence length is 32768." |
| Card **exercises** | `max_length = 8192` in its own examples | "We set the model sequence length to **4096** for the evaluation results below." |
| RoPE base before scaling | — (`rope_scaling: null`) | `original_max_position_embeddings` **16,384** |

**The fourth number explains the first.** Nemotron's `rope_parameters` are
`{"rope_type": "yarn", "factor": 16.0, "original_max_position_embeddings": 16384}`, and
16,384 x 16 = **262,144** exactly. vLLM's advertisement is not arbitrary and is not a bug: it is the
YaRN-scaled base, computed from the checkpoint. Qwen3 carries `rope_scaling: null` and simply
advertises its `max_position_embeddings` of 40,960, which is why only one of the two families shows a
wild number.

**`apply_yarn_scaling: False` is a trap, and the card says so.** Read plainly it means long context is
switched off, which would put Nemotron's real window at 16,384 and halve its Chunk budget. It does not
mean that. NVIDIA's card states: *"`apply_yarn_scaling` is retained as a temporary vLLM compatibility
field that preserves the checkpoint's intended long-context RoPE behavior. **Do not remove it from
`config.json`.**"* It emits a load-time warning and points at
[vLLM issue #48621](https://github.com/vllm-project/vllm/issues/48621). **Anyone who "fixes" that
warning by deleting the field changes the model's positional behaviour.** The card's 32,768 remains
the authoritative number and [§4.2](#42-the-rule)'s arithmetic is unchanged.

Nemotron's sentence is not a limit — it is the provenance of the benchmark numbers. Every published
score for that model was produced at 4096; above that, quality is not degraded so much as
**unmeasured by the vendor**. That is a different kind of unknown from "the server is lying about its
window", and the map's single word *validated* collapsed the two.

### 4.2 The rule

```
default:   validated_context_length = min(adapter.card_context_length, server_max_model_len)

--context-length N:
    N > server_max_model_len   ->  REJECTED at startup
    N > card_context_length    ->  allowed, with an explicit warning naming both numbers
    otherwise                  ->  allowed silently
```

**The card is authoritative about the model; the server is authoritative about the deployment;
neither is authoritative about both.** `min()` takes the safe half of each. It keeps standing
decision 4.3's protection against a 262,144-token advertisement, and adds protection the map did not
have: an operator who launches with `--max-model-len 8192` against a card that says 32,768 would
otherwise have every long Document hard-fail. For both pinned models on a default serve the rule
yields **32,768**; the value of the rule is that it stays correct when the deployment changes.

This sharpens standing decision 4's governing rule rather than contradicting it: **the server can be
trusted as an *upper* bound and not as a *lower* one.**

**Only `server_max_model_len` is a correctness boundary.** Above it the request cannot succeed, so
rejecting at startup beats failing per Document. The card's number is a *quality claim*, and the
evidence behind it is thinner than the card implies. Refusing an operator that trade would be
paperscale enforcing a quality opinion under cover of a safety check, and this map assigns
embedding-quality judgement to the Consumer.

**The warning above the card must name both numbers and the actual risk** — not "may reduce quality",
which understates it by implying a measurement exists. Above the card, quality is *unmeasured by the
vendor*. The warning is about an absence of evidence and should say so.

**#35's "must not be operator-settable" is overturned, not reconciled.** Its stated reason — that
asking the *server* yields a number wrong in the unsafe direction — remains true, and is exactly why
the default is `min()` rather than the server's figure. But that argument never supported forbidding
an operator override; it supported not *deriving* the value from the server alone. The two were
conflated. `--context-length` is not the `--max-context` #35 rejected: that one would have taken the
server's number as authoritative, which is still the wrong design. This one can only name a number the
server has already agreed to serve.

**Nothing extra is recorded in the Sink.** Whether the override was used above the card is fully
derivable from facts already in provenance — `chunk_budget_tokens` against the `card_context_length`
that `model_id` identifies. #26 set the principle when it dropped `chunk_index` and `n_chunks`:
*storing a derived value invites the two copies to disagree.*

**The card-exercised numbers (8192, 4096) are recorded and deliberately not used.** Choosing them
would be paperscale making a quality call it cannot measure, and it would land hardest on the
classification Consumer specifically: at 4096 the smoke corpus goes from 52 Chunks to 97, so the
Document vector becomes a mean over roughly two Chunks instead of one, adding pooling dilution to the
feature. The override flag is what makes this reversible the moment the Consumer has evidence.

Measured cost of each choice, against the 49-Document smoke corpus:

```
budget(tok)  chars@3.6   Overflow   pct   max Chunks   total Chunks / 49 Documents
      31900     114840          3    6%            2                       52
       8192      29491          6   12%            8                       68
       4096      14746         14   29%           15                       97
```

### 4.3 `SAFETY_MARGIN = 64`, and the reason changed

The implied 868 (32,768 − 31,900, from #26's manifest *example*) is discarded. It was never stated as
a decision, and 2.6% of the context is a lot to spend on nothing recorded.

**The margin does not defend against packing.** #24's subadditivity proof already covers packing, and
covers the Instruction too, since `tokens("passage: ") + tokens(text) >= tokens("passage: " + text)`
by the same argument. What remains is a **route-level** risk: paperscale counts tokens on
`/tokenize` and sends text to `/v1/embeddings`. If those two apply special tokens differently by
even one token, a Chunk sitting exactly on the budget hard-fails — because the design deliberately
relies on vLLM *erroring* on overflow rather than truncating. 64 tokens is 0.2% of context against a
visible hard failure.

### 4.4 The Chunk budget, evaluated

```
chunk_budget = validated_context_length - tokens(document_instruction) - 64
```

| Model | `validated_context_length` (default serve) | `tokens(document_instruction)` | `chunk_budget` |
|---|---|---|---|
| Qwen3-Embedding (any size) | 32,768 | 0 (`""`) | **32,704** |
| Nemotron-3-Embed (any size) | 32,768 | ~3 (`"passage: "`) | **~32,701** |

`chunk_budget_tokens` is computed at startup and recorded in provenance. **#26's manifest example
showing `31900` is stale** and should not be copied into an implementation.

---

## 5. Chunking

**Greedy page packing, no overlap, token counts asked of the server, boundaries recorded as character
offsets *and* page spans** (#24).

### 5.1 The two facts that shaped it

1. **The page spans are already exact character ranges.** `build_dolma_document` in
   `src/paperscale/pipeline.py` builds `attributes.pdf_page_numbers` as `[start_char, end_char,
   page_num]` triples that **tile** the text: `span[i].end == span[i+1].start`, no gaps. So
   page-respecting chunking costs no offset math — a run of consecutive pages *is* a character range.
   Two consequences to know before writing the code:
   - the `\n` joiner between pages is folded into the **preceding** page's span, so slicing
     `text[start:end]` for a page includes its trailing newline (all but the last page);
   - a page whose `natural_text` is `None` emits a **zero-width** span. Empty pages are real entries
     that cost 0 tokens and can never force a Chunk break.
2. **paperscale cannot count tokens today.** There is no `transformers`, `tokenizers` or `tiktoken`
   in `pyproject.toml`. Counting tokens is a *new capability*, not a call to something that exists.

### 5.2 Token counting — ask the server

`POST /tokenize` returns the exact count for the model actually loaded. **The route carries no
`/v1`**: vLLM mounts `tokenize` and `detokenize` at the *top level*, and only the OpenAI-compatible
routes sit under `/v1`. This document said `/v1/tokenize` in 23 places, which 404s on every build
and so failed every Document until a live run caught it — see
[§17.3](#173-stale-text-still-standing-in-the-record). No new dependency, no
second tokenizer to keep in sync with the server's, and it follows the precedent already set in
`src/paperscale/evaluation/pplx.py` of reading token facts off the server rather than reimplementing
them. Two properties make it affordable: the route is handled CPU-side in the API server process and
**never enters the engine scheduler**, and the common case costs exactly **one** call per Document.

This turned out to be the least conventional decision that was right. #36 found that **nobody counts
tokens with the right tokenizer**: LlamaIndex defaults to `cl100k_base` *regardless of the configured
embedding model*, and Vespa's chunker counts **characters** (1000 by default, with a "counting
codepoints… too expensive" comment) while feeding a 512-**token** embedder.

### 5.3 The algorithm

```
budget = chunk_budget                                # §4.4
spans  = record["attributes"]["pdf_page_numbers"]    # [start_char, end_char, page_num], tiling

total = tokenize(record["text"])                     # 1 call
if total <= budget:                                  # the common case
    -> single Chunk spanning the whole Document, done

# Overflow path only
counts = [tokenize(text[s:e]) for s, e, _ in spans]  # N calls, on rare Documents
for each page i:
    if counts[i] > budget:                -> oversized-page path (cut at last \n, else hard cut)
    elif running + counts[i] > budget:    -> close Chunk, start a new one at page i
    else:                                 -> append page i to the current Chunk
```

**Why assembled Chunks need no re-verification.** Splitting a string can only *increase* its BPE token
count, because a merge cannot span a split boundary, so `sum(tokenize(page_i)) >=
tokenize(concat(page_i))`. Packing by summed per-page counts is conservative in the safe direction: a
Chunk built this way can never exceed the budget when tokenized whole. This is what keeps the Overflow
path at N calls instead of N plus one re-check per assembled Chunk.

**A free calibration.** The single Document-level call yields both `total` and `len(text)`, so every
Document hands over its own chars-per-token ratio at zero extra cost. That ratio is what the
oversized-page path uses to estimate where the budget falls before searching backwards for a newline —
no corpus-wide constant, no second request.

### 5.4 The decisions inside it

**Cut rule — greedy page packing.** Fill a Chunk with whole pages until the next page would overflow;
then close it. Every Chunk stays citable to real pages and still uses most of the pinned 32K context.
Pure page-per-Chunk was rejected for wasting the context; pure token windows for throwing away the
page citation the JSONL hands over for free.

**Overlap — none.** Chunks tile the Document exactly once. Overlap would double-count the overlapping
spans in the pooled Document vector and make the stored offsets stop describing a partition, so
reconstruction from offsets would no longer round-trip. Overlap is a *retrieval-side* tactic;
re-chunking for retrieval belongs to the Consumer, which has the exact offsets and can do it without
help.

**A single page that alone exceeds the budget.** Cut at the last `\n` at or before the budget; if the
page contains no newline at all, take a hard character cut. **Never drop text, never fail the
Document.** This is the only path that produces a Chunk boundary inside a page, and it is why
`is_partial_page` exists as a recorded field.

**Recorded per Chunk — both coordinate systems.** `[start_char, end_char]` is the primitive: it is the
only thing that can express a partial page, and it is what makes a Chunk reconstructible by slicing
`record["text"]`. `[first_page, last_page]` is derived from it and is what a citation shows a human.
Storing only pages loses the partial-page case; storing only offsets makes every Consumer re-derive
pages from `pdf_page_numbers`. The recorded fields are `start_char`, `end_char`, `first_page`,
`last_page`, `chunk_index`, `n_chunks`, `token_count`, `is_partial_page` — of which `chunk_index` and
`n_chunks` are later dropped from the `.npz` as derived ([§8.4](#the-decisions-under-that-table)).

**Rejected: deriving offsets from the tokenizer's output.** `/tokenize` can return
`return_token_strs`, and it is tempting to reconstruct character positions from them.
`pplx.py:201` already carries the warning: decoded tokens carry marker glyphs (SentencePiece `▁`, BPE
`Ġ`) whose handling differs per tokenizer, so character attribution built on them breaks silently
across models. Slicing by the character offsets already in hand is exact and model-independent.

### 5.5 Zero-Chunk Documents

A Document with no usable text produces `n_chunks == 0` and is recorded as an **empty output**, not a
flag and not a failure (#28). The chunker must be able to produce it. See
[§11.4](#114-documents-with-no-usable-text).

---

## 6. Dimensions and pooling

### 6.1 MRL slicing — client-side, on by default

**`--embed-dim`, default 768, applied client-side** (#34). 768 is inside the valid range of every
pinned model at every size (Qwen3 1024 / 2560 / 4096, Nemotron 2048 / 4096).

paperscale requests **native** vectors and slices them itself; it never sends `dimensions`. This was
decided on operational grounds only, because the two routes are **mathematically identical**:

```
server-side:  normalize(slice(raw))
client-side:  normalize(slice(normalize(raw)))       # the server normalized at native width

slice(raw/‖raw‖) / ‖slice(raw/‖raw‖)‖
  = (slice(raw)/‖raw‖) / (‖slice(raw)‖/‖raw‖)        # a positive scalar cancels
  = slice(raw)/‖slice(raw)‖
  = normalize(slice(raw))
```

Slicing is coordinate selection, hence linear; normalizing divides by a positive scalar; the scalar
cancels. Both routes yield the same unit vector to float rounding.

What remains is a trade between ~5.3× larger responses (4096 → 768) and a launch flag paperscale
cannot set, cannot verify, and that fails at run time when forgotten: **`dimensions` is a real
per-request parameter on vLLM and neither pinned family can use it on a default launch.** All four
`config.json` files declare neither `is_matryoshka` nor `matryoshka_dimensions`, so
`PoolingParams._set_default_parameters` raises before inference — *"Model 'X' does not support
Matryoshka embeddings; dimensions must be unset"*. Server-side truncation would require launching
with `--hf-overrides '{"is_matryoshka": true}'`. On a LAN server the bandwidth is cheap and the
client-side slice is a few lines of numpy, so the zero-precondition route wins. (The response-size
cost is separately bought back on the wire by `encoding_format: base64` — [§12.5](#125-the-wire-format).)

Also worth knowing: vLLM's accepted range is `[1, embedding_size]` — **its floor is 1**. Qwen3's
published floor of 32 is a model-quality claim the server will never enforce, which is why `min_dim`
is an Adapter constant.

**Validation.** paperscale rejects `--embed-dim` outside `[min_dim, native_dim]`. A value above
`native_dim` is **rejected outright rather than clamped** — a user asking a 2048-dim model for 4096
has a wrong mental model, and silently handing back 2048 hides it.

**Order of operations: slice each Chunk vector, re-normalize, then pool.** The alternative (pool
native, then slice) was rejected because the Document vector would then depend on dimensions that are
never written. Under the chosen order, **a Consumer holding one Sink can recompute the Document vector
from the stored Chunk vectors and get the same answer.** The slice itself commutes with averaging;
only the per-Chunk re-normalization makes the two orders differ at all.

`stored_dim` and `native_dim` are both recorded. Once a vector is cut to 768, `native_dim` is the only
surviving evidence of which model served the Invocation — an 8B and a 4B of the same family produce
indistinguishable output otherwise. Explicit `truncated` and `normalized` fields were rejected:
`truncated` is exactly `stored_dim != native_dim`, and normalization is unconditional, so a
constant-true field records nothing.

### 6.2 Pooling — token-weighted mean

```
chunk_vector[i] = normalize(slice(server_vector[i], stored_dim))   # §6.1

if n_chunks == 1:
    document_vector = copy(chunk_vector[0])            # exact; no arithmetic runs
else:
    w = [chunk.token_count for chunk in chunks]
    document_vector = normalize(sum(w[i] * chunk_vector[i] for i in range(n_chunks)))
```

Note the missing division. Dividing the weighted sum by `sum(w)` is a no-op, because the `normalize`
that follows discards any positive scale. Leaving it out removes a step and one source of float error.
(#36 found LangChain implements this exact algorithm — short-circuit included — hand-rolled in pure
Python after numpy was dropped as a dependency; it *does* divide by `total_weight` first, which is the
no-op #25 identified.)

**Why token-weighted.** The deciding property is **invariance to the cut**. The Chunk budget is an
implementation detail — it moves when the validated context length moves, and #24's oversized-page
path can move it again inside a single Document. A token-weighted mean stays approximately put across
those changes; a uniform mean swings hard. Greedy packing makes this concrete: a two-Chunk Document
can be 40 pages plus a 1-page tail, and a uniform mean hands that tail page half the Document vector.

Recorded as a knowing trade rather than a free win: because Chunk vectors are unit length, token
weighting asserts that text volume equals importance. In a legal Document a one-page cover letter may
be more discriminative than a forty-page exhibit. That is a claim about this corpus that cannot be
tested inside this effort, and "proportional to content" is the neutral default — it is also what a
single-shot embedding of the whole Document would approximate, which is the thing chunking stands in
for.

**Every Document gets a Document vector. No sentinel, no threshold.** The dilution concern is real —
averaging forty diverse Chunk vectors does drift toward the corpus centroid — but a sentinel needs a
Chunk-count threshold, and no measurement available in this effort could justify one. `CONTEXT.md`
assigns exactly this judgement to the Consumer, which has the Chunk vectors and has `n_chunks`
recorded so it can make the call without reading a single vector.

**The single-Chunk case is a copy, not an average.** It does not fall out of the arithmetic: a
weighted mean of one element is then re-normalized, and the divisor is `1 ± ε` rather than `1`, so the
low bits move. paperscale short-circuits and copies. **A test asserts bit-equality, not approximate
equality.**

**Edge cases, defined rather than left open.** If `sum(w) == 0`, fall back to uniform weights — this
is unreachable in practice (a Document with more than one Chunk overflowed, so it has tokens) but the
alternative is a division by zero in a path nobody tests. Separately, a weighted sum with zero norm
would require the Chunk vectors to cancel exactly; treat that as an error rather than writing `NaN`
into a Sink.

**Properties this guarantees.**

- **Reproducible from one Sink.** A Consumer holding `chunk_vector` and `token_count` recomputes the
  Document vector and gets the same answer. `token_count` is therefore a **weight, not a diagnostic**,
  and both Sinks must keep it readable next to the Chunk vectors.
- **Invariance to the cut is approximate, not exact.** A Chunk vector covering forty concatenated
  pages is not the mean of forty single-page vectors, so re-chunking the same Document at a different
  budget shifts the Document vector slightly. The claim is that it shifts far less than under uniform
  weighting.
- **The identity case is exact by construction**, not by tolerance.
- **Every stored vector is unit length.** #36 found two shipping counter-examples worth knowing:
  vLLM's own long-text embedding never re-normalizes its cross-chunk mean (so dot-product scoring
  silently breaks), and Vespa truncates MRL-style with `normalize` defaulting to `false`.

---

## 7. Identity: the Document name

One name does three jobs: the `.npz` filename, the LanceDB primary key, and the Resume key. A weak
answer fails in three places at once (#22).

### 7.1 The rule

Given a Record's `metadata["Source-File"]`:

1. **Tarball form** (`archive.tar.gz::internal/path.pdf`) — the tarball basename, stripped of `.tar`
   / `.tar.gz`, becomes a directory, and the internal path continues beneath it. This reuses the
   markdown export's existing handling.
2. **Otherwise** — `os.path.normpath(source)`, then `lstrip("/")`, then drop remaining `..` and empty
   components.
3. **The output filename appends the source extension; it does not replace it.**
   `case.pdf` → `case.pdf.npz`, sidecar `case.pdf.json`. `case.tiff` → `case.tiff.npz`.
4. **Run label in the path only when one Invocation embeds more than one Run:**
   - one `--run` → `<out>/<mirrored path>` (**`bare`** layout)
   - two or more → `<out>/<label>/<mirrored path>` (**`labelled`** layout)
5. **A digest of the raw `Source-File` string exists for every Document, always** —
   `sha256(source_file.encode()).hexdigest()[:16]`.

**The Document name itself** is the sanitized relative path *including* the source extension (e.g.
`run/media/cc/data/law/doc9419897.pdf`). The `.npz` Sink appends `.npz` and `.json` to it; the
LanceDB Sink stores it verbatim in `document_name`. One derivation, two Sinks.

**Rule 2's normalization is `embed`-specific and deliberate.** The markdown export's sanitizer at
`src/paperscale/pipeline.py:686-689` reads `safe_parts = [p for p in parts if p and p != ".."]`: it
**drops** `..` instead of resolving it, so `/a/b/../c.pdf` and `/a/b/c.pdf` derive the same name, and
it keeps `.`, so `/a/./b.pdf` derives `a/./b.pdf` while the file lands at `a/b.pdf`. Resume's
correctness rests entirely on name stability, so `embed` cannot inherit that. `normpath` resolves
interior `..` and `.` correctly and leaves *leading* `..` alone, so the traversal guard is still
required after it, not replaced by it. `realpath` is **not** used: the PDFs may be gone by embed time,
and resolving symlinks would change which Documents are considered the same rather than normalizing
how one is spelled.

**Rule 3 changed after #41.** It originally replaced the extension, to mirror `get_markdown_path`.
[#32](https://github.com/charitarthchugh/paperscale/issues/32) established that this is exactly where
that function is wrong — it silently overwrites when two sources differ only by extension — and #22's
stated reason for mirroring was consistency, not correctness. Consistency with a bug is not worth
keeping. The property that matters: **the name is a pure function of one Record's `Source-File`**. It
does not depend on iteration order and it does not depend on which other Documents are in the
Invocation, which is what keeps Resume stable.

### 7.2 Why the digest is over the path and never over the text

Every Record already carries `id = sha1(document_text).hexdigest()`
(`src/paperscale/pipeline.py:654`). It is tempting and it is wrong. That digest changes whenever the
corpus is re-OCR'd, so adopting it as identity would make a re-OCR look like an entirely new set of
Documents. Combined with standing decision 7 (no content-change detection), that converts "Resume
silently keeps stale vectors" into "Resume silently duplicates the whole corpus" — the opposite
failure, equally invisible. The digest is over the `Source-File` **string**, which is stable across
re-OCR. (#36 found olmOCR reached the same conclusion independently, digesting sorted path strings
with sha1 and never content.)

`sha256(...)[:16]` is 64 bits — far past what a corpus of this size needs — and 16 characters stays
usable as a filename when the no-usable-path fallback fires. **Both Sinks must compute it with the
same function**, or they disagree about one Document's identity, which is precisely the failure the
digest exists to prevent.

### 7.3 What the digest is for — two jobs, not three

1. It is written into every output as provenance, so a Document is identifiable even when its
   filename is not unique.
2. It is **the name itself** when no usable path exists — `Source-File` missing or empty, the
   sanitized path reducing to nothing, or a path component exceeding the filesystem limit (ext4 caps a
   component at 255 bytes, which long legal filenames do reach).

**It is not a collision tiebreak.** #22 gave it that third job and #41 removed it. #28 derives Resume
from the outputs, so a tiebroken name would have to be stable across Invocations, and neither
available scheme is: suffixing the loser depends on iteration order, and suffixing every member of a
colliding set silently reverts when the set changes. Both are silent costs.

### 7.4 Collisions are prevented, and the residue is fatal at startup

**Prevented:**

| class | example | prevented by |
|---|---|---|
| extension replacement | `case.pdf` + `case.tiff` | rule 3 (append) |
| `..` components | `/a/b/../c.pdf` + `/a/b/c.pdf` | rule 2 (normalization) |

**Fatal at startup:**

| class | example |
|---|---|
| leading slash / empty components | `/a/case.pdf` + `a/case.pdf` |
| tarball collapse | `x.tar.gz::doc.pdf` + `x.tar::doc.pdf` |
| two Records both with an empty `Source-File` | both derive the same digest name |

The Invocation stops before any GPU work, listing **every colliding group with both raw `Source-File`
values**. This is not in tension with #30's *"one bad PDF must never end a run"* — that rule governs
embedding failures mid-run; this is a startup check costing seconds, and the operator gets an
actionable message naming exactly what to fix.

**The check is scoped to one Run.** Both Sinks key on `(run_label, document_name)`, and the `labelled`
layout puts each Run in its own subtree, so two Runs cannot collide with each other. *(This scoping is
a clarification made here; the record implies it via the Sink keys but never states it.)*

**Duplicate raw `Source-File` within one Run is also fatal**, matching `DuplicateSourceFileError` in
`src/paperscale/evaluation/runs.py:45`, whose docstring reads *"Two records in one run share a
Source-File -- the join key is ambiguous."* That is a genuinely different event from a derived-name
collision — two Records that cannot be told apart at all, versus two distinguishable Records that
merely want the same filename. #22 cited it for the second and #41 corrected that: the contradiction
readers found on #22 was one correct sentence sitting next to one misapplied citation, not two
decisions in tension.

**Measured before deciding:** the live corpus is **39,905 files, all `.pdf`, with zero collisions of
any class**; the 49-Document smoke set derives 49 distinct names. The strict option costs nothing
today. It remains a real hazard because `--pdfs` accepts *"Local PDF/image paths"*, so a mixed corpus
produces the collision immediately.

### 7.5 Reserved names

`<out>/paperscale-embed.json` (the manifest) and `<out>/paperscale-embed-failures.txt` sit at `<out>/`
in **both** layouts. In the `labelled` layout nothing else lives at `<out>/`. In the `bare` layout
they can only collide with a source at the filesystem root literally named `paperscale-embed.<ext>`.
The `paperscale-embed` prefix is **reserved**, and a Document whose derived output lands on it is
handled as the same class of fatal collision rather than silently overwriting the manifest (#26).

### 7.6 The limitation normalization cannot close

`Source-File` is **not** normalized upstream: `_expand_pdf_inputs`
(`src/paperscale/pipeline.py:1093`) does `glob.glob(p)` or `[p]` and calls neither `abspath` nor
`realpath`. So the same PDF OCR'd as `docs/a.pdf` and as `/home/cc/docs/a.pdf` yields two different
names, decided by the caller's working directory. **Nothing records the OCR-time working directory,
so no post-hoc processing can equate them.**

It sounds worse than it is: Resume re-reads the *same* JSONL, so names are stable unless the corpus is
re-OCR'd from a different directory. Mirroring the layout of the markdown export was chosen over an
explicit `--source-root` flag because the two trees sitting side by side —
`vectors/home/cc/corpus/x.pdf.npz` beside `markdown/home/cc/corpus/x.md` — is worth more than papering
over an upstream inconsistency with a flag on every Invocation. **This belongs in the user docs.**

`source_file` is recorded **raw** in both Sinks. Normalization applies to the derived name only, so
the original string survives for anyone reconciling by hand.

---

## 8. The `.npz` Sink

### 8.1 Provenance splits by scope

`.npz` has no metadata header, and object arrays are **impossible, not merely unwise**: `np.savez`
accepts a Python dict, but reading it back raises `ValueError: Object arrays cannot be loaded when
allow_pickle=False`. Any metadata inside an `.npz` must be a real dtype, and asking a Consumer for
`allow_pickle=True` means asking it to execute whatever the file contains.

So provenance splits **by scope, not by convenience**: one Invocation manifest for the facts that
never vary, one sidecar per Document for the facts that do, and the `.npz` holds arrays only (#26).

```
<out>/paperscale-embed.json                      <- the Invocation manifest, written once
<out>/paperscale-embed-failures.txt              <- rewritten each Invocation, if any failed
<out>/run/media/cc/data/law/doc9419897.pdf.npz   <- 8 arrays, nothing else
<out>/run/media/cc/data/law/doc9419897.pdf.json  <- the Document sidecar
```

### 8.2 `paperscale-embed.json` — the Invocation manifest

**Nine invariant facts plus an append-only log plus the enabled-Sink set.**

```json
{
  "model_id": "nvidia/Nemotron-3-Embed-8B-BF16",
  "stored_dim": 768,
  "native_dim": 4096,
  "document_instruction": "passage: ",
  "query_instruction": "query: ",
  "pooling": "token_weighted_mean",
  "chunker": "greedy_page_pack",
  "chunk_budget_tokens": 32701,
  "layout": "bare",
  "sinks": ["npz"],
  "invocations": [
    {"created": "2026-08-18T09:00:00Z", "paperscale_version": "0.8.0"}
  ]
}
```

The invariant block grew twice: `layout` joined it in #28 (seven → eight), and #37 split `instruction`
into `document_instruction` and `query_instruction` (eight → nine). **`sinks` is deliberately outside
the invariant block** — the set is allowed to change (#35).

**A second Invocation compares the invariants and stops on disagreement**, reporting both values,
before any Document is written. This is the check the Adapter's `native_dim` assertion cannot make: a
run appended to a tree an earlier Invocation built with a *different model*. Standing decision 7
removed content detection, so Resume will not catch it — the manifest is the only thing that can.

On a match, a new `{created, paperscale_version}` entry is appended: about 60 bytes, and the only
record of how a tree built over several Invocations came to be. Overwriting the manifest was rejected
outright — it would leave the file describing vectors it did not describe.

**The manifest is the parent process's, never a worker's.** It is read and appended once, before
workers start. Concurrent appends would race and corrupt it.

**If `sinks` changed**, the Invocation says so before starting — *"`--lancedb` is new; this will
re-embed 47,000 Documents"* — rather than silently spending a day on it. This is the same instinct as
the model checks: the map's whole shape is *make the expensive silent thing loud*, because standing
decision 7 removed the mechanism that would otherwise notice.

### 8.3 `<name>.json` — the Document sidecar

The four facts that vary per Document:

```json
{
  "source_file": "/run/media/cc/data/law/pdfs/pdfDownload10/doc9419897.pdf",
  "source_digest": "9f2b1c4d8e0a3f57",
  "run_label": "nemotron-8b",
  "created": "2026-08-18T09:04:37Z"
}
```

### 8.4 `<name>.npz` — eight arrays, no metadata

| name | shape | dtype |
|---|---|---|
| `chunk_vectors` | `(n_chunks, stored_dim)` | `float32` |
| `document_vector` | `(stored_dim,)` | `float32` |
| `start_char` | `(n_chunks,)` | `int32` |
| `end_char` | `(n_chunks,)` | `int32` |
| `first_page` | `(n_chunks,)` | `int32` |
| `last_page` | `(n_chunks,)` | `int32` |
| `token_count` | `(n_chunks,)` | `int32` |
| `is_partial_page` | `(n_chunks,)` | `bool` |

#### The decisions under that table

**Page range per Chunk is stored, never inferred.** The one hard case — #24's oversized-page path,
which cuts inside a page — produces several Chunks carrying the same page number, and
`start_char`/`end_char` separate them exactly. `is_partial_page` flags it.

**`chunk_index` and `n_chunks` are dropped.** They are exactly `np.arange(len(token_count))` and
`len(token_count)`. Storing a derived value invites the two copies to disagree.

**`int32` throughout.** It caps at 2.1 billion characters against a 218k-character largest Document in
the smoke sample. It also *says* "small count" where `int64` says nothing.

**No `normalized` field, no `engine` field.** Normalization is unconditional and standing decision 10
fixed vLLM as the only engine, so both would be constant-true and record nothing. This overrides #26's
own ticket text, which listed `normalized` as mandatory.

**Chunk text is not stored.** Two reasons. The Record already holds the text and `start_char`/
`end_char` are exact, so `record["text"][start:end]` reconstructs a Chunk. And numpy stores unicode as
fixed-width UTF-32, measured at 46,473 B against 11,421 B on a real 8,700-character Document — while
storing UTF-8 bytes instead would put **two incompatible coordinate systems in one file**, because the
offsets count characters and a byte blob is indexed by bytes. That defect surfaces on the first
Document containing a non-ASCII character, which in a legal corpus is not hypothetical.

**Uncompressed — `savez`, not `savez_compressed`.** Measured on the final layout: 8,225 B → 7,400 B
for one Chunk, 42,248 B → 38,908 B for twelve. About 10%, paid for with CPU on every write and
decompression on every read of a format whose Consumer is a classifier build that reads it many times.
Normalized float vectors are close to random bytes; deflate has nothing to remove. (An earlier 0.32
ratio that made compression look worthwhile was an artifact of UTF-32 padding in a layout that no
longer exists.)

### 8.5 Write order is load-bearing

**Sidecar first, then the `.npz`, each written to a temporary name and renamed.** Rename is atomic
within a filesystem, so this establishes the invariant:

> **If the `.npz` exists, the sidecar exists.**

#28 accepted that invariant and uses the `.npz` alone as its completion marker, which makes this
ordering load-bearing rather than a preference. Without it, an interrupted Invocation leaves Documents
that Resume counts as done and that have no identity, forever.

#36 found **two shipping instances of exactly the bug this prevents**: `unstructured-ingest`'s
`write_data` is `path.open("w")` + `json.dump` with no temp-file swap — while an atomic writer sits
unused in the same module — so a crash leaves a truncated file that satisfies the resume existence
check and is complete forever. ColBERT has the same bug with `.residuals.pt`.

### 8.6 Knowingly accepted costs

- **A sidecar costs a filesystem block, not its content.** An 89-byte file allocates **4,096 bytes**
  plus an inode on both btrfs and ext4. At 100k Documents that is roughly 400 MB of blocks for ~9 MB
  of JSON, 100k extra inodes, and a second `open()` per Document on the Consumer's read path. The same
  four facts as arrays inside the `.npz` would have cost 708 bytes and no extra file. Two files per
  Document was chosen anyway, for a reason the measurement does not weigh: **the sidecar is readable by
  a person, and by any tool, without numpy.**
- **The `.npz` alone is anonymous.** Identity lives in the sidecar, so an `.npz` copied out of the tree
  on its own cannot say which Document it came from. The file path is the only remaining clue.
- **`document_vector` duplicates `chunk_vectors[0]` byte for byte** whenever a Document is one Chunk —
  46 of 49 in the smoke sample. Standing decision 3 requires both arrays so the Consumer writes one
  reader; this is accepted cost, not an oversight.

---

## 9. The LanceDB Sink

Everything in this section was measured against **lancedb 0.37.1 / pyarrow 25.0.1 / numpy 2.5.2**, not
read from documentation (#27).

### 9.1 Two tables, one pair per database

Vector search reads a whole table, so a single table with an `is_doc` discriminator would return a
mixture of Chunk vectors and Document vectors on every query. The Consumer would have to remember a
filter forever, and forgetting it yields wrong neighbours rather than an error. **Two tables make the
wrong query impossible instead of merely discouraged.** Both name the vector column `vector`, so
`.search()` works without naming it.

**`documents`** — one row per Document:

| column | type |
|---|---|
| `document_name` | `string` |
| `run_label` | `string` |
| `source_file` | `string` |
| `source_digest` | `string` |
| `created` | `timestamp[us, UTC]` |
| `n_chunks` | `int32` |
| `vector` | `fixed_size_list<float32, stored_dim>` |

**`chunks`** — one row per Chunk:

| column | type |
|---|---|
| `document_name` | `string` |
| `run_label` | `string` |
| `chunk_index` | `int32` |
| `vector` | `fixed_size_list<float32, stored_dim>` |
| `start_char` | `int32` |
| `end_char` | `int32` |
| `first_page` | `int32` |
| `last_page` | `int32` |
| `token_count` | `int32` |
| `is_partial_page` | `bool` |

`chunks` does **not** repeat `source_file`, `source_digest` or `created`. `document_name` is the
identity and joins the two tables. `token_count` sits next to the Chunk vectors, so a Consumer reading
`chunks` alone can rebuild the Document vector.

**A caution the probes turned up:** omitting a column is **not** an error. An unknown column is
rejected (`field 'surprise' does not exist in table schema`), but a row missing `token_count` lands
with `None` and no complaint. The schema catches typos, not omissions, so the writer must pass every
column explicitly rather than build rows from whatever happens to be present.

### 9.2 Table metadata — eight invariant facts, write-once

```python
metadata = {
    "model_id": "nvidia/Nemotron-3-Embed-8B-BF16",
    "stored_dim": "768", "native_dim": "4096",
    "document_instruction": "passage: ", "query_instruction": "query: ",
    "pooling": "token_weighted_mean",
    "chunker": "greedy_page_pack", "chunk_budget_tokens": "32701",
}
```

Eight, not nine: `layout` is a filesystem fact and LanceDB has no filesystem layout.

`pa.schema(..., metadata=...)` survives reopen and survives `add()`. It comes back as bytes, so a
reader decodes it. It is **write-once**: `Table` exposes only `replace_field_metadata` and
`update_field_metadata`, both field-level, and the dataset-level `replace_schema_metadata` raises
`ImportError` without a separate `pylance` install. **That immutability is the point** — it makes the
block an assertion about the table rather than a comment on it.

Repeating the eight as columns was rejected: there is exactly one model per table by construction, so
`WHERE model_id = …` answers a question nobody has.

**No `invocations` log.** Lance is versioned; `list_versions()` already records every write with a
timestamp and a row count, so writing our own would be a second, worse copy of something the format
maintains.

### 9.3 The two checks

**Width is enforced by the column type, for free.** A `fixed_size_list<float32, 768>` refuses a
4096-wide vector: `Cast error: Cannot cast to FixedSizeList(768): value at index 0 has length 4096`.
Unlike the `.npz` Sink, this needs no separate file and cannot be skipped.

**Model identity is enforced by comparing metadata before the first write.** Open the table, decode
the eight facts, compare against the current Invocation, stop and report both values on disagreement.

#36 found the counter-example that makes this worth its lines: **ColBERT's resume carries a literal
`# TODO: Verify config matches`**, so resuming with a different checkpoint or `dim` silently corrupts
the index. #26's manifest comparison and this table metadata are that TODO, done.

### 9.4 Namespacing — a column, not a table name

One database holds one `documents` and one `chunks`; `run_label` is a column and part of the key.

Table-per-label was rejected on a concrete obstacle rather than taste: **table names accept only
`[A-Za-z0-9._-]`** — `/`, space, `:`, `+`, `@` and `#` are all rejected — while `_parse_runs`
(`src/paperscale/cli.py:66`) only strips whitespace and checks non-empty and unique. Table-per-label
therefore needs a label sanitizer, which is a second name-mangling rule with its own collision
question — precisely the work #22 already did once, redone worse. (The CLI answer to the same problem
is [§14.4](#144-run-label-validation--enforced-in-embed-only): validate, do not sanitize.)

**Namespaces exist and cannot be used.** `db.create_namespace(["legal"])` succeeds and
`list_namespaces()` lists it, but `create_table` has no `namespace` parameter in 0.37.1's Python sync
API.

To keep two embedding models side by side, use a second database directory. The metadata check makes
the alternative fail loudly instead of silently.

### 9.5 Write semantics — `add()` when new, `merge_insert` only to replace

**This is the amended write path (#40); #27's original wrote everything through `merge_insert`.**

`merge_insert` is a read-modify-write against the **whole table**, so each call costs O(table) rather
than O(batch), and N Documents at batch B costs **O(N²/B)**. That put batch size in a fight with the
crash-loss window with no comfortable value in between. It does not have to fight: Resume already
knows whether a Document is new, and for a new Document `merge_insert` buys nothing — there is no
matching row to read, update or delete.

**New Document → `add()`. Replacing an existing Document → `merge_insert`.** The quadratic term
disappears from the common case, and batch size is then bounded by crash-loss alone.

On the overwrite path, everything #27 specified is still load-bearing:

```python
# documents: key (run_label, document_name)
tbl.merge_insert(["run_label", "document_name"]) \
   .when_matched_update_all().when_not_matched_insert_all().execute(rows)

# chunks: key (run_label, document_name, chunk_index) + a Document-scoped delete
tbl.merge_insert(["run_label", "document_name", "chunk_index"]) \
   .when_matched_update_all().when_not_matched_insert_all() \
   .when_not_matched_by_source_delete(f"run_label = '{rl}' AND document_name = '{dn}'") \
   .execute(rows)
```

**The scoped delete on `chunks` is not optional, and it was measured.** Seed a Document with
`chunk_index` 0, 1, 2; re-embed after a re-OCR that yields two Chunks; a plain upsert writes 0 and 1
and **leaves `chunk_index = 2` behind**. The Document then has a phantom third Chunk whose vector
describes text that no longer exists, and the Document vector recomputed from that table is wrong.
Scoping the delete to the Document leaves every other Document untouched — verified: the re-embedded
Document dropped to two Chunks while its neighbour kept its rows.

**That predicate is a SQL string built from a filesystem path, and it must escape `'` as `''`.**
Measured with `law/O'Brien v. State.pdf`, an unremarkable name in a legal corpus: the naive f-string
raises `Error tokenizing statement`. It failed loudly that time; a name shaped like
`x' OR document_name != '` would not, and `when_not_matched_by_source_delete` **deletes**. Every
identifier interpolated into a LanceDB predicate comes from `Source-File`, which is knowingly left
unnormalized and which no one in this pipeline controls.

Append was rejected because it duplicates whenever Resume and the Sink disagree. Overwrite was
rejected because it discards the current version of a table other Runs share — though note the
ticket's original premise that *"overwrite loses concurrent work"* is **false** for Lance: the format
is versioned, and after `mode="overwrite"` the previous rows are still readable via `checkout(v)`.
Overwrite is the wrong default, but not for the stated reason.

**Single writer.** `merge_insert` is materially worse under concurrency than `add`. The shape that
falls out across #26, #27 and #30 is: **workers embed, one writer commits.**

### 9.6 Batch size — 64 Documents, not a flag

It anchors to `--concurrency 64`, so one batch is about one full sweep of in-flight requests rather
than an unrelated constant. It bounds a crash to 64 Documents of GPU time. And because a Document is
done only when *every* enabled Sink holds it, a smaller batch means LanceDB lags the `.npz` tree by
less, so derived Resume state after a crash is closer to the truth.

**Not a flag.** Both costs are structural and neither is observable mid-run, so a flag would invite
tuning against a metric that does not exist.

**The accepted cost, stated plainly.** One `add()` produces one data file — three appends made three
fragments in the probe. Using #37's smoke measurement (49 Documents produced 52 Chunks plus one
Document vector each, so ~2.06 rows per Document):

| batch | rows per fragment | fragments per 100k Documents |
|---|---|---|
| **64** | ~132 | ~1,562 |
| 512 | ~1,055 | ~195 |

These are small fragments, and fragmentation is a read-time cost the Consumer pays forever. **The
mitigation is LanceDB compaction after a large Invocation, and it is now verified:** one
`Table.optimize(cleanup_older_than=timedelta(0))` collapsed forty fragments to one with every row intact
([§16.1](#161-lancedb-compaction--confirmed-by-measurement)). It is the Consumer's call to make, not
`embed`'s.

**Fragmentation costs read time, and does not cost file descriptors.** Both full-table scans `embed`
performs — Resume derivation at startup and the `merge_insert` overwrite path — hold a **flat +16**
descriptors over baseline whether the table carries 50 fragments or 600, and a 600-fragment table
completes both under a soft `RLIMIT_NOFILE` of 128
([§16.1](#161-lancedb-compaction--confirmed-by-measurement)). So the ~1,562 figure above is nowhere near
any descriptor limit, including the 1024 that containers and older distributions commonly default to.
**Batch 64 is bounded by crash-loss and read-time cost, and by nothing else.**

### 9.7 Indexes

**A `BTree` scalar index on `document_name` in both tables** — `create_index("document_name",
config=BTree())`. `create_scalar_index` is deprecated in favour of this form — verified on lancedb
0.37.1, where it raises `DeprecatedWarning: … deprecated as of 0.25.0` and the `config=BTree()` form
raises nothing ([§16.1](#161-lancedb-compaction--confirmed-by-measurement)). It is lossless, so it
costs nothing but build time, and it serves the two lookups this design actually performs: the
`merge_insert` key match, and Resume asking whether a Document is known.

**No vector index.** `create_index` builds IVF_PQ, which is **lossy**: it trades recall for speed, and
the right trade depends on a corpus size and a query pattern that live in the Consumer. Search works
with no index at all — brute force, exact, returns `_distance` — so the default is correct rather than
merely absent. (`create_index(metric=...)` is likewise deprecated in favour of
`create_index(col, config=IvfPq(...))`, should the Consumer ever build one.) There is therefore **no
index-build phase** to display or to wait on at the end of an Invocation.

---

## 10. Provenance: every recorded fact and where it lives

| Fact | Scope | `.npz` Sink | LanceDB Sink | Decided by |
|---|---|---|---|---|
| `model_id` | Invocation | manifest | table metadata (both tables) | #26 / #27 |
| `stored_dim` | Invocation | manifest | table metadata; also the `vector` column width | #34 / #26 / #27 |
| `native_dim` | Invocation | manifest | table metadata | #34 |
| `document_instruction` | Invocation | manifest | table metadata | #37 / std. dec. 9 |
| `query_instruction` | Invocation | manifest | table metadata | #37 |
| `pooling` = `"token_weighted_mean"` | Invocation | manifest | table metadata | #25 / #26 |
| `chunker` = `"greedy_page_pack"` | Invocation | manifest | table metadata | #24 / #26 |
| `chunk_budget_tokens` | Invocation | manifest | table metadata | #24 / #37 |
| `layout` (`bare` \| `labelled`) | Invocation | manifest | **not carried** (no filesystem layout) | #28 |
| enabled Sinks | Invocation | manifest, outside the invariant block | see [§17](#17-what-the-record-does-not-settle) | #35 |
| `paperscale_version` | Invocation | manifest `invocations[]` | `list_versions()` instead | #26 / #27 |
| `source_file` (raw, unnormalized) | Document | sidecar | `documents.source_file` | #22 / #28 |
| `source_digest` = `sha256(source_file)[:16]` | Document | sidecar | `documents.source_digest` | #22 / #26 / #27 |
| `run_label` | Document | sidecar (and the path, in `labelled` layout) | `documents.run_label`, `chunks.run_label` | #22 / #26 / #27 |
| `created` | Document | sidecar (ISO-8601 Z) | `documents.created` (`timestamp[us, UTC]`) | #26 / #27 |
| `document_name` | Document | the file path itself | `documents.document_name`, `chunks.document_name` | #22 / #41 |
| `n_chunks` | Document | derived (`len(token_count)`) | `documents.n_chunks` | #26 / #27 |
| `chunk_index` | Chunk | derived (`np.arange`) | `chunks.chunk_index` | #26 / #27 |
| `start_char`, `end_char` | Chunk | arrays | columns | #24 |
| `first_page`, `last_page` | Chunk | arrays | columns | #24 |
| `token_count` | Chunk | array — **a weight, not a diagnostic** | column | #24 / #25 |
| `is_partial_page` | Chunk | array | column | #24 |
| Chunk vector | Chunk | `chunk_vectors[i]` | `chunks.vector` | std. dec. 3 |
| Document vector | Document | `document_vector` | `documents.vector` | std. dec. 3 / #25 |

**Deliberately absent**: `normalized` (constant true), `engine` (constant `vllm`), `truncated`
(exactly `stored_dim != native_dim`), `context_length_overridden` (derivable), chunk text
([§8.4](#the-decisions-under-that-table)), raw vector norms ([§1.5](#15-raw-vector-norms--considered-and-declined)),
and an `invocations` log in LanceDB (`list_versions()` already keeps one).

**A warning worth carrying from #36:** three comparable targets declare a provenance slot and never
fill it — unstructured's `enrichment_origins` (documented specifically for embeddings, written by no
encoder), docling's `chunking_info`, and LanceDB's own `safe_model_dump()`, which persists
`"model": {}` on the idiomatic path, asserted by the repo's own test. A declared-and-empty slot is
worse than none, because a Consumer trusts it. Every field in the table above must be written on every
path, or removed.

---

## 11. Resume

Resume asks exactly one question about each Document: **"do I know this name?"** No content-change
detection (standing decision 7).

### 11.1 State is derived from the outputs, not recorded

No manifest of names, no flag files. The two arguments for a separate manifest both dissolved:

- *"One cheap lookup instead of a filesystem stat per Document."* Measured on btrfs over a tree of
  20,000 Documents (40,000 files): `os.walk` collects every name in **29 ms**; reading a 1.3 MB JSON
  manifest of the same names takes 3 ms. The manifest buys **26 milliseconds**, once per Invocation.
- *"Survives a Sink being written to a remote."* Both Sinks are local by standing decision 5.

What is left is the argument against it: a manifest is a second source of truth that a crash can
desynchronise from the first, and making it trustworthy needs exactly the careful write ordering it
was supposed to spare us. **Derived state cannot drift, because the evidence *is* the work.**

- **`.npz` Sink** — walk `<out>` once and collect the `.npz` paths. #26's ordering makes this sound:
  the presence of an `.npz` implies a complete pair.
- **LanceDB Sink** — `SELECT document_name, run_label FROM documents`, served by the `BTree` index.

Resume reads its state **once at startup**. There is no per-Document lookup inside the loop.

### 11.2 Two Sinks, one answer — the intersection

**A Document is done when *every* enabled Sink holds it.** With one Sink enabled the intersection is
that Sink's set, so the rule needs no special case.

This resolves the conflict the two Sinks would otherwise create. They have different tolerances for a
double write — LanceDB upserts harmlessly, the `.npz` pair rewrites harmlessly via rename — and the
intersection rule exploits that rather than fighting it. A crash between the two Sinks leaves a
Document in one and not the other; the intersection says "not done"; the next Invocation re-embeds it
and writes it to both, where one write is a no-op and the other completes. **The gap heals itself, and
it heals without anyone detecting that it happened.**

Two consequences, both accepted knowingly:

- **The batched Sink sets the pace.** LanceDB lags the `.npz` tree by up to one batch (64 Documents),
  so a crash re-embeds up to that many Documents the `.npz` Sink already holds. That is the price of
  batching, paid in GPU time.
- **Enabling a Sink later re-embeds the corpus.** Run once with `.npz`, then add `--lancedb`, and the
  intersection is empty. This is *correct* and expensive; the manifest's `sinks` field is what makes
  it loud before it happens ([§8.2](#82-paperscale-embedjson--the-invocation-manifest)). The vectors
  already sit in the `.npz` files and could be backfilled without touching the server, but that path
  is not designed here — it is an optimization, not a decision the design waits on.

### 11.3 The layout guard

The run-label directory appears only when one Invocation embeds more than one Run, so a `bare` tree
and a `labelled` tree are different layouts over the same output directory. With derived state the
failure is not silent *mixing* but silent **duplication**: the new paths match nothing, so every
Document is re-embedded into a parallel subtree and the old tree is orphaned.

**`layout` is therefore an invariant manifest fact**, values `bare` and `labelled`. A second
Invocation that would change it stops, reports both values, and tells the operator to use the same run
set or a fresh output directory. Relayouting on demand was rejected: moving a Consumer's files to
spare them a flag is a large act for a small convenience.

The guard is `.npz`-specific. LanceDB has no filesystem layout, and `run_label` already separates Runs
without one.

### 11.4 Documents with no usable text

They must be recorded, or they are retried on every Invocation forever. With derived state the record
has to be an **output**, since there is nowhere else for one to live.

- **`.npz`** — every array at length zero: `chunk_vectors` shaped `(0, stored_dim)`, `document_vector`
  shaped `(0,)`, and the six per-Chunk arrays shaped `(0,)`. Verified to round-trip with dtypes
  intact; the file is 2,060 B. A reader distinguishes it in one line: `z["document_vector"].size == 0`.
- **LanceDB** — one `documents` row with `n_chunks = 0` and a **NULL** vector, and no `chunks` rows.
  Verified: a `fixed_size_list` column accepts NULL, reads back as `None`, and **vector search skips
  the row** — searching a table containing one returned only the real neighbour.

A zero vector was rejected: it is not a unit vector, nothing else in the store is anything but a unit
vector, and it would sit in a search index looking like data. The empty output says what actually
happened.

The two Sinks disagree in *representation* while agreeing in *meaning* — NULL is expressible in Arrow
and not in `.npz`, which has no null. Empty is a **`run`-group outcome, not an `issue`**
([§13.2](#132-what-each-row-reads)).

### 11.5 `--no-resume` — re-embed and overwrite, delete nothing

This **diverges from the OCR precedent deliberately.** There, `_wipe_workspace_progress`
(`src/paperscale/pipeline.py:1147`) `rmtree`s `results`, `done_flags` and `worker_locks`. That is safe
because an OCR workspace is scratch. **An embed output is the deliverable**, quite possibly already
being read by the Consumer, and one pair of LanceDB tables holds several Runs — so wiping would mean a
scoped delete built from unnormalized paths, which is exactly where the `O'Brien` quoting hazard
lives.

Both Sinks are idempotent, so ignoring prior state is sufficient: every Document is re-embedded, the
`.npz` pair is rewritten via rename, and LanceDB replaces via `merge_insert`. The end state is
identical to a wipe except in one respect — outputs whose Documents have left the input are not
removed. Removing them is a different operation from "ignore prior progress", and standing decision 7
already declines to track what a corpus used to contain.

The existing help text — *"Ignore prior progress and reprocess the workspace from scratch"* —
describes this behaviour accurately; it is the OCR-side implementation that goes further than it says.
**The divergence must be said out loud in user docs, or it reads as a bug.**

### 11.6 The user-facing warning standing decision 7 requires

Drafted on #28 and ready to place verbatim:

> **Re-OCR-ing a corpus leaves stale vectors, and `embed` will not notice.**
>
> Resume asks one question about each Document: *have I seen this name before?* It does not look at
> the text. If you re-run OCR over the same PDFs — with a different model, different settings, or a
> newer version — the Documents keep their names, and `embed` will skip every one of them. The vectors
> in your output will continue to describe the *old* text, and nothing in the tool will tell you.
>
> `embed` does check two things before it starts, and stops the run if either disagrees: that the
> embedding model and its settings match what built the output, and that the output layout has not
> changed. Neither of those notices changed *text*.
>
> After a re-OCR, either embed into a fresh output directory, or pass `--no-resume` to re-embed
> everything in place.

---

## 12. The pipeline, end to end

### 12.1 Startup, in order

The order matters — several steps are preconditions for the next, and three of them can stop the
Invocation before any GPU work happens.

1. **Parse and validate flags.** Run labels against `[A-Za-z0-9._-]+`; reject `--no-npz` without
   `--lancedb`.
2. **Build the Adapter** from `--embed-model`; validate `--embed-dim` inside `[min_dim, native_dim]`.
3. **Read every Record** from every Run, derive Document names, and run the collision check.
   **Fatal on collision** ([§7.4](#74-collisions-are-prevented-and-the-residue-is-fatal-at-startup)).
4. **Ask `GET /v1/models`** → `.id` (the served model id, for the panel header and `model_id`
   provenance) and `.max_model_len`.
5. **Compute `validated_context_length`** = `min(card, server)`, then apply `--context-length` if
   given: reject above the server, warn above the card ([§4.2](#42-the-rule)).
6. **Compute `chunk_budget`** = `validated_context_length − tokens(document_instruction) − 64`. The
   Instruction's token count is one `/tokenize` call, or 0 for the empty string.
7. **Compute the effective request budget** = `max(--request-tokens, chunk_budget)`, with an explicit
   log line when it is raised ([§12.3](#123-batching--a-token-budget-never-a-count-of-chunks)).
8. **Probe the output dimension** — one cheap `/v1/embeddings` request — and assert the observed width
   equals `adapter.native_dim`. **Stop on mismatch**, reporting both numbers.
9. **Open the Sinks.** The `.npz` manifest and the LanceDB table metadata are each compared against
   the current Invocation's invariants; **stop on disagreement**, reporting both values. Warn if the
   enabled-Sink set changed.
10. **Construct the reporter** with `title=f"paperscale embed · {served_model_id}"`. This is after
    step 4 on purpose ([§13.3](#133-model-moves-to-the-header)).
11. **Derive Resume state**: one `os.walk` and one `SELECT`; intersect; log the skip count.
12. **Run.**

### 12.2 The unit of work is the Document

Already forced by three closed tickets before it was asked: #26 writes two files per Document, #27
keys its upsert on the Document, and #28 defines done as *every Sink holds this Document*. A
Chunk-level unit would need a completion state that nothing writes.

The objection is real and is recorded rather than solved: a 300-page Document and a 1-page Document
are wildly different units, so a single progress bar under-reports early and over-reports late. #29
accepted that when it chose one bar over two.

Per Document, in order:

1. `POST /tokenize` on the whole text — one call.
2. If it fits, one Chunk. Otherwise the Overflow path ([§5.3](#53-the-algorithm)).
3. Chunks are packed into `/v1/embeddings` requests bounded by the effective request token budget,
   **mixing Documents**.
4. Each returned vector is sliced to `stored_dim` and re-normalized.
5. Chunk vectors are pooled into the Document vector; `n_chunks == 1` short-circuits to a copy.
6. The `.npz` Sink writes sidecar-then-`.npz`, each temp-name plus rename — **two creates and two
   renames per Document**, not one write.
7. The Document is handed to the single LanceDB writer, which commits in batches of 64.

### 12.3 Batching — a token budget, never a count of Chunks

A fixed count of Chunks per request is meaningless as a control here. Greedy packing means one Chunk
may be a single short page and the next may be forty-five dense pages at the full budget, so a setting
of "16 Chunks" means anywhere from a few hundred to half a million prefill tokens, and the operator
cannot reason about it. A token budget is computable for free, because every Chunk's exact count is
already known before packing.

Two properties fall out:

- **The floor is forced.** The budget can never be smaller than one Chunk's maximum, or a full-size
  Chunk could not be sent at all. **Enforced by raising, not by rejecting:** the effective budget is
  `max(--request-tokens, chunk_budget)` with a log line. Rejecting would reject the default
  configuration, because #37's arithmetic puts the 32,000 default **below** the floor for *both*
  pinned families (Qwen3's Chunk budget is 32,704 — 704 tokens above the default; Nemotron's
  Instruction would have to exceed 704 tokens to land below 32,000, and it is about three). Silently
  ignoring the floor would make a full-size Chunk unsendable, which is the real fault. Raising is
  always safe: it only permits a larger request.
- **Requests mix Documents, and must.** The common case is a single Chunk of a few thousand tokens, so
  refusing to mix would make almost every request tiny.

**Corrected rationale — this matters, because the obvious mental model is wrong.** #30 argued the
token budget from the premise that a request is a batch the server processes together. **It is not.**
#36 verified by hand that vLLM fans an `input` array of N texts into **N independent engine requests**
— one `engine_client.encode()` per element, merged with `merge_async_iterators`. There is no
`--max-client-batch-size` analogue. The decision stands and the reason above stands; what fails is the
mechanism:

- Request batching amortizes **HTTP round trips only**. It does not help the engine batch.
- **Concurrency, not batch size, fills the engine**, which makes `vllm:num_requests_waiting` a *more*
  direct instrument than #30 assumed.
- **Split-on-failure is cheaper than #30 assumed** — re-issuing one Document at a time costs more HTTP
  round trips and **identical** engine work.

### 12.4 Concurrency — a fixed default, with an advisory

**`--concurrency 64`, fixed.** Derived-from-discovery is unavailable: vLLM publishes no batch ceiling
(#23 recorded it as *"no (none exists)"*), so every bound here is ours to choose.

The OCR side's `--max_concurrent_requests 500` does not transfer. There one request is one page image;
here one request may carry a hundred thousand prefill tokens, and vLLM's scheduler already batches
internally. The client's job is to keep the server's queue non-empty, not to flood it.

**The challenge, and why the simple thing stayed.** #36 found Vespa's feed client *adapts* without
needing a published ceiling: a `DynamicThrottler` optimising `throughput / inflight^0.3` across 128
log-scale buckets with an upward-skewed random walk. It measures the server instead of asking it
anything, which answers #30's stated reasoning directly. Two things temper it: Vespa feeds a
distributed store with a genuinely unknown aggregate capacity, whereas this feeds one vLLM server
whose queue depth is directly observable; and **pyvespa — the closer analogue — wires its
`AdaptiveThrottler` into queries only, never feeding**, so even in that ecosystem the adaptive path is
narrower than it first reads. A control loop here would be a new failure mode — oscillation, and a
throughput number that moves for reasons the operator cannot see — bought against a signal already
published.

**What changes is that the panel stops being passive.** When `vllm:num_requests_waiting` stays above
zero for a sustained window (~60 s), the event pane emits:

```
queue depth sustained; --concurrency 64 may be too high for this server
```

Three deliberate properties: it **names the flag**, so the advice is actionable without reading this
document; it is **advisory only**, so nothing oscillates; and it needs **no new plumbing**, because
`vllm:num_requests_waiting` is already parsed and already surfaced.

**`/tokenize` gets its own concurrency bound, not `--concurrency` slots.** A producer stage
(tokenize, chunk, pack) feeds a consumer stage (embed); `--concurrency` bounds the consumer only.
Sharing the slots would idle the GPU during CPU-side work.

**The bound is `(concurrency * 3) // 2`** — 96 at the default 64. Tokenize never reaches the GPU, so it
can run ahead of embedding without contending for it, but it is still HTTP against the same API server
process, so it is not free either. Integer arithmetic rather than `1.5 *` keeps it exact for odd inputs
(`--concurrency 65` -> 97) and degrades sensibly at the bottom (`--concurrency 1` -> 1). Unmeasured, on
the same footing as every other constant in [§14.2](#142-where-the-numbers-come-from).

At the defaults that is **160 concurrent sockets** (64 + 96). Comfortable against a 1024 soft
`RLIMIT_NOFILE` — see [§16.1](#161-lancedb-compaction--confirmed-by-measurement) for the other consumer
and the total.

### 12.5 The wire format

- **`encoding_format: base64`, with `embed_dtype` and `endianness` both sent explicitly.** Measured by
  JSON-encoding a real float vector against base64 of the same float32 bytes: 768-wide, 15,976 B →
  4,096 B (**3.90×**); 4096-wide, 84,994 B → 21,848 B (**3.89×**). Nearly 4× on the wire, and it
  compounds with client-side MRL, since every response arrives at native width. Parsing also becomes one
  `frombuffer` instead of thousands of float parses. **vLLM honours it — confirmed by a source read at
  0.27.1, not by a live call ([§16.2](#162-encoding_format-base64--confirmed-by-source-read)).** The
  body is:

  ```json
  {"model": "…", "input": ["…"], "encoding_format": "base64",
   "embed_dtype": "float32", "endianness": "little"}
  ```

  and the decode is `np.frombuffer(base64.b64decode(s), dtype="<f4")`. **Neither extra parameter may be
  left to its default.** `endianness` defaults to `"native"`, which is the *serving host's* byte order —
  never stated in the response, while the client's `frombuffer` would use its own; a byte-reversed
  IEEE-754 vector is finite, correctly sized and normalizes to unit length, so nothing downstream would
  catch it. `"little"` makes the server byteswap only if it must, and is a no-op on every host this
  design targets. `embed_dtype` defaults to `float32` today, but `float16` and the two fp8 values are
  lossy and return fewer bytes, which `"<f4"` would decode as a short vector rather than an error.
  `float` remains the fallback and the two are mathematically identical, so the *format* is a throughput
  decision only — **the two extra parameters are not.**
- **`truncate_prompt_tokens` is never sent.** vLLM's default on oversized input is to *error*, which
  is the safe case, so the enforcement is that the parameter never appears in a request body rather
  than that some flag is set. This is why silent truncation disqualified two engines: stored offsets
  would otherwise describe text that was never embedded, and nothing downstream could detect it.
- **`dimensions` is never sent** ([§6.1](#61-mrl-slicing--client-side-on-by-default)).

### 12.6 Retries and the failure taxonomy

**`embed` mirrors `src/paperscale/evaluation/pplx.py`, not the OCR path.** It is the same workload —
prefill-only against vLLM — its delay is bounded, and it raises instead of calling `sys.exit(1)` from
inside a worker. (`try_single_page_with_backoff` has one axis, an **uncapped** delay reaching 85
minutes at attempt 10, and `sys.exit(1)`.)

Three axes:

| axis | budget | delay | on exhaustion |
|---|---|---|---|
| fd exhaustion (`EMFILE`/`ENFILE`) | unbounded, **consumes no attempt** | `min(2**n, 30)` | n/a — self-resolves |
| connection error | 6 | **`uniform(0, min(10 * 2**(n-1), 120))`** — full jitter | raise; terminal for the Invocation |
| bad response / timeout | **`--max-request-retries 8`** | `min(2**n, 30)` | raise; terminal for the Document |

**Full jitter on the connection axis is new — there is no jitter anywhere in paperscale today.** At
`--concurrency 64` a server restart fails all 64 in flight *from a single cause*, and without jitter
all 64 retry in the same instant against a server that is still booting. The response axis fails per
request for per-request reasons, so lockstep is not a risk there and the extra variance would only
slow it down.

**Terminal without retry, for the Document:** a `400` context overflow, and `413`.

**A context-overflow 400 is a bug signal, not a routine outcome.** Chunks are sized from an untruncated
token count precisely so it cannot happen. If it happens, either the Adapter's card context length is
wrong or the packer is, and the Invocation should say so loudly at the end rather than absorb it —
which is what the `oversize` counter is for.

**A failed request splits before it fails Documents.** Because requests mix Documents, a single poison
Document would otherwise take every Document sharing its request down with it. On terminal failure of
a multi-Document request, its Documents are re-issued **one at a time**, and only those that fail
alone are recorded as failed. #30's own framing — *"one bad PDF must never end a run"* — is the reason;
without the split, one bad Document fails forty. (#36 found this is genuinely uncommon: **blast radius
is the batch, not the record, everywhere except olmOCR.**)

**A `/tokenize` failure fails the Document, not the Invocation.** Without a token count a Chunk
cannot be sized, and the design depends on vLLM erroring on overflow rather than truncating, so
proceeding on a guess is unsafe. Counted in `failed`. Tokenize shares the taxonomy and the backoff
above — same client, same server, same failure modes.

### 12.7 The end-of-run report

**An Invocation that ends with any failed Document exits non-zero.** An Invocation that exits 0 having
quietly failed 3% of a corpus is the same class of silent wrongness standing decision 7 forces this
design to guard against everywhere else.

**The report prints counts by outcome** — embedded, skipped, empty, failed, oversize — and, when
`oversize` is non-zero, says plainly that it indicates a chunker or context-length bug rather than a
corpus problem.

**Failed Documents are listed in `<out>/paperscale-embed-failures.txt`**, one Document name per line.
It is a convenience, not state: Resume derives from the outputs, so a failed Document simply has no
output and is retried automatically next time. The file is rewritten each Invocation and shares the
reserved `paperscale-embed` prefix, so the manifest collision guard covers it.

---

## 13. The TUI panel

**Variant A** — the shape the repo already renders: equal-weight counters in horizontal stat groups, a
single phase bar, the event pane below, built on the existing `RichReporter` / `Phase` in
`src/paperscale/tui.py` (standing decision 8, #29).

Two alternatives were prototyped against the **real** `RichReporter` (branch `prototype/embed-panel`,
commit `b2757f5`, local only) and rejected on what the prototype showed rather than on taste:

- **Variant B, saturation-first with an explicit verdict string.** At 80 columns the verdict cropped
  to `SATURATED (server-bound` and its own key cropped to `server queu`. **The one line the variant
  exists to show is the first thing lost.** A judgement that survives only on a wide terminal is worse
  than raw numbers, which stay legible when crushed.
- **Variant C, a second bar for Chunks.** The Chunk total is unknowable until every Document has been
  chunked, and an indeterminate bar (`total=None`) draws as `━━━━━━━━━━          ━━━━━━━━━━` — two
  bars sitting beside a real one.

### 13.1 The groups

- **`run`** — `documents`, `chunks`, `skipped`, `empty`
- **`server`** — `dim`, `tok/s`, `in-flight`
- **`issues`** — `failed`, `retrying`, `oversize`

`empty` belongs to **`run` alone**. #29's resolution listed it in both `run` and `issues`; #28 made a
zero-Chunk Document a *recorded outcome with a real output* rather than a problem, so it is not an
issue. The freed `issues` slot went to `retrying`, which is what an operator wants when throughput
drops and nothing has failed yet.

### 13.2 What each row reads

| Field | Group | Source |
|---|---|---|
| `documents` | `run` | client counter — Documents written to every enabled Sink |
| `chunks` | `run` | client counter |
| `skipped` | `run` | client counter — Resume skips, from derived state |
| `empty` | `run` | client counter — zero-Chunk Documents |
| `dim` | `server` | `stored/native`, e.g. `768/4096` |
| `tok/s` | `server` | **`Rates.prompt_tps`** |
| `in-flight` | `server` | `<client outstanding>/<vllm:num_requests_waiting>`, e.g. `16/3` |
| `failed` | `issues` | client counter — Documents that exhausted retries |
| `retrying` | `issues` | client counter — requests currently in backoff |
| `oversize` | `issues` | client counter — Chunks rejected as too long; should be 0 |

**`gen_tps` is structurally zero and must not be displayed.** Embeddings are prefill-only, so
`vllm:generation_tokens_total` never moves. This is the one place the embed panel reads a *different
field* from the OCR panel rather than merely labelling it differently, and it is impossible to notice
at runtime: the wrong field simply reads zero, which looks like an idle server rather than a bug.
`Rates` already exposes both, so nothing needs writing — it only needs getting right once.

**`in-flight` carries two numbers in one slot.** The client figure is what `--concurrency` controls;
the server figure (`Rates.waiting`, from `vllm:num_requests_waiting`) is what says the flag is set too
high. Note this is *waiting*, not the OCR panel's *running*: queue depth is the signal, admitted
requests are not. Either half renders `-` when absent, never `0`, per `format_rate`'s existing rule.

**It counts `/v1/embeddings` requests only.** `/tokenize` is served CPU-side in the API server
process and never enters the engine scheduler, so it cannot produce the queue it would be compared
against. Folding it in would inflate the client number with traffic that cannot cause the thing the
row measures. **Tokenize gets no panel row at all**; it belongs in the end-of-run report if anywhere.

Every metric name `src/paperscale/vllm_stats.py` maps was verified present on a live vLLM 0.27.2rc1:
`prompt_tokens_total`, `generation_tokens_total`, `prefix_cache_hits_total`,
`prefix_cache_queries_total`, `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`.

### 13.3 `model` moves to the header

It cannot fit as a row. `_VLLM_ROWS`'s comment records the width arithmetic: at 80 columns each of
three panels gets 22 cells of content, and the key column takes the widest label plus 2 of padding.
The OCR panel's widest label is `running` (7), leaving a value 13 cells; **embed's widest label is
`in-flight` (9), leaving 11**. The two pinned model ids are `Qwen/Qwen3-Embedding-8B` (23 characters)
and `nvidia/Nemotron-3-Embed-1B-BF16` (31). Both would render as `Qwen/Qwen…` forever. Basename-only
does not fix it (`Nemotron-3-Embed-1B-BF16` is 24), and a short Adapter-supplied tag destroys the
row's purpose: the value is sourced from `/v1/models` `.id` precisely so it reports what the server
*is serving*, not what paperscale *believes*.

The header is full pane width and the model id is constant for a whole Invocation. Precedent exists:
`pipeline.py:1308` builds the OCR reporter with `title=f"paperscale · {args.ocr_model_name}"`.

**One implementation condition: ask `/v1/models` *before* constructing the reporter**, so the header
shows the id the server returned and not the string the operator typed. The `native_dim` assertion does
not cover this — it checks the *dimension*, which does not distinguish two models of equal width.

**This also removes a height cliff, which is why it is the right trade rather than merely a fix.**
`_layout_budget` grows sections in the order bars, events, stats, and `_stat_columns` truncates each
group from the tail. With four rows in `run` and four in `server`, stats reach their fourth row only
once events have grown from `MIN_EVENT_ROWS` to `MAX_EVENT_ROWS`. Verified by running
`_layout_budget(h, 4, 1)` directly:

```
height=17  stat_rows=3
height=18  stat_rows=4
```

So below 18 rows a four-row `server` group drops its last row — `in-flight`, the saturation signal.
That is exactly the class of failure variant B was rejected for. A three-row `server` group is complete
**wherever stats render at all**, since `MIN_STAT_ROWS` is 3. The freed fourth slot stays empty; a
three-row group in a four-row panel renders one blank line, which costs nothing.

### 13.4 `embed` gets its own push function, and `vllm` is renamed `server`

**Own push function, in `src/paperscale/embed/`.** Not a parameterised `push_vllm_stats`: the mismatch
is in the *inputs*, not the row names. `push_vllm_stats(rep, stats, poller)` can reach exactly
`stats.rates()` and `poller.available`. Of embed's four original `server` rows, only `tok/s` is
reachable from that pair; `model` comes from `/v1/models`, `dim` from the flag and the Adapter, and the
client half of `in-flight` does not exist anywhere yet. Parameterising would mean threading a model id,
two dimensions and a live outstanding-request counter through a function whose entire premise is that
it needs none of them. **`push_vllm_stats` is not edited.**

What *is* shared, imported from `vllm_stats`:

- `format_rate` — absent renders `-`, never `0`, because zero is a measurement.
- `Rates` and the poller — the scraping side is engine-specific and already correct.
- **The fixed-row-set discipline.** `set_stat` can add and overwrite but can never take a row back off
  the panel, so **every branch must write every row**, or a stale row from the other branch survives on
  screen. `_VLLM_ROWS`'s comment records the bug that taught this: a permanent `status: unavailable`
  sitting next to live token rates. The embed function inherits the rule, not the row tuple. Its fixed
  row set is `("dim", "tok/s", "in-flight")`.

**The existing `vllm` group is renamed `server`** rather than joined by one. One concept keeps one
name; a box titled `server` stays honest against any server, and a box titled `vllm` becomes a lie the
first time one is not. The code keeps the vendor where the vendor is true — `vllm_stats.py`,
`push_vllm_stats`, `_VLLM_ROWS` and `VLLMStats` are all genuinely vLLM-specific, because they parse
vLLM's metric names. The complete change set is six lines:

| file | line | change |
|---|---|---|
| `src/paperscale/vllm_stats.py` | 351 | `group="vllm"` → `group="server"` |
| `src/paperscale/tui.py` | 266 | `("run", "vllm", "issues")` → `("run", "server", "issues")` |
| `tests/test_tui.py` | 347, 422, 505 | three group-name string literals |

The only OCR-visible difference is the panel's title. Every OCR row, `VLLMStats`, `Rates`, `Snapshot`,
`_CANDIDATES`, the poller and both call sites are unchanged.

### 13.5 The `tui.py` prerequisites — one discharged, one settled here, one open

#29 recorded three changes to `src/paperscale/tui.py` that variant A needs before it renders
correctly. None is an embed-specific bug; all block the chosen panel.

1. ~~**Group ordering.**~~ **Discharged by §13.4.** `_stat_columns()` hardcodes `order = [g for g in
   ("run", "vllm", "issues") if g in self._stats]` and appends the rest, which would render embed's
   groups as `run | issues | server` — `issues` promoted above the group carrying throughput, and the
   group an operator actually watches landing last. The `vllm` → `server` rename satisfies #29's
   prerequisite exactly as worded, without the ordering list carrying a redundant entry forever.
2. **Event-pane padding — open.** `_layout_budget()` hands surplus height to events, so an embedding
   Invocation — nearly silent until something fails — renders three log lines and nine blank rows. The
   OCR pipeline is chatty enough that this never showed. Either cap growth when the log is short, or
   let stats absorb the surplus instead.
3. **Resumed skips must read differently from work performed — settled here.** In the prototype the `run`
   group showed `documents 407` beside a bar reading `4309/12480`: the counter is Documents *embedded*,
   the bar is Documents *done* including 3,902 resumed skips. Both correct, together confusing. Because
   standing decision 7 removed change detection, **a large silent skip count is exactly the symptom of a
   stale output directory** — it needs to be conspicuous, not buried as one counter among four.

   **Mechanism: the bar counts only Documents that will actually be embedded.** Resume state is derived
   once at startup — one `os.walk`, measured at 29 ms over 20,000 Documents — so the skip count is known
   *before* the bar is constructed. Set the total to `corpus - skipped` and every unit in the bar is work
   performed. Four properties earn it over the alternatives:

   - **It is structural, not a counter.** #29 rejected burying this as one counter among four, which is
     exactly what a `skipped` row in `run` is. Changing what the bar *counts* cannot be overlooked.
   - **In the failure case it reads `0/0`.** Point `embed` at a stale output directory and the bar total
     is zero: the Invocation visibly has nothing to do. That is the precise symptom standing decision 7
     leaves undetected, surfaced with no threshold and no judgement call.
   - **The rate stays honest.** With skips inside the bar, the first frame reads 31% complete and then
     crawls — the confusion the prototype produced.
   - **It invents no number.** A "warn above X% skipped" rule needs a threshold nobody can justify, and
     #25 already refused a chunk-count sentinel on exactly that ground.

   The `skipped` row in `run` stays and carries the number; one startup log line states the split; the
   end-of-run report prints counts by outcome. Those complete it — the bar total is the mechanism.
   The phase description cannot carry the split: `TextColumn` is `no_wrap` with crop and holds roughly
   20-25 cells at 80 columns once the bar, `MofNCompleteColumn` and `TimeElapsedColumn` take theirs.

**New plumbing this design adds, beyond the three:** a client-side outstanding-request counter over
`/v1/embeddings` incremented on send and decremented on completion; `/v1/models` probed before the
reporter is constructed; and the sustained-queue advisory of
[§12.4](#124-concurrency--a-fixed-default-with-an-advisory).

---

## 14. The CLI surface

`paperscale embed` is shaped like `paperscale evaluate` — **hyphenated throughout** — writing `.npz`
by default with `--lancedb PATH` opting the table Sink in (#35).

### 14.1 The flag table

| Flag | Default | Notes |
|---|---|---|
| `--run LABEL=PATH` | *required* | repeatable; shares `_parse_runs`, plus embed-only label validation |
| `--out PATH` | `./vectors` | the `.npz` tree, the manifest, and the failures file |
| `--embed-model NAME` | *required* | selects the Adapter, mirroring `--ocr-model` |
| `--embed-url URL` | `http://localhost:8000` | mirrors `--pplx-url` |
| `--embed-dim N` | `768` | client-side MRL slice; rejected outside `[min_dim, native_dim]` |
| `--context-length N` | `min(card, server)` | rejected above the server's `max_model_len`; warns above the model card |
| `--api-key KEY` | `None` | mirrors the OCR side's `--api_key`, rehyphenated |
| `--lancedb PATH` | *unset* | presence enables the table Sink |
| `--no-npz` | off | disables the file Sink |
| `--concurrency N` | `64` | anchored to `--pplx-concurrency` |
| `--request-tokens N` | `32000` | anchored to `_MAX_TOKENS_PER_CHUNK`; **raised to the Chunk budget when below it** |
| `--max-request-retries N` | `8` | bounds the bad-response/timeout axis only |
| `--no-resume` | off | re-embed and overwrite; **deletes nothing** |
| `--tui` | off | "(needs the 'tui' extra)" |
| `--tui-poll-interval S` | `5.0` | mirrors `evaluate` |
| `--disk-logging PATH` | `None` | mirrors `evaluate` |

### 14.2 Where the numbers come from

Three of these defaults were shipped by #30 marked **explicitly unmeasured**, because no live embedding
server existed to measure against: the per-request token budget, the concurrency limit, and the request
retry ceiling. #35 found an in-repo sibling running a **prefill-only** workload against vLLM on the
same hardware, which is a materially better basis than an invented number and makes the embed flags
consistent with a CLI the operator already knows:

- **`--pplx-concurrency`, default 64** — chunk requests in flight against a vLLM server.
- **`_MAX_TOKENS_PER_CHUNK = 32_000`** (`src/paperscale/evaluation/pplx.py:58`), whose docstring gives
  the reason directly: *"smaller prompts let vLLM co-schedule far more of them."*
- **`--max_page_retries`, default 8** on the OCR side.

Perplexity scoring is not embedding, so **these are calibrated starting points from a comparable
workload, not measurements of this one.** They should be described that way in user docs, with
`vllm:num_requests_waiting` named as the instrument for tuning concurrency. The sustained-queue
advisory window (~60 s) sits on the same footing.

*(#35 additionally recorded a "pleasing consequence" — that `--request-tokens 32000` sat exactly on the
one-Chunk floor. That arithmetic assumed `SAFETY_MARGIN = 868`. At 64 it **inverts**: the default now
sits ~704 tokens *below* the floor for both pinned families, which is why the floor is enforced by
raising. See [§12.3](#123-batching--a-token-budget-never-a-count-of-chunks).)*

### 14.3 `--embed-model` is required where `--ocr-model` is not

This departs from `--ocr-model`'s `DEFAULT_MODEL` deliberately. The two flags do different kinds of
damage when wrong: **OCR models produce text that is roughly comparable across models; embedding models
produce vectors that are meaningless across models**, and both Sinks bake the model into the output's
identity. A default would silently pick the semantics of an entire corpus.

### 14.4 Run-label validation — enforced, in `embed` only

Labels are validated against `[A-Za-z0-9._-]+`, rejecting with a message that names the offending
character. Today `--run 'legal/2024=…'` silently creates a nested directory, because the label goes
into a filesystem path while `_parse_runs` (`src/paperscale/cli.py:66`) only strips whitespace and
checks non-empty and unique.

**Sanitizing was rejected.** It would invent a second name-mangling rule with its own collision
question — which #22 refused once and #27 refused again when it declined table-per-label over this same
charset.

**The check lives in the `embed` handler, not in `_parse_runs`.** That function is shared with
`evaluate`, where labels go into a SQLite column and never needed a constraint; tightening it globally
would start rejecting inputs that subcommand accepts today. Breaking an existing subcommand's contract
to serve a new one is the wrong trade — though the charset is worth *recommending* in `evaluate`'s docs
so labels stay portable between the two.

### 14.5 Sink selection — the path is the opt-in

`.npz` writes by default; `--lancedb PATH` enables the table Sink; `--no-npz` disables the file Sink.
LanceDB needs a database path regardless, so a separate boolean would be a second way to say the same
thing. **At least one Sink must be live: `--no-npz` without `--lancedb` is rejected.**

This is a correctness question, not a convenience one, because a Document is done only when every
enabled Sink holds it — so **enabling a second Sink later re-embeds the whole corpus**. The manifest's
`sinks` field exists to make that loud before it happens
([§8.2](#82-paperscale-embedjson--the-invocation-manifest)).

### 14.6 `--no-resume` means something different here

It re-embeds and overwrites and **deletes nothing**, where the OCR pipeline's `--no-resume` `rmtree`s
the workspace's `results`, `done_flags` and `worker_locks`. [§11.5](#115---no-resume--re-embed-and-overwrite-delete-nothing)
gives the reason. **This must be stated in user docs or it reads as a bug.**

It follows `evaluate`'s bare `store_true` rather than the OCR pipeline's mutually-exclusive
`--resume`/`--no-resume` pair. The two existing subcommands already disagree; matching the closer
sibling is the smaller inconsistency.

### 14.7 Flags deliberately not added

Each looks like it should exist and each is answered by a closed decision. Recording them stops them
being "added for completeness" during implementation.

- **No `--embed-served-model`.** The served model id is *asked* from `/v1/models`.
- **No `--chunk-tokens`.** The Chunk budget is *derived*, not chosen. `--request-tokens` bounds how
  many Chunks share a request; it does not resize a Chunk. Conflating them would let an operator
  silently change what a Chunk is between Invocations, which the manifest then records as a changed
  `chunk_budget_tokens` and refuses.
- **No `--truncate` of any spelling.** Enforcement is that `truncate_prompt_tokens` never appears in a
  request body.
- **No `--jobs`.** `evaluate` needs one for CPU-bound scoring. Here the only client-side compute is
  slicing and normalizing 768-float vectors, which has no phase worth parallelizing.
- **No LanceDB batch-size flag.** Both of its costs are structural and neither is observable mid-run
  ([§9.6](#96-batch-size--64-documents-not-a-flag)).
- ~~**No `--max-context`.**~~ **Overturned** — see [§4.2](#42-the-rule). The flag that was rejected
  would have taken the server's number as authoritative; `--context-length` can only name a number the
  server has already agreed to serve.

---

## 15. Packaging

```toml
[project.optional-dependencies]
tui = ["rich (>=13.0.0,<15.0.0)"]
embed = ["numpy", "lancedb", "pyarrow"]
```

**None of numpy, lancedb or pyarrow is a paperscale dependency today** — all three are absent from
`dependencies`, from `[project.optional-dependencies]` (which holds only `tui`) and from the dev group.
Everything measured for #26, #27 and #35 was against **lancedb 0.37.1 / pyarrow 25.0.1 / numpy 2.5.2**.

**One extra per feature** is the `tui` precedent, and `embed` is one feature. Splitting so that
`.npz`-only users skip LanceDB and pyarrow — most of the weight — tracks the Sink default exactly and
is a real saving. It was rejected because **the split can be made later without breaking anyone and the
merge cannot**, and because a second extra is a second thing to explain in every install instruction.

**Imports stay inside the handler**, following `evaluate`'s treatment of wordfreq and rapidfuzz, so
`paperscale` for OCR never pays for numpy. This is also why `embed` is its own package
([§18.1](#181-module-layout)): `src/paperscale/models/__init__.py` eagerly imports all nine OCR
Adapters, so embed Adapters placed there would pull embed code into every OCR Run.

---

## 16. Two probed claims

The record carried both forward deliberately and both have now been probed. **Neither probe was a live
call against a running server** — read what each one actually establishes before leaning on it.

### 16.1 LanceDB compaction — confirmed by measurement

**Measured against lancedb 0.37.1 / pyarrow 25.0.1**, the versions the record used, in a throwaway
environment and never installed into the project.

`Table.optimize(*, cleanup_older_than: timedelta | None = None, delete_unverified: bool = False,
retrain: bool = False)` is the operation, and it does merge small fragments. Forty 64-row `add()` calls
into a `fixed_size_list<float32, 768>` table left **forty data files** under `documents.lance/data/`;
one `optimize(cleanup_older_than=timedelta(0))` left **one**, with all 2,560 rows intact and on-disk
bytes essentially unchanged. Compaction is one of three jobs `optimize` covers — the others are pruning
old versions and adding new data to existing indexes.

Three things the signature gives up that the prose would not: `retrain` is still accepted but is
deprecated and no longer used; the call returns **`None`** on this version, so there are no stats to
log or report; and `compact_files()` survives as compaction alone. **`optimize()` is the one to call** —
every `add()` in this design leaves an old version behind as well as a fragment, and pruning those is
the other half of the win.

**[§9.6](#96-batch-size--64-documents-not-a-flag)'s batch-64 trade therefore stands as written.** The
mitigation is real, it is one call, and it is the Consumer's to run after a large Invocation; nothing in
`embed` changes.

**A third finding, from a question the record never asked: fragmentation costs no file descriptors.**
The worry was that batch 64's ~1,562 fragments would consume one fd each during the two operations that
scan the whole table — Resume derivation at startup, and the `merge_insert` overwrite path — which would
exceed a 1024 soft `RLIMIT_NOFILE` on its own, before any of `embed`'s 160 sockets. Measured by sampling
`/proc/self/fd` from a watcher thread during a full scan:

| fragments | fds at rest | peak during scan | delta |
|---|---|---|---|
| 100 | 14 | 30 | +16 |
| 400 | 14 | 30 | +16 |
| 800 | 14 | 30 | +16 |

**Flat.** Lance uses a bounded reader pool, not one handle per fragment, so 1,562 fragments cost the same
+16 as 100. Compaction reduces it further (peak 15 after `optimize()`), but it is not needed for this.

So the whole-process budget at the defaults is roughly **200 fds** — 64 embedding sockets, 96 tokenize
([§12.4](#124-concurrency--a-fixed-default-with-an-advisory)), ~30 for LanceDB, ~10 baseline — against a
1024 soft limit, in one process. **Batch size is bounded by crash-loss alone**, as [§9.6](#96-batch-size--64-documents-not-a-flag)
assumed. The fd-exhaustion retry axis inherited from `pplx.py` is therefore defensive rather than
load-bearing: it cannot be reached by normal operation and its test must force `EMFILE` through the fake
transport.

**Fragment count does not drive file descriptors — the concern is retired.** It was concrete: two
operations scan the whole table — Resume derivation at startup (`document_name`, `run_label` over
`documents`) and the `merge_insert` overwrite path, which is a read-modify-write — and if Lance held one
fd per fragment, ~1,562 fragments would exhaust a **1024** soft `RLIMIT_NOFILE` on their own, before any
of `embed`'s ~160 HTTP sockets. It does not. Sampling `/proc/self/fd` every 0.5 ms **during** each
operation rather than only after it, the peak is **flat at +16 descriptors over baseline**, unchanged
across a 12× range of fragment counts:

| fragments | Resume scan (2 cols) | full scan (incl. vector) | `merge_insert` |
|---|---|---|---|
| 50 | +16 | +16 | +13 |
| 300 | +16 | +16 | +16 |
| 600 | +16 | +16 | +16 |
| 1, after `optimize()` | +1 | +1 | +1 |

**Bounded, not linear.** Lance reads fragments through a fixed-width IO pool and closes them as it goes,
so the constant is the pool and not the table. Confirmed the decisive way rather than by extrapolation:
with the soft `RLIMIT_NOFILE` forced down to **128**, a 600-fragment table still completes all three
operations without error, peaking at 30 open descriptors in total. (This host's own limit is 1,048,576,
which is precisely why a linear relationship would not have been noticed here — hence the forced-limit
run.)

Compaction improves the descriptor picture too — the peak falls to +1 — but incidentally, not
necessarily. **The batch-size trade in [§9.6](#96-batch-size--64-documents-not-a-flag) is bounded by
crash-loss and read-time cost, and by nothing else; the fd limit does not constrain it.**

**A second finding the probe turned up unasked: [§9.7](#97-indexes)'s index guidance is accurate.**
`create_scalar_index("document_name")` raises `DeprecatedWarning: create_scalar_index is deprecated as
of 0.25.0. Use create_index() with config=BTree()/Bitmap()/LabelList() instead.`, and
`create_index("document_name", config=BTree())` builds the `BTree` with **no** warning at all. The form
this document specifies is the surviving one.

### 16.2 `encoding_format: base64` — confirmed by source read

**No embedding server was available to call.** `gigaspark:8000` serves a generative model, and
`/v1/embeddings` returns 404 there because the pooling route is mounted only for pooling-task models. So
this was settled the way #36 says to settle things — **read signatures, not prose** — against the vLLM
installed at `~/.local/share/uv/tools/vllm-omni/…/site-packages/vllm`, whose `_version.py` reads
**0.27.1**.

**The route is wired, and it is the route `embed` calls.** `entrypoints/pooling/embed/api_router.py`
mounts `POST /v1/embeddings` with body type `EmbeddingRequest`, a union whose every member — including
`EmbeddingCompletionRequest`, the `{"input": [...]}` shape `embed` sends — inherits
`EmbedRequestMixin(EncodingRequestMixin)`. `ServingEmbedding._build_openai_response` branches on
`request.encoding_format` and, for `"float" | "base64"`, calls `_openai_json_response`, which builds its
encoder from `get_pooling_output_encoder(...)`. `EmbeddingResponseData.embedding` is typed
`list[float] | str`; the `str` arm **is** the base64 string. This is not some other pooling route that
happens to share a mixin.

**The three type aliases, resolved.** All live in `vllm/utils/serial_utils.py`:

| alias | legal values | default |
|---|---|---|
| `EncodingFormat` | `float`, `base64`, **`bytes`, `bytes_only`** | `float` |
| `EmbedDType` | `float32`, `float16`, `bfloat16`, `fp8_e4m3`, `fp8_e5m2` | `float32` |
| `Endianness` | `native`, `big`, `little` | `native` |

`EncodingFormat` is wider than this record assumed. `bytes` and `bytes_only` return a raw
`StreamingResponse` of concatenated tensors with the shapes in a `metadata` header rather than JSON.
**They are deliberately not adopted:** they buy a little over base64's already-~3.9× saving in exchange
for a second parsing path and a second failure mode, and the response stops being a JSON body the rest
of the client can treat uniformly.

**What the bytes actually are.** `tensor2binary(tensor, embed_dtype, endianness)` casts to the dtype,
flattens, views to the numpy-safe view dtype, byteswaps **only** when `endianness` is neither `"native"`
nor already the server's own `sys.byteorder`, and returns `.tobytes()`; `encode_pooling_output_base64`
`pybase64.b64encode`s exactly that. For `embed_dtype="float32"` the payload is `native_dim × 4` raw
IEEE-754 single-precision bytes — `native_dim`, not `stored_dim`, because the MRL slice is client-side
and every response arrives at native width ([§6.1](#61-mrl-slicing--client-side-on-by-default)) — no length prefix, no shape, no dtype tag, nothing to parse. 768-wide
is 3,072 B, which base64s to **4,096 characters**: precisely the number
[§12.5](#125-the-wire-format) measured independently, and the 4096-wide figure matches too.

**Two parameters this record never anticipated, and `embed` must send both explicitly.**

- **`endianness: "little"`, never the default.** `"native"` means *the serving host's* byte order,
  decided on the server and stated nowhere in the response, while paperscale's `np.frombuffer` would use
  *the client's*. On a big-endian server every decoded float is byte-reversed — and a byte-reversed
  IEEE-754 word is usually still an ordinary finite number, arriving in the right count and normalizing
  to unit length, so **nothing downstream reliably detects it** and both Sinks fill with plausible
  garbage. That is exactly the class of silent
  wrongness standing decision 7 exists to refuse. Sending `"little"` makes the server byteswap if and
  only if its own order is not little, so the bytes are deterministically little-endian whatever it is
  running. On the x86-64 and aarch64 hosts this design actually targets it is a no-op — which is why it
  costs nothing to make unconditional.
- **`embed_dtype: "float32"`, likewise explicit.** The default *is* `float32` today, so this is
  belt-and-braces — but the other four values are lossy, and `float16` returns **half** the bytes, which
  `np.frombuffer(..., "<f4")` decodes as a vector of half the width rather than raising. Pin it, and
  keep [§18.3](#183-test-obligations) item 5's width check as the backstop for the day a default moves.

**So the request body is:**

```json
{"model": "…", "input": ["…"], "encoding_format": "base64",
 "embed_dtype": "float32", "endianness": "little"}
```

**and the decode is `np.frombuffer(base64.b64decode(s), dtype="<f4")`** — the explicit `"<f4"`, never a
bare `np.float32`, which is the *client's* native order and would agree with the bug rather than catch
it.

**What is still unverified, precisely.** Two things, both cheap to close with one request against any
pooling-task server:

1. **That 0.27.2rc1 carries these fields.** This read is 0.27.1; #30 verified its metric names against
   **0.27.2rc1**, so the document already spans two vLLM builds. `encoding_format` is long-standing and
   safe. `embed_dtype` and `endianness` are newer and **were not checked against 0.27.2rc1**.
2. **That an older or newer build fails loudly if it lacks them — it does not.** vLLM's
   `OpenAIBaseModel` sets `extra="allow"` and merely `logger.debug`s the ignored keys. Sending the two
   fields is therefore always *safe*, but on a build that does not know them `endianness` silently
   reverts to `"native"` and the guarantee above quietly evaporates. **A live check must assert on the
   response, not on the absence of an error:** send one known input twice, once with
   `encoding_format: "float"` and once with `base64` + `float32` + `little`, and assert the decoded
   vectors are bit-equal and `len(b64decode(s)) == native_dim * 4`.

**A method note from #36 that belongs in the implementation plan: read signatures, not prose.** Several
comparable targets' docstrings contradict their own code — including LanceDB's own retry default, whose
docstring says 10 while its signature says 7. Verify against the installed source, not the docs. Both
probes above were run that way, and both times the signature carried facts the prose did not: `optimize`
returning `None`, and the two encoding parameters nothing in twenty tickets had named.

---

## 17. What the record does not settle

Everything below is either a hole the twenty tickets left or a place where the record contradicts
itself and no later amendment resolves it. Each says what this document does instead, so an implementer
is never guessing silently.

### 17.1 Genuine contradiction: what "terminal for the run" means

**#30's retry table and #30's own prose disagree.** The table reads:

| Class | Behaviour |
|---|---|
| Terminal for the Document | 400 context overflow, 413 — no retry; recorded and counted |
| **Terminal for the run** | **a retryable class exhausting its backoff ceiling — stop, as the OCR side does** |

Four sentences later the same resolution says *"Retries are counted per request, not per Document, and
**a Document fails when a request carrying it exhausts them**."* Those cannot both be true of the same
event. #40 then split retries into three axes with separate budgets and said `embed` **raises** rather
than `sys.exit(1)`, which changes the mechanism without saying which disposition applies to which axis.

**This document adopts the reading that makes both sentences true of different axes**, and says so
rather than silently picking a side:

- **bad-response / timeout axis exhausted (`--max-request-retries 8`)** → the request fails, splits,
  and the **Document** fails. The Invocation continues. This is what "a Document fails when a request
  carrying it exhausts them" describes, and it is what makes split-on-failure meaningful at all.
- **connection axis exhausted (6 attempts, ~4 minutes of bounded backoff)** → the server is gone.
  **Terminal for the Invocation**, by an exception that propagates out of the worker rather than
  `sys.exit(1)` from inside one. This is what "terminal for the run" describes, and it is necessary:
  otherwise a dead server would burn through the corpus marking every Document failed.

This reading matches `pplx.py`'s three-axis structure, which #40 chose as the precedent. **It is a
reading, not a recorded decision.** If it is wrong, it is wrong in a place with a visible symptom, so
it is cheap to correct.

### 17.2 Holes the record leaves

1. **Where the enabled-Sink set lives for a LanceDB-only Invocation.** #35 says *"the manifest records
   which Sinks were enabled"*, so an Invocation that adds one can warn before spending a day
   re-embedding. But the manifest is an `.npz` Sink artifact. With `--no-npz --lancedb PATH` there is
   no `.npz` tree — though `<out>` still exists, because the failures file lives there. **The record
   does not say whether the manifest is written when `--no-npz` is set.** The smallest consistent
   reading is that it always is, since `<out>` always exists and the invariant comparison is useful
   regardless; that reading is what this document assumes, and it should be confirmed rather than
   inherited.
2. **The `/tokenize` concurrency bound had no value — settled here.** #40 decided it gets *"its own
   bound, not `--concurrency` slots"* and added no flag, but recorded no number. **Settled as
   `(concurrency * 3) // 2`**, 96 at the default, marked unmeasured on the same footing as every other
   constant in [§14.2](#142-where-the-numbers-come-from). See
   [§12.4](#124-concurrency--a-fixed-default-with-an-advisory). *This is a decision made in this
   document, not one recovered from the record.*
3. **The mechanism for making resumed skips visually distinct** (#29 prerequisite 3) was a requirement
   without a design — **settled here: the bar counts only Documents that will actually be embedded**, so
   a fully-skipped Invocation reads `0/0`. The requirement was precise; the mechanism was not recorded.
   Full reasoning and the rejected alternatives are in [§13.5](#135-the-tuipy-prerequisites--one-discharged-one-settled-here-one-open).
   *This is a decision made in this document, not one recovered from the record.*
4. **How `embed` reads Records is not specified.** `evaluation/runs.py`'s `load_run` flattens Records
   into `PageText` and **drops zero-length spans**, which `embed` must keep, since a zero-width page
   span is meaningful to the packer. This document therefore specifies a small Record reader inside the
   embed package that reuses `_iter_jsonl_paths`'s input-resolution semantics — a workspace directory
   (globbing `results/*.jsonl`), a bare directory of `*.jsonl`, or a single `.jsonl` file — and yields
   whole Records. **This is a choice made here, not a recorded decision.**

### 17.3 Stale text still standing in the record

A reader who goes back to the tickets will meet these. Each is superseded by a later amendment carried
in this document.

- **#25's resolution contains *"The document vector is a convenience; the chunk vectors are the
  output."*** It sits mid-resolution, inside the argument for giving every Document a vector with no
  sentinel — which is why a reader meets it while checking something unrelated. That ranking was retracted at the map level (standing decision 3, amended after #36) and
  on #31, but **#39's correction pass targeted #36 and never touched #25**, so the sentence still
  stands on the ticket. **It is superseded** — see [§1.3](#13-the-uniform-output-shape--and-the-two-consumers).
- **#26's manifest example shows `"instruction": "passage: "` and `chunk_budget_tokens: 31900`.** Both
  are stale: the Instruction is two fields (#37) and the budget is ~32,701/32,704 with
  `SAFETY_MARGIN = 64` (#37).
- **#22's rule 3 (replace the extension) and rule 5 (the digest as tiebreak)** were both amended by
  #41.
- **#27's write path (`merge_insert` for everything)** was amended by #40.
- **#29's group listing names `empty` twice**; #30 moved it to `run` alone.
- **#30's batching rationale** is disproven by #36 — the decision stands, the stated mechanism does
  not.
- **#33 is a two-engine client design for an engine now out of scope.** Nothing in it should be built.
- **#35's "must not be operator-settable"** was overturned by #37's amendment.
- **Every ticket that names the tokenizer route writes `POST /v1/tokenize`.** vLLM mounts
  `tokenize` and `detokenize` at the **top level**; only the OpenAI-compatible routes sit under
  `/v1`. The `/v1` form 404s on every build, and a 404 rides the bad-response retry axis, so each
  Document burned its whole budget on an error that could never succeed. Caught by a live run, not
  by the tests: the fake transport answers whatever URL it is handed, so the route assertion pinned
  the wrong string just as confidently. **It is superseded** — the route is `/tokenize`, corrected
  here in 23 places and in `README.md` in 2.
- **`_parse_runs` is at `src/paperscale/cli.py:66`**, not `:65` as #27, #28 and #35 all cite.

---

## 18. Implementation plan

### 18.1 Module layout

**`embed` is its own package: `src/paperscale/embed/`, holding the whole subcommand and not only the
Adapters** (#37). This mirrors `src/paperscale/evaluation/` exactly, including the `_handle_evaluate`
shape where every heavy import sits inside the handler function rather than at module scope, and it is
what makes one `embed` extra implementable at all ([§15](#15-packaging)).

The split of files *inside* the package is decided here:

| module | holds |
|---|---|
| `src/paperscale/embed/__init__.py` | public names only; **no heavy imports** |
| `adapters.py` | `EmbedModel` ABC, the five Adapters, `EMBED_MODEL_REGISTRY`, `build_embed_model` |
| `records.py` | Record reading and input resolution ([§17.2](#172-holes-the-record-leaves) item 4) |
| `names.py` | Document-name derivation, the path digest, the startup collision check |
| `budget.py` | `validated_context_length` rule, `--context-length` handling, `chunk_budget`, the request-budget floor |
| `chunking.py` | greedy page packing and the oversized-page path |
| `client.py` | the vLLM client: `/v1/models`, `/tokenize`, `/v1/embeddings`, the explicit `encoding_format`/`embed_dtype`/`endianness` triple and the `"<f4"` base64 decode ([§16.2](#162-encoding_format-base64--confirmed-by-source-read)), the three retry axes, the outstanding-request counter |
| `vectors.py` | MRL slice + re-normalize, token-weighted pooling, the single-Chunk short-circuit |
| `npz_sink.py` | manifest, sidecar, the eight arrays, write ordering, reserved names |
| `lance_sink.py` | the two tables, table metadata, `add()`/`merge_insert`, the scoped delete and its quoting |
| `resume.py` | derived state from both Sinks, the intersection, the layout guard, the Sink-set warning |
| `panel.py` | `push_embed_stats`, the fixed row set, the sustained-queue advisory |
| `run.py` | the orchestrator: startup order, producer/consumer stages, the single writer, the end-of-run report |

The subcommand itself is wired the way `evaluate` is: an `embed` subparser in
`src/paperscale/cli.py`'s `build_parser()` with `set_defaults(handler=_handle_embed)`, a `_handle_embed`
whose every `paperscale.embed.*` import sits inside the function body, and one more branch in
`pipeline.cli_main`'s shim (`if sys.argv[1:2] == ["embed"]`).

Adapters live in `paperscale/embed/adapters.py` and **not** in `paperscale/models/`, because
`models/__init__.py` eagerly imports all nine OCR Adapters — placing embed Adapters there would make
`import paperscale.models` pull numpy into every OCR Run.

### 18.2 Commit sequence

Small, independently reviewable, each with the test that proves it. Commits 1–3 touch `tui.py` and are
the only ones that change existing behaviour; they can land first and independently.

| # | Commit | Proves it |
|---|---|---|
| 1 | Rename the `vllm` stat group to `server` (6 lines, [§13.4](#134-embed-gets-its-own-push-function-and-vllm-is-renamed-server)) | existing `tests/test_tui.py` group assertions, updated |
| 2 | `_layout_budget`: stop padding the event pane on a quiet run (#29 prereq 2) | a budget test at several heights with a short log |
| 3 | Bar total counts only Documents to be embedded, so skips are structural (#29 prereq 3, [§13.5](#135-the-tuipy-prerequisites--one-discharged-one-settled-here-one-open)) | a reporter test: total is `corpus - skipped`; a fully-resumed Invocation reads `0/0`; the `skipped` row still carries the count |
| 4 | Packaging + skeleton: `embed` extra in `pyproject.toml`, empty `src/paperscale/embed/`, `embed` subparser, `cli_main` shim | `paperscale embed --help`; and `import paperscale.models` still works with numpy uninstalled |
| 5 | `adapters.py`: ABC, five Adapters, registry, `build_embed_model` | registry keys; unknown-name error text; `--embed-dim` range rejection including "above `native_dim` is rejected, not clamped" |
| 6 | `records.py`: Record reader and input resolution | workspace dir / bare dir / single file; zero-width page spans survive |
| 7 | `names.py`: derivation, digest, collision check | the identity property list ([§18.3](#183-test-obligations) 12–17) |
| 8 | `client.py`: HTTP surface, base64 decode, three retry axes with jitter, outstanding counter | a fake transport: each axis's budget and delay shape; jitter present on the connection axis only; `truncate_prompt_tokens` and `dimensions` never in a body; **every embedding body carries `encoding_format: "base64"`, `embed_dtype: "float32"` and `endianness: "little"`, none of them defaulted**; the decoder reads `"<f4"` and a payload whose length is not `native_dim * 4` fails the Document |
| 9 | `budget.py`: context-length rule, `--context-length`, chunk budget, request-budget floor | reject above server; warn above card; `max(flag, budget)` raises with a log line |
| 10 | `chunking.py`: greedy page packer | tiling, no overlap, zero-width pages never break, oversized-page cut, one tokenize call in the common case |
| 11 | `vectors.py`: slice/re-normalize/pool | **bit-equality** for `n_chunks == 1`; recomputability for `n_chunks > 1`; unit length everywhere; `sum(w) == 0` fallback |
| 12 | `npz_sink.py` | round-trip with `allow_pickle=False`; the empty-Document layout; sidecar-then-`.npz` ordering under a simulated crash; manifest compare-and-stop; reserved-name collision |
| 13 | `lance_sink.py` | width refusal; the scoped delete leaving no phantom Chunk and sparing neighbours; `O'Brien` quoting; write-once metadata compare-and-stop; empty Document skipped by search |
| 14 | `resume.py` | intersection across Sinks; a Document in one Sink only is re-embedded; layout-change stop; Sink-set-change warning; `--no-resume` deletes nothing |
| 15 | `panel.py`: `push_embed_stats` + advisory | every branch writes every row; `prompt_tps` not `gen_tps`; `-` not `0` when absent; advisory fires only after a sustained window and changes nothing |
| 16 | `run.py`: orchestration, end-of-run report, exit code, failures file | end-to-end against a fake server: counts by outcome, non-zero exit on any failure, `paperscale-embed-failures.txt` contents |
| 17 | Docs: README `## Embed` section ([§19](#19-documentation-obligations)) | — |

**Both probes [§16](#16-two-probed-claims) called for are done, and neither blocks a commit now.**
LanceDB compaction is measured, so commit 13's docstring may claim fragmentation is mitigated and name
`Table.optimize()`. `encoding_format: base64` is confirmed by source read at vLLM 0.27.1 rather than by
a live call, so **one live check still belongs against whatever build is actually deployed, before or
alongside commit 8** — not to learn whether the fields are accepted (`extra="allow"` means an unknown
field is ignored, never rejected, so acceptance proves nothing) but to prove the round trip: one known
input sent twice, once as `float` and once as `base64` + `float32` + `little`, must decode bit-equal
with `len(b64decode(s)) == native_dim * 4`
([§16.2](#162-encoding_format-base64--confirmed-by-source-read)).

### 18.3 Test obligations

Properties the record says must hold. **[explicit]** = a ticket names a test; **[measured]** = a ticket
records a measurement the implementation must preserve; **[implied]** = the property is stated as
load-bearing but no test is named.

**Pooling and dimensions**

1. **[explicit]** With `n_chunks == 1` the Document vector equals the Chunk vector **bit-exactly** —
   "a test asserts bit-equality, not approximate equality."
2. **[implied]** With `n_chunks > 1`, recomputing `normalize(Σ token_count[i] · chunk_vectors[i])` from
   the stored arrays reproduces `document_vector`.
3. **[implied]** Every stored vector is unit length.
4. **[implied]** `--embed-dim` outside `[min_dim, native_dim]` is rejected; above `native_dim` is
   rejected rather than clamped.
5. **[implied]** A response whose width differs from `adapter.native_dim` stops the Invocation,
   reporting both numbers.
6. **[implied]** `sum(token_count) == 0` with `n_chunks > 1` falls back to uniform weights; a weighted
   sum with zero norm is an error, never `NaN` in a Sink.

**Chunking**

7. **[implied]** Chunks tile the Document exactly once: `chunk[0].start_char == 0`,
   `chunk[i].end == chunk[i+1].start`, `chunk[-1].end == len(text)`.
8. **[implied]** A page whose `natural_text` is `None` (zero-width span) never forces a Chunk break.
9. **[implied]** A page that alone exceeds the budget is cut at the last `\n` at or before it, else
   hard-cut; `is_partial_page` is set; text is never dropped and the Document never fails.
10. **[implied]** The common case costs exactly one `/tokenize` call per Document.
11. **[implied]** Assembled Chunks are never re-tokenized and never exceed the budget when tokenized
    whole.

**Identity**

12. **[implied]** `case.pdf` → `case.pdf.npz` and `case.tiff` → `case.tiff.npz` — distinct.
13. **[implied]** `/a/b/../c.pdf` and `/a/b/c.pdf` derive **different** names (`a/c.pdf` vs
    `a/b/c.pdf`); `/a/./b.pdf` and `/a/b.pdf` derive the **same** name `a/b.pdf`.
14. **[implied]** Leading `..` survives `normpath` and is removed by the traversal guard.
15. **[implied]** The digest is `sha256(source_file.encode()).hexdigest()[:16]`, computed identically
    by both Sinks, over the **string** and never the text.
16. **[implied]** Missing/empty `Source-File`, a path sanitizing to nothing, and an over-long path
    component all fall back to the digest as the name.
17. **[implied]** A leading-slash or tarball-collapse collision is **fatal at startup**, listing every
    colliding group with both raw `Source-File` values, before any request is sent.

**`.npz` Sink**

18. **[measured]** The empty-Document layout round-trips with dtypes intact — `chunk_vectors
    (0, stored_dim)`, `document_vector (0,)`, six per-Chunk arrays `(0,)`; `z["document_vector"].size
    == 0` distinguishes it.
19. **[implied]** `np.load(..., allow_pickle=False)` succeeds — no object arrays anywhere.
20. **[implied]** Write order sidecar → `.npz`, each temp-name plus rename, so the `.npz` implies the
    sidecar. A simulated crash between them leaves no `.npz`.
21. **[implied]** A second Invocation with any of the nine invariant facts changed **stops** and
    reports both values; on a match exactly one `{created, paperscale_version}` entry is appended and
    nothing else changes.
22. **[implied]** A Document whose derived output lands on the reserved `paperscale-embed` prefix is a
    fatal collision, never an overwrite.

**LanceDB Sink**

23. **[measured]** `fixed_size_list<float32, 768>` refuses a 4096-wide vector.
24. **[measured]** Re-embedding a Document that drops from three Chunks to two leaves **no** phantom
    `chunk_index = 2`, **and the neighbouring Document keeps its rows**.
25. **[measured]** `law/O'Brien v. State.pdf` round-trips through the delete predicate (`'` escaped as
    `''`); a name shaped like `x' OR document_name != '` must not delete other Documents.
26. **[measured]** Table metadata survives reopen and `add()`, is write-once, and a mismatching second
    Invocation stops before the first row is written.
27. **[measured]** An empty Document is one `documents` row with `n_chunks = 0` and a NULL vector, no
    `chunks` rows, and **vector search skips it**.
28. **[implied]** Every column is passed explicitly on every write — a row omitting `token_count` lands
    as `None` with no complaint.
29. **[measured]** One `add()` produces one data file; writes are batched at 64 and `merge_insert` runs
    from a single writer.
30. **[implied]** A Document Resume knows to be new is written with `add()`, never `merge_insert`.

**Resume**

31. **[implied]** A Document is done only when **every** enabled Sink holds it; a Document present in
    one Sink and not the other is re-embedded and both Sinks end consistent.
32. **[implied]** Enabling a second Sink re-embeds the whole corpus, and the Invocation says so before
    starting.
33. **[implied]** A layout change (`bare` ↔ `labelled`) stops the Invocation with both values reported.
34. **[implied]** `--no-resume` re-embeds and overwrites and **deletes nothing** — no `rmtree`, no
    scoped delete.

**CLI and packaging**

35. **[implied]** `--embed-model` is required; omitting it is an error, not a default.
36. **[implied]** A run label outside `[A-Za-z0-9._-]` is rejected **by `embed`** with a message naming
    the offending character — and the same label is still **accepted by `evaluate`**.
37. **[implied]** `--no-npz` without `--lancedb` is rejected.
38. **[implied]** `--context-length` above `server_max_model_len` is rejected; above
    `card_context_length` it is allowed and warns, naming both numbers.
39. **[implied]** `paperscale` for OCR still imports without numpy / lancedb / pyarrow installed.

**Panel and reporting**

40. **[implied]** The `server` group renders **ahead of** `issues`.
41. **[implied]** The event pane does not pad with blank rows on a quiet Invocation.
42. **[implied]** The panel reads `prompt_tps`, never `gen_tps`.
43. **[implied]** Resume skips are visually distinct from work performed.
44. **[implied]** The advisory fires only after `vllm:num_requests_waiting > 0` for a sustained window,
    and changes nothing.
45. **[implied]** An Invocation with any failed Document exits non-zero and writes
    `<out>/paperscale-embed-failures.txt`.
46. **[implied]** A multi-Document request that fails terminally is re-issued one Document at a time,
    and only Documents that fail alone are recorded failed.

**Wire format**

47. **[implied]** Every `/v1/embeddings` body carries `encoding_format: "base64"`,
    `embed_dtype: "float32"` and `endianness: "little"` — the last two explicit, never left to their
    defaults ([§16.2](#162-encoding_format-base64--confirmed-by-source-read)).
48. **[implied]** The base64 decoder reads `"<f4"` and never a bare `np.float32`, and a payload whose
    length is not `native_dim * 4` fails the Document rather than yielding a short or byte-reversed
    vector.

---

## 19. Documentation obligations

**The pinned-model rationale belongs here, not in user-facing docs.** #31 asked the question
explicitly, and the split is between *why* and *how*:

- **This document owns the rationale** — why Qwen3-Embedding and Nemotron-3-Embed, why "long context"
  means 32K rather than "it always fits", why the two engines that instrument this workload better were
  ruled out, why the Adapter carries exactly three facts. Those are design arguments with downstream
  consequences (the chunking machinery, the Adapter registry shape, the engine narrowing). A user
  reading the README to run a command does not need them, and burying them there would mean the next
  person to question a decision reads a usage guide instead of a design record.
- **The README owns the consequences.** A new `## Embed` section, sibling to `## Evaluate`, carrying:
  1. Which models are supported and how to serve them under vLLM (the registry keys, verbatim).
  2. The flag table, with the three anchored-but-unmeasured defaults labelled as starting points and
     `vllm:num_requests_waiting` named as the tuning instrument.
  3. **The re-OCR warning of [§11.6](#116-the-user-facing-warning-standing-decision-7-requires),
     verbatim.** This is the single most important paragraph in the user docs, because it describes a
     failure the tool deliberately cannot detect.
  4. **`--no-resume` deletes nothing here**, unlike the OCR command — stated plainly, or it reads as a
     bug.
  5. That `Source-File` is unnormalized upstream, so a corpus OCR'd from a different working directory
     derives different Document names ([§7.6](#76-the-limitation-normalization-cannot-close)).
  6. That enabling a second Sink re-embeds the corpus.
  7. That `--context-length` above the model card is permitted and warns, and what the warning means:
     quality above the card is *unmeasured by the vendor*, not measured and lower.
  8. The `embed` extra and the install line.
- **`CONTEXT.md` was updated alongside this document**, and is now tracked. Its **Sink** and **Resume**
  entries were scoped to the **Invocation** rather than the **Run** — the distinction this document
  leans on throughout, and which those two entries alone still blurred. **Chunk budget** and
  **Stored / Native dimension** were added as terms; **Overflow** gained its measured 6%; and
  **Document name** gained the collision rule from #41. Deliberately *not* added: `manifest`,
  `sidecar`, `stored_dim`. Those are one Sink's file roles and field names, not domain language, and a
  glossary that absorbs implementation nouns stops being one.
- **`evaluate`'s docs should recommend** the `[A-Za-z0-9._-]` label charset, so labels stay portable
  between the two subcommands, without `evaluate` enforcing it
  ([§14.4](#144-run-label-validation--enforced-in-embed-only)).

**Why this file lives at `docs/design/embed.md`.** #21 asks for "one document committed to the repo
holding both the design and the implementation plan". `docs/superpowers/specs/` and
`docs/superpowers/plans/` are a tool's output directory, not the project's design record, and this
document is neither a spec awaiting a plan nor a plan without its design — it is both, and it outlives
the effort that produced it. `docs/research/` is already the convention the research tickets used for
their findings, so `docs/design/` sits beside it naturally and leaves room for the next design without
either colonising `docs/` root or hiding inside a skill's tree.

---

## 20. Appendix: decision index

Every bare `#N` above refers to one of these. The map (#21) is open only because this document closes
it.

| # | Title | Status |
|---|---|---|
| [#21](https://github.com/charitarthchugh/paperscale/issues/21) | Map: a from-scratch design for paperscale embed | open — the map; holds the standing decisions |
| [#22](https://github.com/charitarthchugh/paperscale/issues/22) | The document identity rule: one name, three jobs | closed — rules 3 and 5 amended by #41 |
| [#23](https://github.com/charitarthchugh/paperscale/issues/23) | What an embedding server will tell you, and what it never will | closed |
| [#24](https://github.com/charitarthchugh/paperscale/issues/24) | Chunking strategy when a document exceeds the discovered limit | closed |
| [#25](https://github.com/charitarthchugh/paperscale/issues/25) | Pooling: one document vector from N chunk vectors | closed — carries one retracted sentence ([§17.3](#173-stale-text-still-standing-in-the-record)) |
| [#26](https://github.com/charitarthchugh/paperscale/issues/26) | The .npz file: arrays, dtypes, and provenance without a header | closed — manifest example stale |
| [#27](https://github.com/charitarthchugh/paperscale/issues/27) | The LanceDB schema and its namespacing | closed — write path amended by #40 |
| [#28](https://github.com/charitarthchugh/paperscale/issues/28) | How resume knows a document is done | closed |
| [#29](https://github.com/charitarthchugh/paperscale/issues/29) | The stats + logs panel | closed — `empty` corrected; `server` group amended by #38 |
| [#30](https://github.com/charitarthchugh/paperscale/issues/30) | Concurrency, batching, and what the panel measures | closed — batching rationale disproven; concurrency amended |
| [#31](https://github.com/charitarthchugh/paperscale/issues/31) | Write the design and implementation document | open — **this document** |
| [#32](https://github.com/charitarthchugh/paperscale/issues/32) | Markdown export silently overwrites when two sources differ only by extension | open — out of scope; gains a guard, keeps its names |
| [#33](https://github.com/charitarthchugh/paperscale/issues/33) | What SGLang exposes for embeddings, and whether it truncates silently | closed — **entirely out of scope; build nothing from it** |
| [#34](https://github.com/charitarthchugh/paperscale/issues/34) | Is MRL truncation offered, and how does an adapter express validity? | closed |
| [#35](https://github.com/charitarthchugh/paperscale/issues/35) | The CLI surface: flags, defaults, and how embed installs | closed — "not operator-settable" overturned by #37 |
| [#36](https://github.com/charitarthchugh/paperscale/issues/36) | How comparable pipelines are actually implemented | closed — its item 2 retracted in full |
| [#37](https://github.com/charitarthchugh/paperscale/issues/37) | The Adapter's concrete contents, and the Chunk budget as numbers | closed — `--context-length` bounded by the server, not the card |
| [#38](https://github.com/charitarthchugh/paperscale/issues/38) | The panel's server group: who produces its rows | closed |
| [#39](https://github.com/charitarthchugh/paperscale/issues/39) | Clear the stale text from the closed record | closed |
| [#40](https://github.com/charitarthchugh/paperscale/issues/40) | The four handoffs #30 never picked up | closed |
| [#41](https://github.com/charitarthchugh/paperscale/issues/41) | What happens when two Documents derive the same name | closed |
| [#42](https://github.com/charitarthchugh/paperscale/issues/42) | Unify output-path derivation between `embed` and the markdown export | open — filed by #41 |

**Research and prototype branches**, all **local only** (the repo's push guardrail blocked publishing):

| branch | commit | ticket | contents |
|---|---|---|---|
| `research/embedding-server-discovery` | `7449919` | #23 | `docs/research/embedding-server-discovery.md`, 932 lines — vLLM / TEI / Ollama / NIM inventory |
| `research/sglang-embedding-surface` | `361297a` | #33 | `docs/research/sglang-embedding-surface.md`, 942 lines |
| `research/comparable-pipeline-implementations` | `1576e21` / `3b95a7a` | #36 | four files, ~690 commit-pinned permalinks over fourteen pipelines |
| `prototype/embed-panel` | `b2757f5` | #29 | `prototypes/embed_panel_prototype.py`, three panel variants against the real `RichReporter` |

Branch `feat/embed` holds an abandoned 4,869-line implementation of an **earlier, rejected** design. It
is reference only and should not be read: nothing in this document derives from it, and reading it
reintroduces assumptions this map spent twenty tickets discarding.

---

## 21. Appendix: the design in Simplified Technical English

This appendix says §§1–20 again in **ASD-STE100 Simplified Technical English**, with the
`CONTEXT.md` terms. It exists for a reader who wants the whole design in short sentences before
reading the arguments, and for a reader whose first language is not English.

**It restates; it does not decide.** Every number, measurement and decision below comes from a
section above. Where the two disagree, the section above wins and this appendix is the error.

### 21.0 Context

paperscale does OCR on large sets of PDFs. A new subcommand, `paperscale embed`, will turn that OCR
text into vectors. The design was made in GitHub issue #21. That issue is a map. It holds 20 closed
decision tickets. Ticket #31 asked for one document that holds the design and the plan. That
document is this file. It is locked. No code exists yet.

### 21.1 What `embed` does, and who reads the output

**Scope.**

- `embed` reads the `results/*.jsonl` files that an OCR **Run** made.
- It sends each **Document**'s text to an external server. The server speaks the OpenAI
  `/v1/embeddings` API.
- It writes the vectors into one **Sink**, or into two.
- paperscale stops when the vectors exist and have identity. The **Consumer** does search, rerank,
  RAG and quality tests.

**Invocation.** One execution of `embed` is one **Invocation**. One Invocation uses one embedding
model. One Invocation can read more than one **Run**, because `--run LABEL=PATH` can occur more than
one time. Thus the model facts do not change inside an Invocation, and they go into one manifest or
one metadata block. The run label changes per Document, and it goes onto the Document.

**No PDFs.** `embed` never sees a directory of PDFs. It reads JSONL only. Nothing records the root
of the PDF tree. Thus every identity question gets its answer from the `Source-File` string in a
**Record**.

**The vectors are the deliverable.** This is a decision, not the usual practice. Ticket #36 read 14
comparable pipelines at pinned commits. `nomic-embed` keeps its vectors in RAM and a FAISS index,
uses them to filter, then deletes them. Its only output is a JSON list of ids. paperscale does the
opposite on purpose. The vectors live longer than the Invocation. A different project reads them
later, with no access to this repository. Thus a stranger with numpy must be able to read a Sink.

**Two Consumers, and neither one is first.**

| Consumer | Reads | Why |
|---|---|---|
| RAG | **Chunk vectors** | Ranking is max-over-Chunks. It depends on the query. A stored **Document vector** cannot depend on the query, so for RAG it is only a coarse filter. |
| Classification | **Document vectors** | sklearn, xgboost and torch want `X` with shape `(n_documents, stored_dim)` and one label per Document. There is no query. Thus somebody must do the reduction. The only question was where. |

Zero of the 14 pipelines make a Document vector. All 14 are retrieval pipelines. They do a different
job, so the agreement between them means little. An earlier note told this document to call the
Document vector "a convenience over the Chunk vectors". That note was **retracted in full**. Do not
write it. #36's research file is in the repository, so this paragraph also stops a future reader
from deleting the Document vector on that file's authority.

The usual case is `n_chunks == 1`. Then the Document vector is a bit-exact copy of the one Chunk
vector. That happened for 46 of 49 Documents in the smoke corpus. It is an accepted cost of a Sink
that is complete alone.

**Which Sink to use.** Both Sinks hold both artifacts. A split (Chunk vectors to LanceDB, Document
vectors to `.npz`) was rejected. A split puts coarse-to-fine RAG across a join between two Sinks,
empties the `documents` table of its purpose, destroys the property that one Sink alone can
recompute the Document vector, and makes both Sinks necessary for a Consumer that does both jobs.

- **LanceDB** is the working store, for both jobs.
- **`.npz`** is for portability and archive. No database. numpy is the only requirement.

Measured, because this is where the guidance bites. To build `X` from the `.npz` Sink you must open
one ZIP per Document:

```
2000 Documents, 768-dim
  per-Document .npz :  206.9 ms   (103 us/Document)
  single .npy matrix:    1.7 ms
  ratio: 124x
  100,000 Documents:  10.3 s  vs 0.08 s
  1,000,000 Documents: 103.4 s  vs 0.84 s
```

This is real but not disqualifying. It costs seconds to two minutes, one time, and the Consumer can
cache it. A `document_vectors.npy` roll-up was rejected on that number. A roll-up is written one
time at the end. Thus it fights per-Document atomicity and **Resume**: a resumed Invocation must
read every `.npz` again to rebuild it.

The token-weighted mean is a **default feature**, not the only one. A Consumer can build
first-Chunk, max-pool or `mean ⊕ max` from the Chunk vectors. `n_chunks` and `token_count` are
themselves usable features.

**Raw vector norms: examined, then declined.** Every stored vector is L2-normalized, so magnitude is
gone. You could recover it: ask for `use_activation: false`, record one float32 per Chunk, then
slice and normalize in the client. The algebra in §6.1 already proves that route equal. It is
declined because the raw norm of these models follows token count and word frequency more than
meaning, because `token_count` already gives length, and because it adds a server-side condition to
buy a feature that no Consumer asked for. It is recorded as **declined**, not as **unexamined**.

**Out of scope.** Query-side embedding (paperscale records the query **Instruction** and never
applies it); retrieval, rerank and RAG; embedding-quality tests; Qdrant or any second store;
safetensors; SGLang, TEI, Ollama and NIM as engines; and the markdown export's extension bug (#32),
whose divergence from `embed` is filed as #42.

### 21.2 The pinned models and the one engine

**Two model families: Qwen3-Embedding and Nemotron-3-Embed.** Both cards state 32,768 tokens.

"Long context" means 32K. It does not mean "the Document always fits". At about 3.6 characters per
token, 32K is about 45 dense pages.

**Overflow is the minority path, and it is measured.** Against the 49-Document smoke corpus,
**Overflow** is **3 of 49, about 6%**. The number stays the same for ratios of 3.0 to 4.0 characters
per token. The median Document is 8,880 characters. Then there is a cliff to 121k, 171k and 218k
characters. The largest is 89 pages. The tail is real and large. That is what makes the greedy
packer and the token weighting worth their complexity for a path that fires one time in sixteen. The
map first said Overflow was "routine, not exceptional". The only measurement contradicts that, so
the text was corrected.

**vLLM is the only supported engine.**

- **TEI and Ollama truncate silently.** TEI does this since v1.9.0. Ollama cannot be controlled.
  Silent truncation is fatal here: the stored character offsets would describe text that the model
  never saw, and nothing downstream could find the fault.
- **NIM** cannot report its maximum input tokens over HTTP.
- **SGLang is the better engine for this work, and it is still out.** SGLang errors on oversize
  input by default, and `/server_info` lets you ask — so truncation safety could have been a startup
  assertion. Its `GET /v1/loads` needs no flag and carries `total_prefill_uncached_tokens` and
  `total_prefill_busy_us`. Those give GPU-busy prefill tokens per second and a saturation ratio
  directly. vLLM has no equal. SGLang was dropped because **it does not serve Nemotron-3-Embed**:
  the `Ministral3Model` architecture is unregistered, SGLang does not map `…Model` to
  `…ForCausalLM`, and the Transformers fallback cannot give the MEAN pooling that Nemotron needs.
  Two model families are worth more than two engines.

**The known cost.** vLLM has no GPU-busy-time counter. Thus the panel's throughput row comes from
`vllm:prompt_tokens_total` over wall-clock time. You lose the busy-time *ratio*, not the numbers.
vLLM mounts `/metrics` always, and `src/paperscale/vllm_stats.py` already parses it.

**Do not build #33's engine-detection scheme or its error-body table.** They describe a two-engine
client for an engine that is out of scope.

### 21.3 The Adapter seam

**The rule.** Ask the server for everything it can be trusted on. The **Adapter** holds only the
facts that the server cannot be trusted for. The first form of this rule — "hardcode nothing but the
model id" — cannot be built. Three facts do not go away.

1. **MRL validity, as a range `[min_dim, native_dim]`, never a list.** No pinned family publishes a
   list. Qwen3's cards state a continuous range, "32 to N", where N is the model's own **Native
   dimension** (1024, 2560 or 4096). The value 2560 also kills any "valid points are powers of two"
   rule. Nemotron's cards give examples ("keep the first 1024 or 512 dimensions"), which are
   examples of a range. `native_dim` also serves as the wrong-model assertion.
2. **The Instruction convention and its exact strings, for the query side and the document side.**
3. **The model's card context length.** You cannot discover it, and every engine over-reports it. A
   default `vllm serve` advertises `max_model_len` 262,144 for Nemotron and 40,960 for Qwen3-8B
   against a card value of 32,768. The server's answer is wrong **in the unsafe direction**.

**The output dimension is a probe, never an ask** — one cheap request — on every engine surveyed.
TEI issue #148 is still true; a maintainer closed it COMPLETED with a comment that reframed the
question, not with a fix.

**Everything else comes from the server.** The served model id comes from `/v1/models` (`.id` and
`.root`). Exact token counts come from `POST /tokenize`.

**The complete Adapter contract:**

```python
class EmbedModel(abc.ABC):
    card_context_length: int      # the card's number; min()'d against the server at run time
    native_dim: int               # published per model; doubles as the wrong-model assertion (#34)
    min_dim: int = 32             # Qwen3's published floor, adopted as the convention
    document_instruction: str     # applied by paperscale
    query_instruction: str        # recorded for the Consumer; never applied
```

Five typed class attributes with defaults. **No methods** — unlike an OCR Adapter, an embedding
Adapter builds no prompt and parses no response. The shape matches `default_model_name` and
`preferred_longest_image_dim` on `OCRModel`.

**One Adapter per model size, not per family**, because `native_dim` changes with size. Siblings are
short subclasses.

| Registry key | `native_dim` | `card_context_length` | `document_instruction` | `query_instruction` |
|---|---|---|---|---|
| `qwen3-embedding-0.6b` | 1024 | 32768 | `""` | `"Instruct: {task_description}\nQuery:{query}"` |
| `qwen3-embedding-4b` | 2560 | 32768 | `""` | *(as above)* |
| `qwen3-embedding-8b` | 4096 | 32768 | `""` | *(as above)* |
| `nemotron-3-embed-1b` | 2048 | 32768 | `"passage: "` | `"query: "` |
| `nemotron-3-embed-8b` | 4096 | 32768 | `"passage: "` | `"query: "` |

Keys are lowercase and hyphenated, as `MODEL_REGISTRY` is. `EMBED_MODEL_REGISTRY` and
`build_embed_model(name)` mirror `MODEL_REGISTRY` and `build_ocr_model`. **There is no
`DEFAULT_EMBED_MODEL`**, because `--embed-model` is required. That is the one place where the mirror
is not exact.

**paperscale applies the document Instruction and records it.** The map first generalized from Qwen3
and assumed the convention was query-side only. It is not: Nemotron-3-Embed needs `passage: ` on
Documents and `query: ` on queries. Both families emit L2-normalized vectors. A Consumer that does
not know that Documents got a prefix will build queries that do not match. Thus the record is
load-bearing, not defensive.

A sharp edge stays true after the engine narrowing: TEI accepts `--default-prompt`, and `/info` does
not report it. So a serving flag can change what the model sees and appear in no response.

**The Instruction is two plain strings.** It was one provenance fact until #37 split it. Both are
invariant inside an Invocation, and **both Sinks must carry both**. The two families differ in
*shape*, not only in content. Verified character by character from Qwen's own helper:

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'
```

There is a space after `Instruct:` and **no space** after `Query:`. Qwen states: "No need to add
instruction for retrieval documents." So Nemotron's query side is a fixed string, and Qwen3's is a
template with a slot the caller fills. The presence of `{task_description}` tells the Consumer to
supply its own text; its absence tells the Consumer to use the string as it is.

Two facts to write down as facts:

- **`document_instruction` is `""` for Qwen3 — an empty string, never null.** `""` says "we decided
  on none". A missing value says "we did not record it". A reader of the schema will otherwise
  "tidy" one into the other.
- **If the Consumer omits Qwen3's query Instruction, retrieval performance falls by about 1% to
  5%.** Qwen publishes that number. paperscale never applies the query side, but the Consumer cannot
  decide without the number.

**The wrong-model assertion.** paperscale compares the width of the server's response with
`adapter.native_dim`. On a mismatch it **stops the Invocation** and reports both numbers. This is not
a nicety: point `--embed-model` at a server that loaded a different size of the same family, cut
every vector to 768, and the stored data holds no trace of the substitution. The Sink looks perfect
and is wrong. Resume cannot catch it, because Resume asks only about the **Document name**. The
check costs one length comparison per response, because MRL slicing happens in the client and every
response arrives at **Native dimension**.

The assertion **cannot** separate two models of equal width. That is why the served model id must
come from `/v1/models` before the reporter is built, and must appear in the panel header.

### 21.4 The context-length rule, and the Chunk budget as numbers

**There are four numbers, not two.**

| | Qwen3-Embedding-8B | Nemotron-3-Embed-8B |
|---|---|---|
| vLLM advertises on a default serve | 40,960 | 262,144 |
| `config.json` `max_position_embeddings` | 40,960 | 262,144 |
| Card states | "Context Length: 32k" | "max sequence length is 32768" |
| Card **exercises** | `max_length = 8192` in its examples | 4096 "for the evaluation results below" |
| RoPE base before scaling | — (`rope_scaling: null`) | `original_max_position_embeddings` **16,384** |

**The fourth number explains the first.** Nemotron's `rope_parameters` are `{"rope_type": "yarn",
"factor": 16.0, "original_max_position_embeddings": 16384}`, and 16,384 x 16 = **262,144** exactly.
vLLM's advertisement is the YaRN-scaled base. It is not arbitrary and it is not a bug. Qwen3 carries
`rope_scaling: null` and simply advertises 40,960. That is why only one family shows a wild number.

**`apply_yarn_scaling: False` is a trap, and the card says so.** Read plainly it says long context is
off, which would put Nemotron's real window at 16,384 and halve the **Chunk budget**. It does not
mean that. NVIDIA states that the field is a temporary vLLM compatibility field, that it preserves
the checkpoint's intended long-context RoPE behaviour, and that you must not remove it from
`config.json`. It emits a load-time warning and points at vLLM issue #48621. **If a person "fixes"
that warning by deleting the field, the model's positional behaviour changes.** The card's 32,768
stays authoritative.

Nemotron's 4096 sentence is not a limit. It is the provenance of the published benchmark scores.
Above 4096, quality is **unmeasured by the vendor**, not degraded. That is a different unknown from
"the server lies about its window". The map's single word *validated* mixed the two.

**The rule:**

```
default:   validated_context_length = min(adapter.card_context_length, server_max_model_len)

--context-length N:
    N > server_max_model_len   ->  REJECTED at startup
    N > card_context_length    ->  allowed, with an explicit warning naming both numbers
    otherwise                  ->  allowed silently
```

The card is authoritative about the model. The server is authoritative about the deployment. Neither
is authoritative about both. `min()` takes the safe half of each. It keeps the protection against a
262,144 advertisement, and it adds protection the map did not have: an operator who serves
`--max-model-len 8192` against a 32,768 card would otherwise hard-fail every long Document. For both
families on a default serve the rule gives **32,768**. The value of the rule is that it stays correct
when the deployment changes. The rule sharpens the map's governing rule: **trust the server as an
upper bound, never as a lower one.**

Only `server_max_model_len` is a correctness boundary. Above it the request cannot succeed, so
rejection at startup beats failure per Document. The card's number is a quality claim, and its
evidence is thinner than the card implies. To refuse the operator that trade would be paperscale
enforcing a quality opinion under cover of a safety check, and this map gives quality judgement to
the Consumer.

The warning must name both numbers and the real risk. "May reduce quality" understates it, because
it implies a measurement exists. The warning is about an **absence of evidence**, and it must say so.

**#35's "must not be operator-settable" is overturned, not reconciled.** Its reason — that asking the
server gives a number wrong in the unsafe direction — is still true, and it is exactly why the
default is `min()`. But that argument never supported a ban on an override. It supported not
*deriving* the value from the server alone. `--context-length` can only name a number the server
already agreed to serve.

**Nothing extra goes into a Sink.** Whether the override went above the card is derivable from
`chunk_budget_tokens` and the `card_context_length` that `model_id` identifies. A stored derived
value invites two copies to disagree.

**The card-exercised numbers are recorded and deliberately not used.** To choose them would be a
quality call paperscale cannot measure, and it lands hardest on classification: at 4096 the smoke
corpus goes from 52 Chunks to 97, so the Document vector becomes a mean over about two Chunks instead
of one. The override flag makes this reversible as soon as the Consumer has evidence.

```
budget(tok)  chars@3.6   Overflow   pct   max Chunks   total Chunks / 49 Documents
      31900     114840          3    6%            2                       52
       8192      29491          6   12%            8                       68
       4096      14746         14   29%           15                       97
```

**`SAFETY_MARGIN = 64`, and the reason changed.** The implied 868 is discarded. Nobody stated it as a
decision, and 2.6% of the context is too much to spend on nothing recorded. The margin does **not**
defend against packing — subadditivity already covers packing, and covers the Instruction too,
because `tokens("passage: ") + tokens(text) >= tokens("passage: " + text)`. What is left is a
**route-level** risk: paperscale counts tokens on `/tokenize` and sends text to `/v1/embeddings`.
If those two routes apply special tokens differently by one token, a Chunk exactly on the budget
hard-fails — because the design wants vLLM to **error** on overflow, not truncate. 64 tokens is 0.2%
of the context against a visible hard failure.

**The Chunk budget:**

```
chunk_budget = validated_context_length - tokens(document_instruction) - 64
```

| Model | validated | Instruction tokens | **Chunk budget** |
|---|---|---|---|
| Qwen3-Embedding (any size) | 32,768 | 0 (`""`) | **32,704** |
| Nemotron-3-Embed (any size) | 32,768 | about 3 (`"passage: "`) | **about 32,701** |

`chunk_budget_tokens` is computed at startup and recorded. #26's manifest example of `31900` is
**stale**. Do not copy it.

### 21.5 Chunking

The method is **greedy page packing, no overlap**. Token counts come from the server. Boundaries are
recorded as character offsets **and** page spans.

**Two facts shaped it.**

1. **The page spans are already exact character ranges.** `build_dolma_document` in
   `src/paperscale/pipeline.py` builds `attributes.pdf_page_numbers` as `[start_char, end_char,
   page_num]` triples that **tile** the text: `span[i].end == span[i+1].start`, with no gaps. Thus
   page-respecting chunking needs no offset arithmetic — a run of consecutive pages *is* a character
   range. Two details to know first:
   - The `\n` joiner between pages belongs to the **preceding** page's span. So `text[start:end]`
     for a page includes its trailing newline, for all pages but the last.
   - A page whose `natural_text` is `None` gives a **zero-width** span. Empty pages are real
     entries. They cost 0 tokens and can never force a Chunk break.
2. **paperscale cannot count tokens today.** There is no `transformers`, `tokenizers` or `tiktoken`
   in `pyproject.toml`. To count tokens is a new capability, not a call to something that exists.

**Ask the server: `POST /tokenize`.** It gives the exact count for the model that is loaded. It
adds no dependency and no second tokenizer to keep in step.

**The route has no `/v1`.** vLLM puts `tokenize` and `detokenize` at the top level. Only the
OpenAI-compatible routes use `/v1`. The wrong route gives a 404 on every build. It made each
Document fail until a live run found it. It follows the precedent in
`src/paperscale/evaluation/pplx.py`. Two properties make it affordable: the API server process
handles the route on the CPU and it never enters the engine scheduler, and the common case costs
exactly **one** call per Document.

This was the least conventional decision that is right. #36 found that **nobody counts tokens with
the correct tokenizer**. LlamaIndex defaults to `cl100k_base` whatever embedding model is configured.
Vespa's chunker counts **characters** (1000 by default, with a comment that says counting codepoints
is "too expensive") and then feeds a 512-**token** embedder.

**The algorithm:**

```
budget = chunk_budget
spans  = record["attributes"]["pdf_page_numbers"]

total = tokenize(record["text"])                     # 1 call
if total <= budget:                                  # the common case
    -> single Chunk spanning the whole Document, done

# Overflow path only
counts = [tokenize(text[s:e]) for s, e, _ in spans]  # N calls, on rare Documents
for each page i:
    if counts[i] > budget:                -> oversized-page path
    elif running + counts[i] > budget:    -> close Chunk, start a new one at page i
    else:                                 -> append page i to the current Chunk
```

**Assembled Chunks need no re-check.** To split a string can only *increase* its BPE token count,
because a merge cannot cross a split boundary. Thus `sum(tokenize(page_i)) >=
tokenize(concat(page_i))`. To pack by summed per-page counts is conservative in the safe direction.
This keeps the Overflow path at N calls instead of N plus one re-check per Chunk.

**A free calibration.** The one Document-level call gives both `total` and `len(text)`. So every
Document gives its own characters-per-token ratio at no extra cost. The oversized-page path uses that
ratio to estimate where the budget falls, then searches backwards for a newline. There is no
corpus-wide constant and no second request.

**The decisions inside it.**

- **Cut rule: greedy page packing.** Fill a Chunk with whole pages until the next page would
  overflow, then close it. Every Chunk stays citable to real pages and still uses most of the 32K
  context. One page per Chunk was rejected because it wastes the context. Pure token windows were
  rejected because they throw away the page citation that the JSONL gives for free.
- **No overlap.** Chunks tile the Document exactly one time. Overlap would double-count spans in the
  pooled Document vector, and the stored offsets would stop describing a partition, so reconstruction
  would not round-trip. Overlap is a retrieval-side tactic. The Consumer has the exact offsets and
  can re-chunk without help.
- **A single page larger than the budget.** Cut at the last `\n` at or before the budget. If the page
  has no newline, take a hard character cut. **Never drop text. Never fail the Document.** This is
  the only path that cuts inside a page, and it is why `is_partial_page` exists.
- **Record both coordinate systems.** `[start_char, end_char]` is the primitive: only it can express
  a partial page, and it makes a Chunk reconstructible by a slice of `record["text"]`. `[first_page,
  last_page]` comes from it and is what a citation shows a person. Pages alone lose the partial-page
  case. Offsets alone force every Consumer to derive pages again. The recorded fields are
  `start_char`, `end_char`, `first_page`, `last_page`, `chunk_index`, `n_chunks`, `token_count`,
  `is_partial_page`. The `.npz` Sink later drops `chunk_index` and `n_chunks` as derived.
- **Rejected: offsets from the tokenizer's output.** `/tokenize` can return `return_token_strs`,
  and it is tempting to rebuild character positions from them. `pplx.py:201` already carries the
  warning: decoded tokens carry marker glyphs (SentencePiece `▁`, BPE `Ġ`) whose handling differs per
  tokenizer, so character attribution built on them breaks silently across models. A slice by the
  character offsets already in hand is exact and model-independent.

**Zero-Chunk Documents.** A Document with no usable text gives `n_chunks == 0`. It is recorded as an
**empty output** — not a flag, not a failure. The chunker must be able to produce it.

### 21.6 Dimensions and pooling

**MRL slicing happens in the client and is on by default.** `--embed-dim` has default 768. 768 is
inside the valid range of every pinned model at every size (Qwen3 1024 / 2560 / 4096, Nemotron 2048 /
4096).

paperscale asks for **native** vectors and slices them itself. It never sends `dimensions`. The
reason is operational only, because the two routes are **the same mathematically**:

```
server-side:  normalize(slice(raw))
client-side:  normalize(slice(normalize(raw)))       # the server normalized at native width

slice(raw/‖raw‖) / ‖slice(raw/‖raw‖)‖
  = (slice(raw)/‖raw‖) / (‖slice(raw)‖/‖raw‖)        # a positive scalar cancels
  = slice(raw)/‖slice(raw)‖
  = normalize(slice(raw))
```

A slice is coordinate selection, so it is linear. To normalize is to divide by a positive scalar. The
scalar cancels. Both routes give the same unit vector, to float rounding.

What is left is a trade: responses about 5.3x larger (4096 -> 768) against a launch flag that
paperscale cannot set, cannot verify, and that fails at run time when a person forgets it.
**`dimensions` is a real per-request parameter on vLLM, and neither pinned family can use it on a
default launch.** All four `config.json` files declare no `is_matryoshka` and no
`matryoshka_dimensions`, so `PoolingParams._set_default_parameters` raises before inference: *"Model
'X' does not support Matryoshka embeddings; dimensions must be unset"*. Server-side truncation would
need `--hf-overrides '{"is_matryoshka": true}'`. On a LAN server the bandwidth is cheap and the
client-side slice is a few lines of numpy, so the route with no precondition wins. `encoding_format:
base64` buys the response size back on the wire.

Also know this: vLLM's accepted range is `[1, embedding_size]` — **its floor is 1**. Qwen3's published
floor of 32 is a model-quality claim that the server will never enforce. That is why `min_dim` is an
Adapter constant.

**Validation.** paperscale rejects `--embed-dim` outside `[min_dim, native_dim]`. A value above
`native_dim` is **rejected, never clamped**. A person who asks a 2048-dim model for 4096 has a wrong
mental model, and to hand back 2048 silently hides it.

**Order of operations: slice each Chunk vector, re-normalize, then pool.** The other order (pool
native, then slice) was rejected, because then the Document vector would depend on dimensions that
nobody writes. Under the chosen order, **a Consumer that holds one Sink can recompute the Document
vector and get the same answer.** The slice itself commutes with the average; only the per-Chunk
re-normalization makes the two orders differ.

**Both `stored_dim` and `native_dim` are recorded.** After a cut to 768, `native_dim` is the only
surviving evidence of which model served the Invocation — an 8B and a 4B of one family are otherwise
indistinguishable. Explicit `truncated` and `normalized` fields were rejected: `truncated` is exactly
`stored_dim != native_dim`, and normalization is unconditional, so a constant-true field records
nothing.

**Pooling: token-weighted mean.**

```
chunk_vector[i] = normalize(slice(server_vector[i], stored_dim))

if n_chunks == 1:
    document_vector = copy(chunk_vector[0])            # exact; no arithmetic runs
else:
    w = [chunk.token_count for chunk in chunks]
    document_vector = normalize(sum(w[i] * chunk_vector[i] for i in range(n_chunks)))
```

Note the missing division. To divide the weighted sum by `sum(w)` is a no-op, because the `normalize`
that follows discards any positive scale. To leave it out removes a step and one source of float
error. #36 found that LangChain implements this same algorithm, short-circuit included, by hand in
pure Python after it dropped numpy — and it *does* divide by `total_weight` first, which is the
no-op.

**Why token-weighted: invariance to the cut.** The Chunk budget is an implementation detail. It moves
when the validated context length moves, and the oversized-page path can move it again inside one
Document. A token-weighted mean stays approximately in place across those changes. A uniform mean
swings hard. Greedy packing makes this concrete: a two-Chunk Document can be 40 pages plus a 1-page
tail, and a uniform mean gives that tail page half of the Document vector.

This is recorded as a knowing trade, not a free win. Because Chunk vectors are unit length, token
weighting asserts that text volume equals importance. In a legal Document a one-page cover letter can
be more discriminative than a forty-page exhibit. That claim about this corpus cannot be tested
inside this effort. "Proportional to content" is the neutral default, and it is also what a
single-shot embedding of the whole Document would approximate — which is the thing that chunking
stands in for.

**Every Document gets a Document vector. No sentinel and no threshold.** The dilution concern is
real: an average of forty diverse Chunk vectors does drift toward the corpus centroid. But a sentinel
needs a Chunk-count threshold, and no measurement in this effort can justify one. `CONTEXT.md` gives
exactly this judgement to the Consumer, which has the Chunk vectors and has `n_chunks` recorded, so
it can decide without reading one vector.

**The single-Chunk case is a copy, not an average.** It does not fall out of the arithmetic: a
weighted mean of one element is then re-normalized, and the divisor is `1 ± ε` instead of `1`, so the
low bits move. paperscale short-circuits and copies. **A test asserts bit-equality, not approximate
equality.**

**Edge cases, defined and not left open.** If `sum(w) == 0`, fall back to uniform weights. This is
unreachable in practice — a Document with more than one Chunk overflowed, so it has tokens — but the
other option is a division by zero in a path nobody tests. Separately, a weighted sum with zero norm
needs the Chunk vectors to cancel exactly. Treat that as an error. Do not write `NaN` into a Sink.

**Guaranteed properties.**

- **Reproducible from one Sink.** A Consumer that holds `chunk_vector` and `token_count` gets the
  same Document vector. Thus `token_count` is a **weight, not a diagnostic**, and both Sinks must
  keep it readable next to the Chunk vectors.
- **Invariance to the cut is approximate, not exact.** A Chunk vector over forty joined pages is not
  the mean of forty single-page vectors. So a different budget shifts the Document vector a little.
  The claim is that it shifts far less than under uniform weighting.
- **The identity case is exact by construction**, not by tolerance.
- **Every stored vector is unit length.** #36 found two shipping counter-examples: vLLM's own
  long-text embedding never re-normalizes its cross-chunk mean, so dot-product scoring breaks
  silently; and Vespa truncates in MRL style with `normalize` default `false`.

### 21.7 Identity: the Document name

One name does three jobs: the `.npz` filename, the LanceDB primary key, and the Resume key. A weak
answer fails in three places at one time.

**The rule.** From a Record's `metadata["Source-File"]`:

1. **Tarball form** (`archive.tar.gz::internal/path.pdf`) — the tarball basename, without `.tar` or
   `.tar.gz`, becomes a directory, and the internal path continues below it. This reuses the markdown
   export's handling.
2. **Otherwise** — `os.path.normpath(source)`, then `lstrip("/")`, then drop remaining `..` and empty
   components.
3. **The output filename appends the source extension. It does not replace it.** `case.pdf` ->
   `case.pdf.npz`, sidecar `case.pdf.json`. `case.tiff` -> `case.tiff.npz`.
4. **The run label appears in the path only when one Invocation embeds more than one Run.** One
   `--run` -> `<out>/<mirrored path>` (**`bare`**). Two or more -> `<out>/<label>/<mirrored path>`
   (**`labelled`**).
5. **A digest of the raw `Source-File` string exists for every Document, always** —
   `sha256(source_file.encode()).hexdigest()[:16]`.

**The Document name** is the sanitized relative path *with* the source extension, for example
`run/media/cc/data/law/doc9419897.pdf`. The `.npz` Sink appends `.npz` and `.json`. The LanceDB Sink
stores it as it is in `document_name`. One derivation, two Sinks.

**Rule 2's normalization belongs to `embed` and is deliberate.** The markdown export's sanitizer at
`src/paperscale/pipeline.py:686-689` reads `safe_parts = [p for p in parts if p and p != ".."]`. It
**drops** `..` instead of resolving it, so `/a/b/../c.pdf` and `/a/b/c.pdf` give the same name. It
also keeps `.`, so `/a/./b.pdf` gives `a/./b.pdf` while the file lands at `a/b.pdf`. Resume's
correctness rests on name stability, so `embed` cannot inherit that. `normpath` resolves interior
`..` and `.` correctly and leaves *leading* `..` alone, so the traversal guard still runs after it.
**`realpath` is not used**: the PDFs can be gone by embed time, and to resolve symlinks would change
which Documents count as the same instead of normalizing how one is spelled.

**Rule 3 changed after #41.** It first replaced the extension, to mirror `get_markdown_path`. #32
established that this is exactly where that function is wrong — it silently overwrites when two
sources differ only by extension — and #22's stated reason to mirror it was consistency, not
correctness. Consistency with a bug is not worth keeping. The property that matters: **the name is a
pure function of one Record's `Source-File`**. It does not depend on iteration order, and it does not
depend on which other Documents are in the Invocation. That is what keeps Resume stable.

**The digest is over the path and never over the text.** Every Record already carries `id =
sha1(document_text).hexdigest()` (`pipeline.py:654`). It is tempting and it is wrong. That digest
changes on every re-OCR, so it would make a re-OCR look like a completely new set of Documents.
Together with "no content-change detection", that converts "Resume silently keeps stale vectors" into
"Resume silently duplicates the whole corpus" — the opposite failure, equally invisible. The digest
is over the `Source-File` **string**, which survives a re-OCR. olmOCR reached the same conclusion
independently, with sha1 over sorted path strings and never over content. `sha256(...)[:16]` is 64
bits, far past what a corpus of this size needs, and 16 characters stays usable as a filename. **Both
Sinks must compute it with the same function.**

**The digest has two jobs, not three.**

1. It goes into every output as provenance, so a Document stays identifiable when its filename is not
   unique.
2. It **is the name** when no usable path exists — `Source-File` missing or empty, the sanitized path
   reducing to nothing, or a path component over the filesystem limit (ext4 caps a component at 255
   bytes, which long legal filenames do reach).

It is **not** a collision tiebreak. #22 gave it that third job and #41 removed it. Resume derives from
the outputs, so a tiebroken name must be stable across Invocations, and neither scheme is: to suffix
the loser depends on iteration order, and to suffix every member of a colliding set reverts silently
when the set changes. Both are silent costs.

**Collisions are prevented, and the residue is fatal at startup.**

| Prevented class | Example | Prevented by |
|---|---|---|
| extension replacement | `case.pdf` + `case.tiff` | rule 3 (append) |
| `..` components | `/a/b/../c.pdf` + `/a/b/c.pdf` | rule 2 (normalization) |

| Fatal class | Example |
|---|---|
| leading slash / empty components | `/a/case.pdf` + `a/case.pdf` |
| tarball collapse | `x.tar.gz::doc.pdf` + `x.tar::doc.pdf` |
| two Records with an empty `Source-File` | both derive the same digest name |

The Invocation stops before any GPU work and lists **every colliding group with both raw
`Source-File` values**. This does not contradict "one bad PDF must never end a run": that rule
governs embedding failures during a run. This is a startup check that costs seconds, and the operator
gets a message that names exactly what to fix.

**The check is scoped to one Run.** Both Sinks key on `(run_label, document_name)`, and the
`labelled` layout puts each Run in its own subtree, so two Runs cannot collide. *(This scoping is a
clarification made in this document. The record implies it through the Sink keys but never states
it.)*

**A duplicate raw `Source-File` inside one Run is also fatal**, which matches
`DuplicateSourceFileError` in `src/paperscale/evaluation/runs.py:45`: *"Two records in one run share a
Source-File -- the join key is ambiguous."* That is a different event from a derived-name collision —
two Records nobody can tell apart, against two distinguishable Records that want the same filename.
#22 cited it for the second, and #41 corrected that. The contradiction readers found on #22 was one
correct sentence next to one misapplied citation, not two decisions in conflict.

**Measured before the decision:** the live corpus is **39,905 files, all `.pdf`, with zero collisions
of any class**. The 49-Document smoke set gives 49 distinct names. The strict option costs nothing
today. The hazard stays real, because `--pdfs` accepts "Local PDF/image paths", so a mixed corpus
makes the collision at once.

**Reserved names.** `<out>/paperscale-embed.json` (the manifest) and
`<out>/paperscale-embed-failures.txt` sit at `<out>/` in **both** layouts. In the `labelled` layout
nothing else lives at `<out>/`. In the `bare` layout they can only collide with a source at the
filesystem root named literally `paperscale-embed.<ext>`. The `paperscale-embed` prefix is
**reserved**. A Document whose output lands on it is the same class of fatal collision, never a
silent overwrite of the manifest.

**The limitation that normalization cannot close.** `Source-File` is **not** normalized upstream:
`_expand_pdf_inputs` (`pipeline.py:1093`) does `glob.glob(p)` or `[p]` and calls neither `abspath` nor
`realpath`. So the same PDF, OCR'd as `docs/a.pdf` and as `/home/cc/docs/a.pdf`, gives two names,
decided by the caller's working directory. **Nothing records the OCR-time working directory, so no
later processing can equate them.** It sounds worse than it is: Resume reads the *same* JSONL, so
names stay stable unless somebody re-OCRs the corpus from a different directory. The mirrored layout
was chosen over a `--source-root` flag because two trees side by side —
`vectors/home/cc/corpus/x.pdf.npz` beside `markdown/home/cc/corpus/x.md` — is worth more than a flag
on every Invocation that papers over an upstream inconsistency. **This belongs in the user docs.**
`source_file` is recorded **raw** in both Sinks, so the original string survives for anyone who
reconciles by hand.

### 21.8 The `.npz` Sink

**Provenance splits by scope.** `.npz` has no metadata header, and object arrays are **impossible,
not merely unwise**: `np.savez` accepts a Python dict, but a read raises `ValueError: Object arrays
cannot be loaded when allow_pickle=False`. Any metadata inside an `.npz` must be a real dtype, and to
ask a Consumer for `allow_pickle=True` is to ask it to execute whatever the file holds.

So provenance splits **by scope, not by convenience**: one Invocation manifest for the facts that
never vary, one sidecar per Document for the facts that do, and arrays only in the `.npz`.

```
<out>/paperscale-embed.json                      <- the Invocation manifest, written once
<out>/paperscale-embed-failures.txt              <- rewritten each Invocation, if any failed
<out>/run/media/cc/data/law/doc9419897.pdf.npz   <- 8 arrays, nothing else
<out>/run/media/cc/data/law/doc9419897.pdf.json  <- the Document sidecar
```

**The manifest: nine invariant facts, plus an append-only log, plus the enabled-Sink set.**

```json
{
  "model_id": "nvidia/Nemotron-3-Embed-8B-BF16",
  "stored_dim": 768,
  "native_dim": 4096,
  "document_instruction": "passage: ",
  "query_instruction": "query: ",
  "pooling": "token_weighted_mean",
  "chunker": "greedy_page_pack",
  "chunk_budget_tokens": 32701,
  "layout": "bare",
  "sinks": ["npz"],
  "invocations": [
    {"created": "2026-08-18T09:00:00Z", "paperscale_version": "0.8.0"}
  ]
}
```

The invariant block grew two times: `layout` joined it in #28 (seven -> eight), and #37 split
`instruction` into two (eight -> nine). **`sinks` sits outside the invariant block on purpose**,
because the set is allowed to change.

**A second Invocation compares the invariants and stops on any disagreement**, reports both values,
and writes no Document. This is the check the `native_dim` assertion cannot make: a run appended to a
tree that an earlier Invocation built with a *different model*. There is no content detection, so
Resume will not catch it. The manifest is the only thing that can.

On a match, `embed` appends one `{created, paperscale_version}` entry — about 60 bytes, and the only
record of how a tree built over several Invocations came to be. To overwrite the manifest was
rejected outright: the file would then describe vectors it does not describe.

**The manifest belongs to the parent process, never to a worker.** It is read and appended one time,
before workers start. Concurrent appends would race and corrupt it.

**If `sinks` changed, the Invocation says so before it starts** — *"`--lancedb` is new; this will
re-embed 47,000 Documents"* — instead of spending a day in silence. This is the same instinct as the
model checks. The map's whole shape is **make the expensive silent thing loud**, because the map
removed the mechanism that would otherwise notice.

**The sidecar: four facts that vary per Document.**

```json
{
  "source_file": "/run/media/cc/data/law/pdfs/pdfDownload10/doc9419897.pdf",
  "source_digest": "9f2b1c4d8e0a3f57",
  "run_label": "nemotron-8b",
  "created": "2026-08-18T09:04:37Z"
}
```

**The `.npz`: eight arrays, no metadata.**

| name | shape | dtype |
|---|---|---|
| `chunk_vectors` | `(n_chunks, stored_dim)` | `float32` |
| `document_vector` | `(stored_dim,)` | `float32` |
| `start_char` | `(n_chunks,)` | `int32` |
| `end_char` | `(n_chunks,)` | `int32` |
| `first_page` | `(n_chunks,)` | `int32` |
| `last_page` | `(n_chunks,)` | `int32` |
| `token_count` | `(n_chunks,)` | `int32` |
| `is_partial_page` | `(n_chunks,)` | `bool` |

- **The page range per Chunk is stored, never inferred.** The hard case is the oversized-page path,
  which cuts inside a page and makes several Chunks with the same page number. `start_char` and
  `end_char` separate them exactly, and `is_partial_page` flags it.
- **`chunk_index` and `n_chunks` are dropped.** They are exactly `np.arange(len(token_count))` and
  `len(token_count)`. A stored derived value invites the two copies to disagree.
- **`int32` everywhere.** It caps at 2.1 billion characters against a largest Document of 218k
  characters in the smoke sample. It also *says* "small count", where `int64` says nothing.
- **No `normalized` field and no `engine` field.** Normalization is unconditional, and vLLM is the
  only engine, so both would be constant-true and record nothing. This overrides #26's own ticket
  text, which listed `normalized` as mandatory.
- **Chunk text is not stored.** Two reasons. The Record already holds the text, and the offsets are
  exact, so `record["text"][start:end]` rebuilds a Chunk. And numpy stores unicode as fixed-width
  UTF-32 — measured at 46,473 B against 11,421 B on a real 8,700-character Document. To store UTF-8
  bytes instead would put **two incompatible coordinate systems in one file**, because the offsets
  count characters and a byte blob is indexed by bytes. That defect appears on the first Document
  with a non-ASCII character, which in a legal corpus is not hypothetical.
- **Uncompressed — `savez`, not `savez_compressed`.** Measured on the final layout: 8,225 B -> 7,400
  B for one Chunk, and 42,248 B -> 38,908 B for twelve. That is about 10%, paid with CPU on every
  write and decompression on every read, in a format whose Consumer is a classifier build that reads
  it many times. Normalized float vectors are close to random bytes, so deflate has nothing to
  remove. An earlier ratio of 0.32 that made compression look worthwhile was an artifact of UTF-32
  padding in a layout that no longer exists.

**Write order is load-bearing.** Write the sidecar first, then the `.npz`. Write each to a temporary
name and rename it. A rename is atomic inside a filesystem, so this establishes the invariant:

> **If the `.npz` exists, the sidecar exists.**

Resume uses the `.npz` alone as its completion marker, so this ordering is load-bearing, not a
preference. Without it, an interrupted Invocation leaves Documents that Resume counts as done and
that have no identity, forever.

#36 found **two shipping instances of exactly this bug**: `unstructured-ingest`'s `write_data` is
`path.open("w")` plus `json.dump` with no temp-file swap — and an atomic writer sits unused in the
same module — so a crash leaves a truncated file that satisfies the resume check and is complete
forever. ColBERT has the same bug with `.residuals.pt`.

**Costs accepted knowingly.**

- **A sidecar costs a filesystem block, not its content.** An 89-byte file allocates **4,096 bytes**
  plus an inode, on btrfs and on ext4. At 100k Documents that is about 400 MB of blocks for about 9
  MB of JSON, 100k extra inodes, and a second `open()` per Document on the Consumer's read path. The
  same four facts as arrays inside the `.npz` would have cost 708 bytes and no extra file. Two files
  per Document was chosen anyway, for a reason the measurement cannot weigh: **a person can read the
  sidecar, and so can any tool, with no numpy.**
- **The `.npz` alone is anonymous.** Identity lives in the sidecar, so an `.npz` copied out of the
  tree cannot say which Document it came from. Only the file path is left as a clue.
- **`document_vector` duplicates `chunk_vectors[0]` byte for byte** when a Document is one Chunk — 46
  of 49 in the smoke sample. The uniform shape requires both arrays so the Consumer writes one
  reader. This is accepted cost, not an oversight.

### 21.9 The LanceDB Sink

Everything here was **measured** against **lancedb 0.37.1 / pyarrow 25.0.1 / numpy 2.5.2**. Nothing
here was read from documentation.

**Two tables, one pair per database.** Vector search reads a whole table, so one table with an
`is_doc` discriminator would return a mix of Chunk vectors and Document vectors on every query. The
Consumer would have to remember a filter forever, and to forget it gives wrong neighbours instead of
an error. **Two tables make the wrong query impossible, not merely discouraged.** Both name the
vector column `vector`, so `.search()` works without a name.

`documents` — one row per Document: `document_name` (string), `run_label` (string), `source_file`
(string), `source_digest` (string), `created` (`timestamp[us, UTC]`), `n_chunks` (`int32`), `vector`
(`fixed_size_list<float32, stored_dim>`).

`chunks` — one row per Chunk: `document_name`, `run_label`, `chunk_index` (`int32`), `vector`,
`start_char`, `end_char`, `first_page`, `last_page`, `token_count` (all `int32`), `is_partial_page`
(`bool`).

`chunks` does **not** repeat `source_file`, `source_digest` or `created`. `document_name` is the
identity and joins the two tables. `token_count` sits next to the Chunk vectors, so a Consumer that
reads `chunks` alone can rebuild the Document vector.

**A caution the probes found:** to omit a column is **not** an error. An unknown column is rejected
(`field 'surprise' does not exist in table schema`), but a row that misses `token_count` lands with
`None` and no complaint. The schema catches typos, not omissions. So the writer must pass every
column explicitly.

**Table metadata: eight invariant facts, write-once.**

```python
metadata = {
    "model_id": "nvidia/Nemotron-3-Embed-8B-BF16",
    "stored_dim": "768", "native_dim": "4096",
    "document_instruction": "passage: ", "query_instruction": "query: ",
    "pooling": "token_weighted_mean",
    "chunker": "greedy_page_pack", "chunk_budget_tokens": "32701",
}
```

Eight, not nine: `layout` is a filesystem fact, and LanceDB has no filesystem layout.
`pa.schema(..., metadata=...)` survives a reopen and survives `add()`. It comes back as bytes, so a
reader decodes it. It is **write-once**: `Table` exposes only `replace_field_metadata` and
`update_field_metadata`, both field-level, and the dataset-level `replace_schema_metadata` raises
`ImportError` without a separate `pylance` install. **That immutability is the point.** It makes the
block an assertion about the table, not a comment on it.

To repeat the eight facts as columns was rejected: there is one model per table by construction, so
`WHERE model_id = …` answers a question nobody has. **No `invocations` log** either — Lance is
versioned, and `list_versions()` already records every write with a timestamp and a row count.

**Two checks.**

- **Width is enforced by the column type, for free.** A `fixed_size_list<float32, 768>` refuses a
  4096-wide vector: `Cast error: Cannot cast to FixedSizeList(768): value at index 0 has length
  4096`. Unlike the `.npz` Sink, this needs no separate file and cannot be skipped.
- **Model identity is enforced by a metadata comparison before the first write.** Open the table,
  decode the eight facts, compare, and stop with both values on a disagreement. #36 found the
  counter-example that earns these lines: **ColBERT's resume carries a literal `# TODO: Verify config
  matches`**, so a resume with a different checkpoint or `dim` corrupts the index silently.

**Namespacing is a column, not a table name.** One database holds one `documents` and one `chunks`.
`run_label` is a column and part of the key. Table-per-label was rejected on a concrete obstacle, not
on taste: **table names accept only `[A-Za-z0-9._-]`** — `/`, space, `:`, `+`, `@` and `#` are all
rejected — while `_parse_runs` (`src/paperscale/cli.py:66`) only strips whitespace and checks
non-empty and unique. So table-per-label needs a label sanitizer, which is a second name-mangling
rule with its own collision question. That is the work #22 already did one time, redone worse.
**Namespaces exist and cannot be used**: `db.create_namespace(["legal"])` succeeds and
`list_namespaces()` lists it, but `create_table` has no `namespace` parameter in 0.37.1's Python sync
API. To keep two embedding models side by side, use a second database directory. The metadata check
makes the alternative fail loudly.

**Write semantics: `add()` when new, `merge_insert` only to replace.** This is the amended path; #27
first wrote everything through `merge_insert`. `merge_insert` is a read-modify-write against the
**whole table**, so each call costs O(table) instead of O(batch), and N Documents at batch B costs
**O(N²/B)**. That put batch size in a fight with the crash-loss window, with no comfortable value
between. It does not have to fight: Resume already knows whether a Document is new, and for a new
Document `merge_insert` buys nothing, because there is no matching row to read, update or delete. The
quadratic term leaves the common case, and crash-loss alone then bounds the batch size.

On the overwrite path, everything #27 specified still holds:

```python
# documents: key (run_label, document_name)
tbl.merge_insert(["run_label", "document_name"]) \
   .when_matched_update_all().when_not_matched_insert_all().execute(rows)

# chunks: key (run_label, document_name, chunk_index) + a Document-scoped delete
tbl.merge_insert(["run_label", "document_name", "chunk_index"]) \
   .when_matched_update_all().when_not_matched_insert_all() \
   .when_not_matched_by_source_delete(f"run_label = '{rl}' AND document_name = '{dn}'") \
   .execute(rows)
```

**The scoped delete on `chunks` is not optional, and it was measured.** Seed a Document with
`chunk_index` 0, 1, 2. Re-embed after a re-OCR that gives two Chunks. A plain upsert writes 0 and 1
and **leaves `chunk_index = 2` behind**. The Document then has a phantom third Chunk whose vector
describes text that no longer exists, and a Document vector recomputed from that table is wrong. The
scoped delete leaves every other Document alone — verified: the re-embedded Document dropped to two
Chunks, and its neighbour kept its rows.

**That predicate is a SQL string built from a filesystem path, and it must escape `'` as `''`.**
Measured with `law/O'Brien v. State.pdf`, an ordinary name in a legal corpus: the naive f-string
raises `Error tokenizing statement`. It failed loudly that time. A name shaped like `x' OR
document_name != '` would not, and `when_not_matched_by_source_delete` **deletes**. Every identifier
that goes into a LanceDB predicate comes from `Source-File`, which stays unnormalized and which
nobody in this pipeline controls.

Append was rejected, because it duplicates whenever Resume and the Sink disagree. Overwrite was
rejected too — but note that the ticket's premise, *"overwrite loses concurrent work"*, is **false**
for Lance: the format is versioned, and after `mode="overwrite"` the previous rows are still readable
through `checkout(v)`. Overwrite is the wrong default, but not for the stated reason.

**Single writer.** `merge_insert` is materially worse under concurrency than `add`. The shape that
falls out of #26, #27 and #30 is: **workers embed, one writer commits.**

**Batch size: 64 Documents, and not a flag.** It anchors to `--concurrency 64`, so one batch is about
one full sweep of in-flight requests instead of an unrelated constant. It bounds a crash to 64
Documents of GPU time. And because a Document is done only when *every* enabled Sink holds it, a
smaller batch means LanceDB lags the `.npz` tree by less, so derived Resume state after a crash is
closer to the truth. It is **not a flag**, because both costs are structural and neither is
observable during a run, so a flag would invite tuning against a metric that does not exist.

**The accepted cost, stated plainly.** One `add()` makes one data file — three appends made three
fragments in the probe. With about 2.06 rows per Document:

| batch | rows per fragment | fragments per 100k Documents |
|---|---|---|
| **64** | ~132 | ~1,562 |
| 512 | ~1,055 | ~195 |

These are small fragments, and fragmentation is a read-time cost the Consumer pays forever. **The
mitigation is LanceDB compaction after a large Invocation, and it is verified:** one
`Table.optimize(cleanup_older_than=timedelta(0))` collapsed forty fragments to one with every row
intact. It is the Consumer's call, not `embed`'s.

**Fragmentation costs read time and does not cost file descriptors.** Both full-table scans that
`embed` performs — Resume derivation at startup, and the `merge_insert` overwrite path — hold a **flat
+16** descriptors over baseline, whether the table carries 50 fragments or 600. A 600-fragment table
completes both under a soft `RLIMIT_NOFILE` of 128. So ~1,562 fragments are nowhere near any limit,
including the 1024 that containers and older distributions commonly use. **Crash-loss and read-time
cost bound batch 64, and nothing else does.**

**Indexes.** Build a **`BTree` scalar index on `document_name` in both tables** —
`create_index("document_name", config=BTree())`. `create_scalar_index` is deprecated in favour of this
form; verified on 0.37.1, where it raises `DeprecatedWarning: … deprecated as of 0.25.0` and the
`config=BTree()` form raises nothing. It is lossless, so it costs only build time, and it serves the
two lookups this design performs: the `merge_insert` key match, and Resume's question. **No vector
index.** `create_index` builds IVF_PQ, which is **lossy**: it trades recall for speed, and the correct
trade depends on a corpus size and a query pattern that live in the Consumer. Search works with no
index at all — brute force, exact, returns `_distance` — so the default is correct, not merely absent.
(`create_index(metric=...)` is likewise deprecated in favour of `config=IvfPq(...)`.) Thus **there is
no index-build phase** to show or to wait for.

### 21.10 Provenance: every fact and where it lives

| Fact | Scope | `.npz` Sink | LanceDB Sink |
|---|---|---|---|
| `model_id` | Invocation | manifest | table metadata (both tables) |
| `stored_dim` | Invocation | manifest | table metadata; also the `vector` column width |
| `native_dim` | Invocation | manifest | table metadata |
| `document_instruction` | Invocation | manifest | table metadata |
| `query_instruction` | Invocation | manifest | table metadata |
| `pooling` = `token_weighted_mean` | Invocation | manifest | table metadata |
| `chunker` = `greedy_page_pack` | Invocation | manifest | table metadata |
| `chunk_budget_tokens` | Invocation | manifest | table metadata |
| `layout` (`bare` / `labelled`) | Invocation | manifest | **not carried** |
| enabled Sinks | Invocation | manifest, outside the invariant block | see §17 |
| `paperscale_version` | Invocation | manifest `invocations[]` | `list_versions()` instead |
| `source_file` (raw) | Document | sidecar | `documents.source_file` |
| `source_digest` | Document | sidecar | `documents.source_digest` |
| `run_label` | Document | sidecar (and the path, in `labelled`) | both tables |
| `created` | Document | sidecar (ISO-8601 Z) | `documents.created` |
| `document_name` | Document | the file path itself | both tables |
| `n_chunks` | Document | derived | `documents.n_chunks` |
| `chunk_index` | Chunk | derived (`np.arange`) | `chunks.chunk_index` |
| `start_char`, `end_char` | Chunk | arrays | columns |
| `first_page`, `last_page` | Chunk | arrays | columns |
| `token_count` | Chunk | array — **a weight, not a diagnostic** | column |
| `is_partial_page` | Chunk | array | column |
| Chunk vector | Chunk | `chunk_vectors[i]` | `chunks.vector` |
| Document vector | Document | `document_vector` | `documents.vector` |

**Deliberately absent:** `normalized` (constant true), `engine` (constant `vllm`), `truncated`
(exactly `stored_dim != native_dim`), `context_length_overridden` (derivable), Chunk text, raw vector
norms, and an `invocations` log in LanceDB.

**A warning from #36:** three comparable targets declare a provenance slot and never fill it —
unstructured's `enrichment_origins` (documented for embeddings, written by no encoder), docling's
`chunking_info`, and LanceDB's own `safe_model_dump()`, which persists `"model": {}` on the idiomatic
path, asserted by the repository's own test. **A declared-and-empty slot is worse than none**, because
a Consumer trusts it. Write every field above on every path, or remove it.

### 21.11 Resume

Resume asks one question about each Document: **"do I know this name?"** There is no content-change
detection.

**State is derived from the outputs.** There is no manifest of names and no flag files. Both arguments
for a separate manifest dissolved:

- *"One cheap lookup instead of a stat per Document."* Measured on btrfs over 20,000 Documents (40,000
  files): `os.walk` collects every name in **29 ms**; a 1.3 MB JSON manifest of the same names reads
  in 3 ms. The manifest buys **26 milliseconds**, one time per Invocation.
- *"It survives a Sink on a remote."* Both Sinks are local.

What is left is the argument against it: a manifest is a second source of truth that a crash can
desynchronise from the first, and to make it trustworthy needs exactly the careful write ordering it
was supposed to save. **Derived state cannot drift, because the evidence *is* the work.**

- **`.npz` Sink** — walk `<out>` one time and collect the `.npz` paths. The write ordering makes this
  sound.
- **LanceDB Sink** — `SELECT document_name, run_label FROM documents`, served by the `BTree`.

Resume reads its state **one time at startup**. There is no per-Document lookup in the loop.

**Two Sinks, one answer: the intersection.** A Document is done when **every** enabled Sink holds it.
With one Sink the intersection is that Sink's set, so the rule needs no special case. This resolves a
conflict rather than fights it: LanceDB upserts harmlessly and the `.npz` pair rewrites harmlessly
through rename. A crash between the two Sinks leaves a Document in one and not the other. The
intersection says "not done". The next Invocation embeds it again and writes both, where one write is
a no-op and the other completes. **The gap heals itself, and it heals with nobody detecting it.**

Two consequences, both accepted:

- **The batched Sink sets the pace.** LanceDB lags the `.npz` tree by up to one batch (64 Documents),
  so a crash re-embeds up to that many Documents that the `.npz` Sink already holds. That is the price
  of batching, paid in GPU time.
- **To enable a Sink later re-embeds the corpus.** Run one time with `.npz`, then add `--lancedb`, and
  the intersection is empty. This is *correct* and expensive. The manifest's `sinks` field makes it
  loud before it happens. The vectors already sit in the `.npz` files and could be backfilled with no
  server, but that path is an optimization, not a decision this design waits on.

**The layout guard.** The run-label directory appears only when one Invocation embeds more than one
Run, so `bare` and `labelled` are two layouts over one output directory. With derived state the
failure is not silent *mixing* but silent **duplication**: the new paths match nothing, every Document
is embedded again into a parallel subtree, and the old tree is orphaned. So **`layout` is an invariant
manifest fact**. A second Invocation that would change it stops, reports both values, and tells the
operator to use the same run set or a fresh output directory. To relayout on demand was rejected: to
move a Consumer's files to save it a flag is a large act for a small convenience. The guard is
`.npz`-specific, because LanceDB has no filesystem layout and `run_label` already separates Runs.

**Documents with no usable text** must be recorded, or every Invocation retries them forever. With
derived state the record must be an **output**.

- **`.npz`** — every array at length zero: `chunk_vectors` `(0, stored_dim)`, `document_vector` `(0,)`,
  and the six per-Chunk arrays `(0,)`. Verified to round-trip with dtypes intact; the file is 2,060 B.
  A reader tells it apart in one line: `z["document_vector"].size == 0`.
- **LanceDB** — one `documents` row with `n_chunks = 0` and a **NULL** vector, and no `chunks` rows.
  Verified: a `fixed_size_list` column accepts NULL, reads back as `None`, and **vector search skips
  the row**.

A zero vector was rejected. It is not a unit vector, nothing else in the store is anything but a unit
vector, and it would sit in a search index and look like data. The empty output says what happened.
The two Sinks differ in *representation* and agree in *meaning* — NULL exists in Arrow and not in
`.npz`. Empty is a **`run` outcome, not an `issue`**.

**`--no-resume` re-embeds and overwrites, and deletes nothing.** This **diverges from the OCR
precedent on purpose**. There, `_wipe_workspace_progress` (`pipeline.py:1147`) `rmtree`s `results`,
`done_flags` and `worker_locks`. That is safe because an OCR workspace is scratch. **An embed output
is the deliverable**, and the Consumer may already read it, and one pair of LanceDB tables holds
several Runs — so a wipe would need a scoped delete built from unnormalized paths, which is exactly
where the `O'Brien` quoting hazard lives. Both Sinks are idempotent, so to ignore prior state is
enough: every Document is embedded again, the `.npz` pair is rewritten by rename, and LanceDB replaces
by `merge_insert`. The end state equals a wipe, except in one respect: outputs whose Documents left
the input are not removed. To remove them is a different operation from "ignore prior progress". The
existing help text — *"Ignore prior progress and reprocess the workspace from scratch"* — describes
this behaviour accurately. It is the OCR-side implementation that goes further than it says. **State
the divergence in user docs, or it reads as a bug.**

The user-facing warning that standing decision 7 requires is in §11.6, ready to place verbatim.

### 21.12 The pipeline, end to end

**Startup, in order.** Several steps are preconditions for the next, and three can stop the Invocation
before any GPU work.

1. Parse and validate flags. Check run labels against `[A-Za-z0-9._-]+`. Reject `--no-npz` without
   `--lancedb`.
2. Build the **Adapter** from `--embed-model`. Validate `--embed-dim` inside `[min_dim, native_dim]`.
3. Read every **Record** from every Run, derive **Document names**, and run the collision check.
   **Fatal on collision.**
4. Ask `GET /v1/models` for `.id` (the served model id, for the panel header and `model_id`) and
   `.max_model_len`.
5. Compute `validated_context_length` = `min(card, server)`. Then apply `--context-length` if given:
   reject above the server, warn above the card.
6. Compute `chunk_budget` = `validated_context_length − tokens(document_instruction) − 64`. The
   Instruction's count is one `/tokenize` call, or 0 for an empty string.
7. Compute the effective request budget = `max(--request-tokens, chunk_budget)`, with an explicit log
   line when it is raised.
8. Probe the output dimension with one cheap `/v1/embeddings` request. Assert the width equals
   `adapter.native_dim`. **Stop on mismatch** and report both numbers.
9. Open the Sinks. Compare the `.npz` manifest and the LanceDB table metadata against this
   Invocation's invariants. **Stop on disagreement** and report both values. Warn if the enabled-Sink
   set changed.
10. Construct the reporter with `title=f"paperscale embed · {served_model_id}"`. This is after step 4
    on purpose.
11. Derive Resume state: one `os.walk` and one `SELECT`. Intersect. Log the skip count.
12. Run.

**The unit of work is the Document.** Three closed tickets forced this before anybody asked: #26
writes two files per Document, #27 keys its upsert on the Document, and #28 defines done as *every
Sink holds this Document*. A Chunk-level unit would need a completion state that nothing writes. The
objection is real and recorded, not solved: a 300-page Document and a 1-page Document are wildly
different units, so one progress bar under-reports early and over-reports late. #29 accepted that when
it chose one bar over two.

**Per Document, in order:**

1. `POST /tokenize` on the whole text — one call.
2. If it fits, one Chunk. If not, the Overflow path.
3. Pack Chunks into `/v1/embeddings` requests bounded by the effective request token budget, **mixing
   Documents**.
4. Slice each returned vector to **Stored dimension** and re-normalize it.
5. Pool the Chunk vectors into the Document vector. `n_chunks == 1` short-circuits to a copy.
6. The `.npz` Sink writes sidecar, then `.npz`, each with a temporary name and a rename — **two
   creates and two renames per Document**.
7. Hand the Document to the single LanceDB writer, which commits in batches of 64.

**Batching uses a token budget, never a count of Chunks.** A fixed count is meaningless here. Greedy
packing means one Chunk can be a short page and the next can be forty-five dense pages at the full
budget, so "16 Chunks" means from a few hundred to half a million prefill tokens, and the operator
cannot reason about it. A token budget is free to compute, because every Chunk's exact count is known
before packing.

Two properties follow:

- **The floor is forced, and it is enforced by raising, not by rejecting.** The budget can never be
  smaller than one Chunk's maximum, or a full-size Chunk could not be sent at all. So the effective
  budget is `max(--request-tokens, chunk_budget)`, with a log line. To reject would reject the default
  configuration, because the 32,000 default sits **below** the floor for *both* pinned families
  (Qwen3's budget is 32,704, which is 704 tokens above the default; Nemotron's Instruction would have
  to exceed 704 tokens to land below 32,000, and it is about three). To ignore the floor silently
  would make a full-size Chunk unsendable, which is the real fault. To raise is always safe: it only
  permits a larger request.
- **Requests mix Documents, and they must.** The common case is one Chunk of a few thousand tokens, so
  a refusal to mix would make almost every request tiny.

**Corrected rationale, because the obvious mental model is wrong.** #30 argued the token budget from
the premise that a request is a batch the server processes together. **It is not.** #36 verified by
hand that vLLM fans an `input` array of N texts into **N independent engine requests** — one
`engine_client.encode()` per element, merged with `merge_async_iterators`. There is no analogue of
`--max-client-batch-size`. The decision stands and the reasons above stand. What fails is the
mechanism:

- Request batching amortizes **HTTP round trips only**. It does not help the engine batch.
- **Concurrency fills the engine, not batch size.** That makes `vllm:num_requests_waiting` a *more*
  direct instrument than #30 assumed.
- **Split-on-failure is cheaper than #30 assumed.** To re-issue one Document at a time costs more HTTP
  round trips and **identical** engine work.

**Concurrency: `--concurrency 64`, fixed.** Derivation from discovery is not available, because vLLM
publishes no batch ceiling. The OCR side's `--max_concurrent_requests 500` does not transfer: there
one request is one page image, and here one request can carry a hundred thousand prefill tokens, and
vLLM's scheduler already batches internally. The client's job is to keep the server's queue non-empty,
not to flood it.

The challenge, and why the simple thing stayed: #36 found that Vespa's feed client *adapts* with no
published ceiling — a `DynamicThrottler` that optimises `throughput / inflight^0.3` across 128
log-scale buckets with an upward-skewed random walk. It measures the server instead of asking it,
which answers #30's stated reasoning directly. Two things temper it. Vespa feeds a distributed store
whose aggregate capacity is genuinely unknown, and this feeds one vLLM server whose queue depth is
directly observable. And **pyvespa — the closer analogue — wires its `AdaptiveThrottler` into queries
only, never into feeding.** A control loop here would be a new failure mode — oscillation, and a
throughput number that moves for reasons the operator cannot see — bought against a signal that is
already published.

**What changes is that the panel stops being passive.** When `vllm:num_requests_waiting` stays above
zero for a sustained window (about 60 s), the event pane emits:

```
queue depth sustained; --concurrency 64 may be too high for this server
```

Three deliberate properties: it **names the flag**, so the advice is actionable with no design
document; it is **advisory only**, so nothing oscillates; and it needs **no new plumbing**, because
`vllm:num_requests_waiting` is already parsed and surfaced.

**`/tokenize` gets its own concurrency bound, not `--concurrency` slots.** A producer stage
(tokenize, chunk, pack) feeds a consumer stage (embed), and `--concurrency` bounds the consumer only.
To share the slots would idle the GPU during CPU-side work. **The bound is `(concurrency * 3) // 2`** —
96 at the default 64. Tokenize never reaches the GPU, so it can run ahead without contention, but it
is still HTTP against the same API server process, so it is not free. Integer arithmetic instead of
`1.5 *` stays exact for odd inputs (65 -> 97) and degrades sensibly at the bottom (1 -> 1). It is
unmeasured, on the same footing as every other constant. At the defaults that is **160 concurrent
sockets** (64 + 96), which is comfortable against a 1024 soft `RLIMIT_NOFILE`.

**The wire format.**

- **`encoding_format: base64`, with `embed_dtype` and `endianness` both sent explicitly.** Measured by
  JSON-encoding a real float vector against base64 of the same float32 bytes: 768-wide, 15,976 B ->
  4,096 B (**3.90x**); 4096-wide, 84,994 B -> 21,848 B (**3.89x**). That is nearly 4x on the wire, and
  it compounds with client-side MRL, because every response arrives at native width. Parsing also
  becomes one `frombuffer` instead of thousands of float parses. vLLM honours it — confirmed by a
  source read at 0.27.1, not by a live call. The body is:

  ```json
  {"model": "…", "input": ["…"], "encoding_format": "base64",
   "embed_dtype": "float32", "endianness": "little"}
  ```

  and the decode is `np.frombuffer(base64.b64decode(s), dtype="<f4")`. **Neither extra parameter may
  take its default.** `endianness` defaults to `"native"`, which is the *serving host's* byte order —
  never stated in the response, while the client's `frombuffer` uses its own. A byte-reversed
  IEEE-754 vector is finite, correctly sized and normalizes to unit length, so nothing downstream
  would catch it. `"little"` makes the server byteswap only if it must, and it is a no-op on every
  host this design targets. `embed_dtype` defaults to `float32` today, but `float16` and the two fp8
  values are lossy and return fewer bytes, which `"<f4"` would decode as a short vector instead of an
  error. `float` stays the fallback, and the two are mathematically identical, so the *format* is a
  throughput decision only — **the two extra parameters are not.**
- **`truncate_prompt_tokens` is never sent.** vLLM's default on oversized input is to *error*, which
  is the safe case. So the enforcement is that the parameter never appears in a request body, not that
  some flag is set. This is why silent truncation disqualified two engines.
- **`dimensions` is never sent.**

**Retries and the failure taxonomy.** `embed` mirrors `src/paperscale/evaluation/pplx.py`, not the OCR
path. It is the same workload — prefill-only against vLLM — its delay is bounded, and it raises
instead of calling `sys.exit(1)` inside a worker. (`try_single_page_with_backoff` has one axis, an
**uncapped** delay that reaches 85 minutes at attempt 10, and `sys.exit(1)`.)

| axis | budget | delay | on exhaustion |
|---|---|---|---|
| fd exhaustion (`EMFILE`/`ENFILE`) | unbounded, **consumes no attempt** | `min(2**n, 30)` | n/a — self-resolves |
| connection error | 6 | **`uniform(0, min(10 * 2**(n-1), 120))`** — full jitter | raise; terminal for the Invocation |
| bad response / timeout | **`--max-request-retries 8`** | `min(2**n, 30)` | raise; terminal for the Document |

**Full jitter on the connection axis is new — paperscale has no jitter anywhere today.** At
`--concurrency 64` a server restart fails all 64 in flight *from one cause*, and with no jitter all 64
retry in the same instant against a server that is still booting. The response axis fails per request
for per-request reasons, so lockstep is not a risk there, and extra variance would only slow it.

**Terminal without retry, for the Document:** a `400` context overflow, and `413`.

**A context-overflow 400 is a bug signal, not a routine outcome.** Chunks are sized from an untruncated
token count exactly so it cannot happen. If it happens, either the Adapter's card context length is
wrong or the packer is. The Invocation must say so loudly at the end, and that is what the `oversize`
counter is for.

**A failed request splits before it fails Documents.** Because requests mix Documents, one poison
Document would otherwise take down every Document that shares its request. On a terminal failure of a
multi-Document request, re-issue its Documents **one at a time**, and record as failed only those that
fail alone. #30's own framing — *"one bad PDF must never end a run"* — is the reason. #36 found this is
genuinely uncommon: **blast radius is the batch, not the record, everywhere except olmOCR.**

**A `/tokenize` failure fails the Document, not the Invocation.** With no token count a Chunk cannot
be sized, and the design depends on vLLM erroring on overflow rather than truncating, so to proceed on
a guess is unsafe. It counts in `failed`. Tokenize shares the taxonomy and the backoff — same client,
same server, same failure modes.

**The end-of-run report.** **An Invocation that ends with any failed Document exits non-zero.** An
Invocation that exits 0 after it quietly failed 3% of a corpus is the same class of silent wrongness
this design guards against everywhere else. The report prints counts by outcome — embedded, skipped,
empty, failed, oversize — and, when `oversize` is not zero, states plainly that it indicates a chunker
or context-length bug, not a corpus problem. Failed Documents go into
`<out>/paperscale-embed-failures.txt`, one **Document name** per line. It is a convenience, not state:
Resume derives from the outputs, so a failed Document has no output and is retried automatically. The
file is rewritten each Invocation and shares the reserved prefix, so the collision guard covers it.

### 21.13 The TUI panel

**Variant A** — the shape the repository already renders: equal-weight counters in horizontal stat
groups, a single phase bar, the event pane below, on the existing `RichReporter` and `Phase` in
`src/paperscale/tui.py`.

Two alternatives were prototyped against the **real** `RichReporter` (branch `prototype/embed-panel`,
commit `b2757f5`, local only) and rejected on what the prototype showed:

- **Variant B, saturation-first with a verdict string.** At 80 columns the verdict cropped to
  `SATURATED (server-bound` and its key cropped to `server queu`. **The one line the variant exists to
  show is the first thing lost.** A judgement that survives only on a wide terminal is worse than raw
  numbers, which stay legible when crushed.
- **Variant C, a second bar for Chunks.** The Chunk total is unknowable until every Document is
  chunked, and an indeterminate bar (`total=None`) draws as `━━━━━━━━━━          ━━━━━━━━━━` — two
  bars next to a real one.

**The groups.**

- **`run`** — `documents`, `chunks`, `skipped`, `empty`
- **`server`** — `dim`, `tok/s`, `in-flight`
- **`issues`** — `failed`, `retrying`, `oversize`

`empty` belongs to **`run` alone**. #29 listed it in both. #28 made a zero-Chunk Document a recorded
outcome with a real output, not a problem. The freed `issues` slot went to `retrying`, which is what an
operator wants when throughput drops and nothing has failed yet.

| Field | Group | Source |
|---|---|---|
| `documents` | `run` | client counter — Documents written to every enabled Sink |
| `chunks` | `run` | client counter |
| `skipped` | `run` | client counter — Resume skips, from derived state |
| `empty` | `run` | client counter — zero-Chunk Documents |
| `dim` | `server` | `stored/native`, for example `768/4096` |
| `tok/s` | `server` | **`Rates.prompt_tps`** |
| `in-flight` | `server` | `<client outstanding>/<vllm:num_requests_waiting>`, for example `16/3` |
| `failed` | `issues` | client counter — Documents that exhausted retries |
| `retrying` | `issues` | client counter — requests in backoff now |
| `oversize` | `issues` | client counter — Chunks rejected as too long; must be 0 |

**`gen_tps` is structurally zero and must not be displayed.** Embeddings are prefill-only, so
`vllm:generation_tokens_total` never moves. This is the one place the embed panel reads a *different
field* from the OCR panel, not merely a different label, and it is impossible to notice at run time:
the wrong field reads zero, which looks like an idle server instead of a bug. `Rates` already exposes
both, so nothing needs writing. It needs to be correct one time.

**`in-flight` carries two numbers in one slot.** The client figure is what `--concurrency` controls.
The server figure (`Rates.waiting`, from `vllm:num_requests_waiting`) is what says the flag is too
high. Note this is *waiting*, not the OCR panel's *running*: queue depth is the signal, admitted
requests are not. Either half renders `-` when absent, never `0`, per `format_rate`'s existing rule.
**It counts `/v1/embeddings` requests only.** `/tokenize` is served CPU-side and never enters the
engine scheduler, so it cannot make the queue it would be compared against. **Tokenize gets no panel
row at all**; it belongs in the end-of-run report if anywhere.

Every metric name that `src/paperscale/vllm_stats.py` maps was verified present on a live vLLM
0.27.2rc1: `prompt_tokens_total`, `generation_tokens_total`, `prefix_cache_hits_total`,
`prefix_cache_queries_total`, `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`.

**`model` moves to the header.** It cannot fit as a row. `_VLLM_ROWS`'s comment records the
arithmetic: at 80 columns each of three panels gets 22 cells of content, and the key column takes the
widest label plus 2 of padding. The OCR panel's widest label is `running` (7), which leaves 13 cells
for the value. **embed's widest label is `in-flight` (9), which leaves 11.** The two pinned ids are
`Qwen/Qwen3-Embedding-8B` (23 characters) and `nvidia/Nemotron-3-Embed-1B-BF16` (31). Both would render
as `Qwen/Qwen…` forever. Basename-only does not fix it (`Nemotron-3-Embed-1B-BF16` is 24). A short
Adapter tag destroys the row's purpose, because the value comes from `/v1/models` `.id` exactly so it
reports what the server *is serving*, not what paperscale *believes*. The header is full pane width,
and the model id is constant for a whole Invocation. Precedent exists at `pipeline.py:1308`.

**One implementation condition: ask `/v1/models` *before* you construct the reporter**, so the header
shows the id the server returned and not the string the operator typed. The `native_dim` assertion does
not cover this, because it checks the *dimension*, which cannot separate two models of equal width.

**This also removes a height cliff, which is why it is the right trade and not merely a fix.**
`_layout_budget` grows sections in the order bars, events, stats, and `_stat_columns` truncates each
group from the tail. With four rows in `run` and four in `server`, stats reach a fourth row only after
events grow from `MIN_EVENT_ROWS` to `MAX_EVENT_ROWS`. Verified by running `_layout_budget(h, 4, 1)`:

```
height=17  stat_rows=3
height=18  stat_rows=4
```

So below 18 rows a four-row `server` group drops its last row — `in-flight`, the saturation signal.
That is exactly the class of failure variant B was rejected for. A three-row `server` group is complete
**wherever stats render at all**, because `MIN_STAT_ROWS` is 3. The freed fourth slot stays empty; a
three-row group in a four-row panel renders one blank line, which costs nothing.

**`embed` gets its own push function, and `vllm` is renamed `server`.** The push function goes in
`src/paperscale/embed/`. It is not a parameterised `push_vllm_stats`, because the mismatch is in the
*inputs*, not the row names. `push_vllm_stats(rep, stats, poller)` can reach exactly `stats.rates()`
and `poller.available`. Of embed's four original `server` rows, only `tok/s` is reachable from that
pair: `model` comes from `/v1/models`, `dim` from the flag and the Adapter, and the client half of
`in-flight` does not exist anywhere yet. To parameterise it would thread a model id, two dimensions and
a live counter through a function whose whole premise is that it needs none of them. **`push_vllm_stats`
is not edited.**

What *is* shared, imported from `vllm_stats`:

- `format_rate` — absent renders `-`, never `0`, because zero is a measurement.
- `Rates` and the poller — the scraping side is engine-specific and already correct.
- **The fixed-row-set discipline.** `set_stat` can add and overwrite, and it can never take a row off
  the panel. So **every branch must write every row**, or a stale row from the other branch survives on
  screen. `_VLLM_ROWS`'s comment records the bug that taught this: a permanent `status: unavailable`
  next to live token rates. The embed function inherits the rule, not the row tuple. Its fixed row set
  is `("dim", "tok/s", "in-flight")`.

**The rename is six lines.** One concept keeps one name. A box titled `server` stays honest against any
server, and a box titled `vllm` becomes a lie the first time one is not. The code keeps the vendor
where the vendor is true: `vllm_stats.py`, `push_vllm_stats`, `_VLLM_ROWS` and `VLLMStats` are all
genuinely vLLM-specific, because they parse vLLM's metric names.

| file | line | change |
|---|---|---|
| `src/paperscale/vllm_stats.py` | 351 | `group="vllm"` -> `group="server"` |
| `src/paperscale/tui.py` | 266 | `("run", "vllm", "issues")` -> `("run", "server", "issues")` |
| `tests/test_tui.py` | 347, 422, 505 | three group-name string literals |

The only OCR-visible difference is the panel's title. Every OCR row, `VLLMStats`, `Rates`, `Snapshot`,
`_CANDIDATES`, the poller and both call sites stay unchanged.

**The `tui.py` prerequisites: one discharged, one settled, one open.**

1. ~~Group ordering.~~ **Discharged by the rename.** `_stat_columns()` hardcodes `order = [g for g in
   ("run", "vllm", "issues") if g in self._stats]` and appends the rest, which would render embed's
   groups as `run | issues | server` — `issues` above the group that carries throughput. The rename
   satisfies the prerequisite exactly as worded, and the ordering list carries no redundant entry
   forever.
2. **Event-pane padding — open.** `_layout_budget()` gives surplus height to events, so an embedding
   Invocation — nearly silent until something fails — renders three log lines and nine blank rows. The
   OCR pipeline is chatty enough that this never showed. Either cap growth when the log is short, or
   let stats absorb the surplus.
3. **Resumed skips must read differently from work performed — settled here.** In the prototype the
   `run` group showed `documents 407` beside a bar that read `4309/12480`: the counter is Documents
   *embedded*, the bar is Documents *done*, including 3,902 resumed skips. Both correct, together
   confusing. Because there is no change detection, **a large silent skip count is exactly the symptom
   of a stale output directory**. It must be conspicuous, not buried as one counter among four.

   **Mechanism: the bar counts only Documents that will actually be embedded.** Resume state is derived
   one time at startup — one `os.walk`, measured at 29 ms over 20,000 Documents — so the skip count is
   known *before* the bar is constructed. Set the total to `corpus - skipped`, and every unit in the
   bar is work performed. Four properties earn it:
   - **It is structural, not a counter.** #29 rejected a burial among four counters, which is what a
     `skipped` row is. A change to what the bar *counts* cannot be overlooked.
   - **In the failure case it reads `0/0`.** Point `embed` at a stale output directory and the bar
     total is zero: the Invocation visibly has nothing to do. That is the exact symptom that goes
     undetected otherwise, surfaced with no threshold and no judgement call.
   - **The rate stays honest.** With skips inside the bar, the first frame reads 31% complete and then
     crawls.
   - **It invents no number.** A "warn above X% skipped" rule needs a threshold nobody can justify, and
     #25 already refused a Chunk-count sentinel on that ground.

   The `skipped` row stays and carries the number. One startup log line states the split. The
   end-of-run report prints counts by outcome. The phase description cannot carry the split:
   `TextColumn` is `no_wrap` with crop and holds about 20 to 25 cells at 80 columns, after the bar,
   `MofNCompleteColumn` and `TimeElapsedColumn` take theirs.

**New plumbing beyond those three:** a client-side outstanding-request counter over `/v1/embeddings`,
incremented on send and decremented on completion; `/v1/models` probed before the reporter is
constructed; and the sustained-queue advisory.

### 21.14 The CLI surface

`paperscale embed` is shaped like `paperscale evaluate` — **hyphenated throughout**. It writes `.npz`
by default. `--lancedb PATH` opts the table Sink in.

| Flag | Default | Notes |
|---|---|---|
| `--run LABEL=PATH` | *required* | repeatable; shares `_parse_runs`, plus embed-only label validation |
| `--out PATH` | `./vectors` | the `.npz` tree, the manifest, and the failures file |
| `--embed-model NAME` | *required* | selects the Adapter, mirroring `--ocr-model` |
| `--embed-url URL` | `http://localhost:8000` | mirrors `--pplx-url` |
| `--embed-dim N` | `768` | client-side MRL slice; rejected outside `[min_dim, native_dim]` |
| `--context-length N` | `min(card, server)` | rejected above the server's `max_model_len`; warns above the card |
| `--api-key KEY` | `None` | mirrors the OCR side's `--api_key`, rehyphenated |
| `--lancedb PATH` | *unset* | presence enables the table Sink |
| `--no-npz` | off | disables the file Sink |
| `--concurrency N` | `64` | anchored to `--pplx-concurrency` |
| `--request-tokens N` | `32000` | anchored to `_MAX_TOKENS_PER_CHUNK`; **raised to the Chunk budget when below it** |
| `--max-request-retries N` | `8` | bounds the bad-response/timeout axis only |
| `--no-resume` | off | re-embed and overwrite; **deletes nothing** |
| `--tui` | off | "(needs the 'tui' extra)" |
| `--tui-poll-interval S` | `5.0` | mirrors `evaluate` |
| `--disk-logging PATH` | `None` | mirrors `evaluate` |

**Where the numbers come from.** Three defaults shipped marked **explicitly unmeasured**, because no
live embedding server existed to measure against: the per-request token budget, the concurrency limit,
and the retry ceiling. #35 found an in-repo sibling that runs a **prefill-only** workload against vLLM
on the same hardware. That is a better basis than an invented number, and it makes the embed flags
consistent with a CLI the operator already knows:

- **`--pplx-concurrency`, default 64** — chunk requests in flight against a vLLM server.
- **`_MAX_TOKENS_PER_CHUNK = 32_000`** (`pplx.py:58`), whose docstring gives the reason: *"smaller
  prompts let vLLM co-schedule far more of them."*
- **`--max_page_retries`, default 8** on the OCR side.

Perplexity scoring is not embedding. So these are **calibrated starting points from a comparable
workload, not measurements of this one**. Describe them that way in user docs, and name
`vllm:num_requests_waiting` as the instrument for concurrency tuning. The ~60 s advisory window sits on
the same footing. (#35 also recorded a "pleasing consequence" — that `--request-tokens 32000` sat
exactly on the one-Chunk floor. That arithmetic assumed `SAFETY_MARGIN = 868`. At 64 it **inverts**,
and the default now sits about 704 tokens *below* the floor for both families. That is why the floor is
enforced by raising.)

**`--embed-model` is required, where `--ocr-model` is not.** The two flags do different damage when
wrong: **OCR models make text that is roughly comparable across models; embedding models make vectors
that are meaningless across models**, and both Sinks bake the model into the output's identity. A
default would pick the semantics of a whole corpus in silence.

**Run-label validation is enforced, in `embed` only.** Labels are validated against `[A-Za-z0-9._-]+`,
and a rejection names the offending character. Today `--run 'legal/2024=…'` silently creates a nested
directory, because the label goes into a filesystem path while `_parse_runs` (`cli.py:66`) only strips
whitespace and checks non-empty and unique. **Sanitizing was rejected**, because it invents a second
name-mangling rule with its own collision question — refused one time by #22 and again by #27. **The
check lives in the `embed` handler, not in `_parse_runs`**, because that function is shared with
`evaluate`, where labels go into a SQLite column and never needed a constraint. To tighten it globally
would start rejecting inputs that `evaluate` accepts today. Breaking an existing subcommand's contract
to serve a new one is the wrong trade — but the charset is worth *recommending* in `evaluate`'s docs,
so labels stay portable.

**Sink selection: the path is the opt-in.** `.npz` writes by default. `--lancedb PATH` enables the
table Sink. `--no-npz` disables the file Sink. LanceDB needs a database path anyway, so a separate
boolean would be a second way to say the same thing. **At least one Sink must be live: `--no-npz`
without `--lancedb` is rejected.** This is correctness, not convenience, because a Document is done
only when every enabled Sink holds it — so **to enable a second Sink later re-embeds the whole
corpus**. The manifest's `sinks` field makes that loud first.

**`--no-resume` means something different here.** It re-embeds and overwrites and **deletes nothing**,
where the OCR pipeline's `--no-resume` `rmtree`s `results`, `done_flags` and `worker_locks`. **State
this in user docs, or it reads as a bug.** It follows `evaluate`'s bare `store_true`, not the OCR
pipeline's mutually-exclusive pair. The two existing subcommands already disagree; to match the closer
sibling is the smaller inconsistency.

**Flags deliberately not added.** Each looks like it should exist, and a closed decision answers each.
Recording them stops somebody adding them "for completeness" during implementation.

- **No `--embed-served-model`.** The served model id is *asked* from `/v1/models`.
- **No `--chunk-tokens`.** The Chunk budget is *derived*, not chosen. `--request-tokens` bounds how
  many Chunks share a request; it does not resize a Chunk. To conflate them would let an operator
  change what a Chunk is between Invocations, which the manifest then records as a changed
  `chunk_budget_tokens` and refuses.
- **No `--truncate` of any spelling.** Enforcement is that `truncate_prompt_tokens` never appears in a
  request body.
- **No `--jobs`.** `evaluate` needs one for CPU-bound scoring. Here the only client-side compute is a
  slice and a normalize of 768 floats.
- **No LanceDB batch-size flag.** Both costs are structural and neither is observable during a run.
- ~~**No `--max-context`.**~~ **Overturned.** The rejected flag would have taken the server's number as
  authoritative. `--context-length` can only name a number the server already agreed to serve.

### 21.15 Packaging

```toml
[project.optional-dependencies]
tui = ["rich (>=13.0.0,<15.0.0)"]
embed = ["numpy", "lancedb", "pyarrow"]
```

**None of numpy, lancedb or pyarrow is a paperscale dependency today.** All three are absent from
`dependencies`, from `[project.optional-dependencies]` (which holds only `tui`) and from the dev group.
Everything measured was against **lancedb 0.37.1 / pyarrow 25.0.1 / numpy 2.5.2**.

**One extra per feature** is the `tui` precedent, and `embed` is one feature. A finer split — so
`.npz`-only users skip LanceDB and pyarrow, which is most of the weight — tracks the Sink default
exactly and is a real saving. It was rejected because **you can make the split later without breaking
anyone, and you cannot undo the merge**, and because a second extra is a second thing to explain in
every install instruction.

**Imports stay inside the handler**, as `evaluate` does with wordfreq and rapidfuzz, so `paperscale`
for OCR never pays for numpy. This is also why `embed` is its own package:
`src/paperscale/models/__init__.py` eagerly imports all nine OCR Adapters, so embed Adapters placed
there would pull embed code into every OCR Run.

### 21.16 The two probed claims

The record carried both forward deliberately. Both are now probed. **Neither probe was a live call
against a running server.** Read what each one establishes before you lean on it.

**LanceDB compaction — confirmed by measurement.** Measured against lancedb 0.37.1 / pyarrow 25.0.1, in
a throwaway environment, never installed into the project.

`Table.optimize(*, cleanup_older_than: timedelta | None = None, delete_unverified: bool = False,
retrain: bool = False)` is the operation, and it does merge small fragments. Forty 64-row `add()` calls
into a `fixed_size_list<float32, 768>` table left **forty data files** under `documents.lance/data/`.
One `optimize(cleanup_older_than=timedelta(0))` left **one**, with all 2,560 rows intact and the
on-disk bytes essentially unchanged. Compaction is one of three jobs `optimize` covers; the others are
to prune old versions and to add new data to existing indexes.

Three things the signature gives that the prose does not: `retrain` is still accepted but is deprecated
and unused; the call returns **`None`** on this version, so there are no stats to log; and
`compact_files()` survives as compaction alone. **Call `optimize()`** — every `add()` leaves an old
version behind as well as a fragment, and to prune those is the other half of the win.

So the batch-64 trade stands as written. The mitigation is real, it is one call, and it is the
Consumer's to run after a large Invocation. Nothing in `embed` changes.

**A third finding, from a question the record never asked: fragmentation costs no file descriptors.**
The worry was that ~1,562 fragments would consume one fd each during the two operations that scan the
whole table — Resume derivation at startup, and the `merge_insert` overwrite path — and would exhaust a
1024 soft `RLIMIT_NOFILE` alone, before any of `embed`'s 160 sockets. Measured by sampling
`/proc/self/fd` from a watcher thread **during** each operation, every 0.5 ms:

| fragments | fds at rest | peak during scan | delta |
|---|---|---|---|
| 100 | 14 | 30 | +16 |
| 400 | 14 | 30 | +16 |
| 800 | 14 | 30 | +16 |

| fragments | Resume scan (2 cols) | full scan (incl. vector) | `merge_insert` |
|---|---|---|---|
| 50 | +16 | +16 | +13 |
| 300 | +16 | +16 | +16 |
| 600 | +16 | +16 | +16 |
| 1, after `optimize()` | +1 | +1 | +1 |

**Flat, and bounded, not linear.** Lance reads fragments through a fixed-width IO pool and closes them
as it goes, so the constant is the pool and not the table. Confirmed the decisive way rather than by
extrapolation: with the soft `RLIMIT_NOFILE` forced down to **128**, a 600-fragment table still
completes all three operations without error, and peaks at 30 open descriptors in total. (This host's
own limit is 1,048,576, which is exactly why a linear relationship would not have been noticed here —
hence the forced-limit run.)

So the whole-process budget at the defaults is about **200 fds**: 64 embedding sockets, 96 tokenize,
about 30 for LanceDB, about 10 baseline, against a 1024 soft limit, in one process. **Crash-loss alone
bounds batch size.** The fd-exhaustion retry axis inherited from `pplx.py` is therefore defensive, not
load-bearing: normal operation cannot reach it, and its test must force `EMFILE` through the fake
transport.

**A second unasked finding: the index guidance is accurate.** `create_scalar_index("document_name")`
raises `DeprecatedWarning: create_scalar_index is deprecated as of 0.25.0. Use create_index() with
config=BTree()/Bitmap()/LabelList() instead.`, and `create_index("document_name", config=BTree())`
builds the `BTree` with **no** warning. The form this document specifies is the surviving one.

**`encoding_format: base64` — confirmed by source read.** **No embedding server was available to call.**
`gigaspark:8000` serves a generative model, and `/v1/embeddings` returns 404 there, because the pooling
route mounts only for pooling-task models. So this was settled the way #36 says to settle things —
**read signatures, not prose** — against the vLLM installed at
`~/.local/share/uv/tools/vllm-omni/…/site-packages/vllm`, whose `_version.py` reads **0.27.1**.

**The route is wired, and it is the route `embed` calls.**
`entrypoints/pooling/embed/api_router.py` mounts `POST /v1/embeddings` with body type
`EmbeddingRequest`, a union whose every member — including `EmbeddingCompletionRequest`, the
`{"input": [...]}` shape `embed` sends — inherits `EmbedRequestMixin(EncodingRequestMixin)`.
`ServingEmbedding._build_openai_response` branches on `request.encoding_format` and, for
`"float" | "base64"`, calls `_openai_json_response`, which builds its encoder from
`get_pooling_output_encoder(...)`. `EmbeddingResponseData.embedding` is typed `list[float] | str`; the
`str` arm **is** the base64 string. This is not some other pooling route that shares a mixin.

| alias | legal values | default |
|---|---|---|
| `EncodingFormat` | `float`, `base64`, **`bytes`, `bytes_only`** | `float` |
| `EmbedDType` | `float32`, `float16`, `bfloat16`, `fp8_e4m3`, `fp8_e5m2` | `float32` |
| `Endianness` | `native`, `big`, `little` | `native` |

All three live in `vllm/utils/serial_utils.py`. `EncodingFormat` is wider than the record assumed.
`bytes` and `bytes_only` return a raw `StreamingResponse` of concatenated tensors with the shapes in a
`metadata` header instead of JSON. **They are deliberately not adopted**: they buy a little over
base64's already ~3.9x saving, in exchange for a second parsing path and a second failure mode, and the
response stops being a JSON body the rest of the client can treat uniformly.

**What the bytes are.** `tensor2binary(tensor, embed_dtype, endianness)` casts to the dtype, flattens,
views to the numpy-safe view dtype, byteswaps **only** when `endianness` is neither `"native"` nor
already the server's own `sys.byteorder`, and returns `.tobytes()`. `encode_pooling_output_base64` then
`pybase64.b64encode`s exactly that. For `embed_dtype="float32"` the payload is `native_dim × 4` raw
IEEE-754 single-precision bytes — **`native_dim`, not `stored_dim`**, because the MRL slice is
client-side. There is no length prefix, no shape and no dtype tag. 768-wide is 3,072 B, which base64s
to **4,096 characters** — exactly the number §12.5 measured independently, and the 4096-wide figure
matches too.

**Two parameters the record never anticipated, and `embed` must send both explicitly.**

- **`endianness: "little"`, never the default.** `"native"` means *the serving host's* byte order,
  decided on the server and stated nowhere in the response, while paperscale's `np.frombuffer` uses
  *the client's*. On a big-endian server every decoded float is byte-reversed — and a byte-reversed
  IEEE-754 word is usually still an ordinary finite number, arrives in the right count, and normalizes
  to unit length. So **nothing downstream reliably detects it**, and both Sinks fill with plausible
  garbage. Sending `"little"` makes the server byteswap if and only if its own order is not little. On
  the x86-64 and aarch64 hosts this design targets it is a no-op, which is why it costs nothing to make
  unconditional.
- **`embed_dtype: "float32"`, likewise explicit.** The default *is* `float32` today, so this is belt
  and braces. But the other four values are lossy, and `float16` returns **half** the bytes, which
  `np.frombuffer(..., "<f4")` decodes as a vector of half the width instead of raising. Pin it, and
  keep the width check as the backstop for the day a default moves.

**The request body:**

```json
{"model": "…", "input": ["…"], "encoding_format": "base64",
 "embed_dtype": "float32", "endianness": "little"}
```

**The decode is `np.frombuffer(base64.b64decode(s), dtype="<f4")`** — the explicit `"<f4"`, never a
bare `np.float32`, which is the *client's* native order and would agree with the bug instead of
catching it.

**What is still unverified, precisely.** Two things, both cheap to close with one request against any
pooling-task server:

1. **That 0.27.2rc1 carries these fields.** This read is 0.27.1, and the metric names were verified
   against **0.27.2rc1**, so the document already spans two vLLM builds. `encoding_format` is
   long-standing and safe. `embed_dtype` and `endianness` are newer and **were not checked against
   0.27.2rc1**.
2. **That an older or newer build fails loudly if it lacks them — it does not.** vLLM's
   `OpenAIBaseModel` sets `extra="allow"` and only `logger.debug`s the ignored keys. So to send the two
   fields is always *safe*, but on a build that does not know them, `endianness` reverts to `"native"`
   in silence and the guarantee evaporates. **A live check must assert on the response, not on the
   absence of an error:** send one known input two times, once with `encoding_format: "float"` and once
   with `base64` + `float32` + `little`, then assert the decoded vectors are bit-equal and
   `len(b64decode(s)) == native_dim * 4`.

**A method note that belongs in the plan: read signatures, not prose.** Several comparable targets'
docstrings contradict their own code — including LanceDB's own retry default, whose docstring says 10
while its signature says 7. Both probes above were run that way, and both times the signature carried
facts the prose did not: `optimize` returning `None`, and the two encoding parameters that nothing in
twenty tickets had named.

### 21.17 What the record does not settle

**One genuine contradiction: what "terminal for the run" means.** #30's retry table and #30's own prose
disagree. The table says a retryable class that exhausts its backoff ceiling is *"terminal for the run
— stop, as the OCR side does"*. Four sentences later the same resolution says *"Retries are counted per
request, not per Document, and a Document fails when a request carrying it exhausts them."* Those
cannot both be true of one event. #40 then split retries into three axes with separate budgets, and
said `embed` **raises** instead of `sys.exit(1)`, which changes the mechanism without saying which
disposition belongs to which axis.

This document adopts the reading that makes both sentences true of different axes, and says so instead
of picking a side in silence:

- **bad-response / timeout axis exhausted (`--max-request-retries 8`)** -> the request fails, splits,
  and the **Document** fails. The Invocation continues. This is what makes split-on-failure meaningful
  at all.
- **connection axis exhausted (6 attempts, about 4 minutes of bounded backoff)** -> the server is gone.
  **Terminal for the Invocation**, by an exception that leaves the worker, not by `sys.exit(1)` inside
  one. This is necessary: otherwise a dead server would burn through the corpus and mark every Document
  failed.

This matches `pplx.py`'s three-axis structure, which #40 chose as the precedent. **It is a reading, not
a recorded decision.** If it is wrong, it is wrong in a place with a visible symptom, so it is cheap to
correct.

**Four holes.**

1. **Where the enabled-Sink set lives for a LanceDB-only Invocation.** #35 says the manifest records
   which Sinks were enabled, so an Invocation that adds one can warn first. But the manifest is an
   `.npz` artifact. With `--no-npz --lancedb PATH` there is no `.npz` tree — although `<out>` still
   exists, because the failures file lives there. **The record does not say whether the manifest is
   written when `--no-npz` is set.** The smallest consistent reading is that it always is, because
   `<out>` always exists and the invariant comparison is useful either way. That reading is what this
   document assumes, and somebody should confirm it rather than inherit it.
2. **The `/tokenize` concurrency bound had no value — settled here** as `(concurrency * 3) // 2`, 96
   at the default, marked unmeasured on the same footing as every other constant. *This is a decision
   made in this document, not one recovered from the record.*
3. **The mechanism to make resumed skips visually distinct** was a requirement with no design —
   **settled here**: the bar counts only Documents that will actually be embedded, so a fully-skipped
   Invocation reads `0/0`. *Also a decision made in this document.*
4. **How `embed` reads Records is not specified.** `evaluation/runs.py`'s `load_run` flattens Records
   into `PageText` and **drops zero-length spans**, which `embed` must keep, because a zero-width page
   span matters to the packer. So this document specifies a small Record reader inside the embed
   package. It reuses `_iter_jsonl_paths`'s input-resolution semantics — a workspace directory
   (globbing `results/*.jsonl`), a bare directory of `*.jsonl`, or one `.jsonl` file — and yields whole
   Records. **This is a choice made here, not a recorded decision.**

**Stale text that still stands in the record.** A reader who goes back to the tickets will meet these.
A later amendment supersedes each one, and this document carries the amendment.

- **#25 still contains *"The document vector is a convenience; the chunk vectors are the output."*** It
  sits mid-resolution, inside the argument for giving every Document a vector, so a reader meets it
  while checking something unrelated. That ranking was retracted at the map level and on #31, but
  **#39's correction pass targeted #36 and never touched #25**. **It is superseded.**
- **#26's manifest example shows `"instruction": "passage: "` and `chunk_budget_tokens: 31900`.** Both
  are stale: the Instruction is two fields, and the budget is about 32,701 / 32,704 with
  `SAFETY_MARGIN = 64`.
- **#22's rule 3 (replace the extension) and rule 5 (the digest as tiebreak)** were both amended by #41.
- **#27's write path (`merge_insert` for everything)** was amended by #40.
- **#29's group listing names `empty` two times**; #30 moved it to `run` alone.
- **#30's batching rationale** is disproven by #36. The decision stands; the stated mechanism does not.
- **#33 is a two-engine client design for an engine now out of scope.** Build nothing from it.
- **#35's "must not be operator-settable"** was overturned by #37's amendment.
- **Every ticket that names the tokenizer route writes `POST /v1/tokenize`.** vLLM puts `tokenize`
  and `detokenize` at the top level. Only the OpenAI-compatible routes use `/v1`. The `/v1` form
  gives a 404 on every build. A 404 goes on the bad-response retry axis. So each Document used its
  full budget on an error that can never succeed. A live run found this. The tests could not: the
  fake transport answers each URL it gets, so the route test held the wrong text. **It is
  superseded.** The route is `/tokenize`.
- **`_parse_runs` is at `src/paperscale/cli.py:66`**, not `:65` as #27, #28 and #35 all cite.

### 21.18 The implementation plan

**Module layout.** `embed` is its own package, `src/paperscale/embed/`, and it holds the whole
subcommand, not only the Adapters. This mirrors `src/paperscale/evaluation/` exactly, including the
`_handle_evaluate` shape where every heavy import sits inside the handler function. That is what makes
one `embed` extra implementable at all.

| module | holds |
|---|---|
| `__init__.py` | public names only; **no heavy imports** |
| `adapters.py` | `EmbedModel` ABC, the five Adapters, `EMBED_MODEL_REGISTRY`, `build_embed_model` |
| `records.py` | Record reading and input resolution |
| `names.py` | Document-name derivation, the path digest, the startup collision check |
| `budget.py` | the `validated_context_length` rule, `--context-length`, `chunk_budget`, the request-budget floor |
| `chunking.py` | greedy page packing and the oversized-page path |
| `client.py` | `/v1/models`, `/tokenize`, `/v1/embeddings`, the explicit wire-parameter triple, the `"<f4"` decode, the three retry axes, the outstanding-request counter |
| `vectors.py` | MRL slice and re-normalize, token-weighted pooling, the single-Chunk short-circuit |
| `npz_sink.py` | manifest, sidecar, the eight arrays, write ordering, reserved names |
| `lance_sink.py` | the two tables, table metadata, `add()`/`merge_insert`, the scoped delete and its quoting |
| `resume.py` | derived state from both Sinks, the intersection, the layout guard, the Sink-set warning |
| `panel.py` | `push_embed_stats`, the fixed row set, the sustained-queue advisory |
| `run.py` | the orchestrator: startup order, producer/consumer stages, the single writer, the end-of-run report |

Wire the subcommand as `evaluate` is wired: an `embed` subparser in `cli.py`'s `build_parser()` with
`set_defaults(handler=_handle_embed)`; a `_handle_embed` whose every `paperscale.embed.*` import sits
inside the function body; and one more branch in `pipeline.cli_main`'s shim (`if sys.argv[1:2] ==
["embed"]`). Adapters live in `paperscale/embed/adapters.py` and **not** in `paperscale/models/`.

**Commit sequence: 17 commits, each with the test that proves it.** Commits 1–3 touch `tui.py` and are
the only ones that change existing behaviour. They can land first and independently. The full table is
§18.2.

**Both probes are done, and neither blocks a commit now.** LanceDB compaction is measured, so commit
13's docstring may claim the mitigation and name `Table.optimize()`. base64 is confirmed by source read
at 0.27.1 and not by a live call, so **one live check still belongs against whatever build is deployed,
before or with commit 8** — not to learn whether the fields are accepted (`extra="allow"` means an
unknown field is ignored, never rejected, so acceptance proves nothing) but to prove the round trip.

**Test obligations: 48 properties**, tagged **[explicit]** (a ticket names a test), **[measured]** (a
ticket records a measurement the code must preserve), or **[implied]** (stated as load-bearing, no test
named). They group as: pooling and dimensions (1–6), chunking (7–11), identity (12–17), `.npz` Sink
(18–22), LanceDB Sink (23–30), Resume (31–34), CLI and packaging (35–39), panel and reporting (40–46),
wire format (47–48). Items 47 and 48 were appended at the end on purpose, so no existing number shifts.
The full list is §18.3.

### 21.19 Documentation obligations

**The pinned-model rationale belongs in this document, not in user docs.** The split is between *why*
and *how*.

- **This document owns the rationale** — why these two families, why "long context" means 32K, why the
  two engines that instrument this workload better were ruled out, why the Adapter carries exactly
  three facts. Those are design arguments with downstream consequences. A user who reads the README to
  run a command does not need them, and to bury them there would mean the next person who questions a
  decision reads a usage guide instead of a design record.
- **The README owns the consequences.** Add a `## Embed` section, sibling to `## Evaluate`, with eight
  items: the supported models and how to serve them; the flag table with the unmeasured defaults
  labelled; the re-OCR warning verbatim; that `--no-resume` deletes nothing here; that `Source-File` is
  unnormalized upstream; that a second Sink re-embeds the corpus; what the above-card
  `--context-length` warning means; and the `embed` extra with its install line. §19 holds the full
  list.
- **`CONTEXT.md` was updated with this document, and it is now tracked.** Its **Sink** and **Resume**
  entries were scoped to the **Invocation** instead of the **Run**. **Chunk budget** and **Stored /
  Native dimension** were added as terms. **Overflow** gained its measured 6%. **Document name** gained
  the collision rule. Deliberately *not* added: `manifest`, `sidecar`, `stored_dim`. Those are one
  Sink's file roles and field names, not domain language, and a glossary that absorbs implementation
  nouns stops being one.
- **`evaluate`'s docs should recommend** the `[A-Za-z0-9._-]` label charset, so labels stay portable,
  without `evaluate` enforcing it.

### 21.20 The decision index

22 tickets, each with its status, are in §20. #21 is the map and is open only because this document
closes it. #31 is this document. #32 (markdown export overwrite) is open and out of scope; it gains a
guard and keeps its names. #42 (unify output-path derivation between `embed` and the markdown export)
is open and was filed by #41. Everything else from #22 to #41 is closed, with the amendments named.

Four research and prototype branches exist, all local only, because the repository's push guardrail
blocked publishing. **Branch `feat/embed` holds an abandoned 4,869-line implementation of an earlier,
rejected design.** It is reference only, and nobody should read it. Nothing in this document comes from
it, and to read it reintroduces the assumptions this map spent twenty tickets discarding.

### 21.21 Where this stands

- The design is complete and locked. **No code exists.** Commit 1 of 17 has not started.
- **Ticket #31 is open.** To close it closes the map.
- **One prerequisite is open**: the event-pane padding (cap growth, or let stats absorb the surplus).
  It is cosmetic and settleable in commit 2's review.
- **One live check is owed** before or with commit 8: the base64 round trip against a real
  pooling-task server.
