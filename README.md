# paperscale

OCR documents to Markdown at scale. paperscale is a local, model-agnostic
reimplementation of [olmOCR](https://github.com/allenai/olmocr)'s batch
pipeline: a work queue of PDF/image groups is drained by a pool of async workers
that render each page, send it to an OpenAI-compatible OCR model, and assemble
per-document [Dolma](https://github.com/allenai/dolma) JSONL (plus optional
Markdown) into a workspace directory.

It keeps olmOCR's document management, queueing, and CLI 1:1, with two additions:

- **Decoupled models** — the prompt and response parsing live in a small
  `OCRModel` adapter, so **any** model that emits Markdown works, not just
  olmOCR. Pick one with `--ocr-model` (`markdown` is the default; `olmocr`
  reproduces the original YAML-front-matter + rotation behavior; `lightonocr2`
  drives [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B), a
  SOTA 1B Markdown-OCR model — image-only prompt, rendered at 1540px by default;
  `lightonocr2-soup` is the same adapter on the more-robust
  [`-ocr-soup`](https://huggingface.co/lightonai/LightOnOCR-2-1B-ocr-soup)
  merged checkpoint; `glm-ocr`, `qianfan-ocr`, `infinity-parser2-flash`, and
  `surya2` add four more document-OCR VLMs, documented below).
- **Opt-out resume** — completed work items are skipped on restart by default
  (olmOCR's done-flag behavior). `--no-resume` wipes prior progress and
  reprocesses the workspace from scratch.

A separate `paperscale evaluate` subcommand ranks several models against each
other from their run outputs, without ground truth — see [Evaluate](#evaluate).

A separate `paperscale embed` subcommand turns those same run outputs into
vectors — chunked, pooled per document, and written to `.npz` files, a LanceDB
database, or both — see [Embed](#embed).

S3 and beaker support from upstream olmOCR are intentionally omitted; workspaces
and inputs are local.

## Install

paperscale renders and reads PDFs with **poppler**, so `pdftoppm`, `pdfinfo`,
and `pdftotext` must be on your PATH:

```bash
sudo pacman -S poppler          # Arch
sudo apt install poppler-utils  # Debian/Ubuntu
brew install poppler            # macOS
```

Then install the package (Poetry project):

```bash
poetry install
```

This installs the `paperscale` console script. Run it with `poetry run paperscale ...`.

Spawning the **internal** vLLM server (i.e. omitting `--server`) additionally
needs `torch`/`transformers`/`vllm`/`huggingface_hub` installed separately —
follow vLLM's GPU install guide. Pointing at an external server needs none of
that.

## Usage

Point `--server` at any OpenAI-compatible endpoint and feed it documents:

```bash
poetry run paperscale ./workspace \
  --pdfs './docs/*.pdf' \
  --server http://127.0.0.1:8000/v1 \
  --model your-served-model-id \
  --markdown
```

- `workspace` (positional) — local directory where the queue, locks, done
  flags, `results/*.jsonl`, and (with `--markdown`) `markdown/` live.
- `--pdfs` — local PDF/image paths, a glob (`'docs/*.pdf'`), `.tar.gz` tarballs,
  or a `.txt` file listing one path per line.
- `--ocr-model {glm-ocr,infinity-parser2-flash,lightonocr2,lightonocr2-soup,markdown,olmocr,qianfan-ocr,surya2}`
  — which OCR adapter drives prompting/parsing.
- `--model` — the served model id sent in each request (or a Hugging Face path
  for the internal server).

Inputs are added to the workspace queue, grouped into work items, and processed
concurrently by `--workers`. Re-running the same command **resumes**: finished
work items are skipped. Use `--no-resume` to start clean.

Report progress without running any job:

```bash
poetry run paperscale ./workspace --stats
```

See `poetry run paperscale --help` for every flag.

### Live dashboard

`--tui` renders a fixed-height dashboard (needs `poetry install --extras tui`).
It shows progress, live vLLM throughput and prefix-cache hit rate scraped from
the server's `/metrics` endpoint, and document outcome counters. All three
commands take it — the OCR pipeline, `paperscale evaluate` and
`paperscale embed` — and it is off by default: without `--tui` the output is
exactly what it always was.

```bash
poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --server http://127.0.0.1:8000/v1 --tui
```

The dashboard is designed for tmux: the layout is recomputed on every frame, so
splitting, zooming, and detach/reattach all resize cleanly. While it is on,
nothing may write to stderr underneath the frame, so the full log is routed to a
file and only warnings appear in the event pane:

- pipeline — `<workspace>/logs/run-<pid>.log`, or the path given to
  `--disk_logging`.
- `evaluate` — `<--db directory>/logs/evaluate-<pid>.log`, or the path given to
  `--disk-logging`. (evaluate has no workspace, so the database's directory is
  its home.)
- `embed` — beneath the `--out` directory, or the path given to
  `--disk-logging`.

The path is printed once the frame is gone, since the alternate screen keeps no
scrollback.

- `PAPERSCALE_TUI_ASCII=1` — force ASCII panel borders, spinner, and truncation
  marker. Use this if your terminal font renders box-drawing or braille
  characters as blanks or boxes. Font coverage cannot be detected: a UTF-8 locale
  says nothing about whether the rendering font has a braille block, and under
  tmux `TERM` describes tmux rather than the outer terminal, so the font actually
  drawing the glyphs is invisible from inside. The progress bar is the one thing
  this does not reach — `rich` picks its bar glyphs from the stream's own
  encoding, so the bar goes ASCII only when stderr is not a UTF-8 stream.
- `PAPERSCALE_TUI_ASCII=0` — force rich glyphs when detection is over-cautious
  (an ASCII-declared stream on a terminal that draws them fine).
- `--tui-poll-interval` — seconds between `/metrics` scrapes (default 5).

If the server exposes no `/metrics` — anything that is not vLLM, such as a
hosted OpenAI-compatible endpoint — the vllm panel reads `unavailable` and the
run continues. A statistics panel never ends a twelve-hour job.

### LightOnOCR-2

`--ocr-model lightonocr2` selects
[LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B). It sends the
page image with no text prompt and renders at 1540px by default (override with
`--target_longest_image_dim`).

`--ocr-model lightonocr2-soup` is the same adapter pointed at the
[`-ocr-soup`](https://huggingface.co/lightonai/LightOnOCR-2-1B-ocr-soup)
checkpoint — a task-arithmetic merge of the base and RLVR-trained weights for
extra robustness on hard pages, with identical prompting and output. (The
`bbox` / `bbox-soup` variants emit bounding boxes and are not wired up.) Serve it
the same way, swapping the model id:

```bash
vllm serve lightonai/LightOnOCR-2-1B-ocr-soup --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model lightonocr2-soup --server http://127.0.0.1:8000/v1 --markdown
```

Against an external vLLM server (recommended):

```bash
vllm serve lightonai/LightOnOCR-2-1B --port 8000 \
  --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 --no-enable-prefix-caching

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model lightonocr2 --server http://127.0.0.1:8000/v1 --markdown
```

With the internal server (omit `--server`), the model id is resolved
automatically; forward vLLM's recommended flags as trailing args:

```bash
poetry run paperscale ./workspace --pdfs './docs/*.pdf' --ocr-model lightonocr2 --markdown \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching --limit-mm-per-prompt '{"image": 1}'
```

### GLM-OCR

`--ocr-model glm-ocr` drives [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR)
(`zai-org/GLM-OCR`, ~0.9B, Apache-2.0), a GLM-4.1V-derived VLM fine-tuned for
document transcription. The page image is sent with a full-page instruction and
the model returns the whole page as Markdown (tables as Markdown, formulas as
LaTeX) in a single call. vLLM serves it natively via the `glm_ocr` architecture
(vLLM PR #33005); use a recent vLLM build and upgrade `transformers` alongside it.

```bash
vllm serve zai-org/GLM-OCR --port 8000 --limit-mm-per-prompt '{"image": 1}'

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model glm-ocr --server http://127.0.0.1:8000/v1 --markdown
```

### Qianfan-OCR

`--ocr-model qianfan-ocr` drives
[Qianfan-OCR](https://huggingface.co/baidu/Qianfan-OCR) (`baidu/Qianfan-OCR`,
~4B), a Qianfan-ViT + Qwen3-4B document model. One image + prompt call returns
clean reading-ordered Markdown (HTML tables, `$$…$$` LaTeX). vLLM runs it as an
`InternVLChatModel` via `--hf-overrides`:

```bash
vllm serve baidu/Qianfan-OCR --port 8000 --trust-remote-code \
  --hf-overrides '{"architectures": ["InternVLChatModel"]}' --limit-mm-per-prompt '{"image": 1}'

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model qianfan-ocr --server http://127.0.0.1:8000/v1 --markdown
```

The adapter raises `max_tokens` to 12000 for long pages and defensively strips a
leading `<think>…</think>` layout block (only emitted when `enable_thinking=True`,
which paperscale never sets).

### Infinity-Parser2 Flash

`--ocr-model infinity-parser2-flash` drives
[`infly/Infinity-Parser2-Flash`](https://huggingface.co/infly/Infinity-Parser2-Flash),
a ~2B Qwen3.5-VL document parser, in **doc2md** mode (Markdown with HTML tables
and LaTeX) rather than its native bbox-JSON mode. Qwen3 reasoning is disabled via
the top-level `chat_template_kwargs={"enable_thinking": false}` request key. Serve
it on vLLM ≥ 0.20 (Python 3.13):

```bash
vllm serve infly/Infinity-Parser2-Flash --port 8000 --trust-remote-code \
  --reasoning-parser qwen3 --mm-encoder-tp-mode data --mm-processor-cache-type shm \
  --enable-prefix-caching --limit-mm-per-prompt '{"image": 1}'

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model infinity-parser2-flash --server http://127.0.0.1:8000/v1 --markdown
```

### Surya OCR 2

`--ocr-model surya2` drives
[Surya OCR 2](https://huggingface.co/datalab-to/surya-ocr-2)
(`datalab-to/surya-ocr-2`, ~0.65B, `Qwen3_5ForConditionalGeneration`). Its
full-page-OCR mode emits reading-ordered **layout-HTML** (`<div data-label=…>`
blocks, `<math>` KaTeX, HTML `<table>`); paperscale converts that to Markdown
(headings, lists, Markdown tables, and `\( … \)` / `\[ … \]` LaTeX). Serve it on a
recent vLLM with native `qwen3_5` support (v0.20.0+; avoid v0.18.0):

```bash
vllm serve datalab-to/surya-ocr-2 --port 8000 --limit-mm-per-prompt '{"image": 1}'

poetry run paperscale ./workspace --pdfs './docs/*.pdf' \
  --ocr-model surya2 --server http://127.0.0.1:8000/v1 --markdown
```

## Outputs

For each work item, paperscale writes `workspace/results/output_<hash>.jsonl` —
one Dolma document per source file, with the extracted `text` and per-page
metadata/attributes. With `--markdown`, it also writes
`workspace/markdown/<input structure>/<doc>.md` containing just the page text.

## Evaluate

`paperscale evaluate` ranks OCR models against each other **without ground
truth**. It reads the `results/*.jsonl` that ordinary runs already produce,
scores every page on a handful of reference-free signals, writes them to SQLite,
and prints a leaderboard:

```bash
poetry run paperscale evaluate \
  --run lightonocr2=./ws-lighton \
  --run surya2=./ws-surya \
  --db ./evaluation.sqlite
```

```
model        corr_rate  corr_rate_win  uncorr_rate  garbage  garbage_win  peer_f1  peer_f1_win  peer_ned  tl_f1  tl_f1_win  tl_ned  reject_rate  reject_rate_win
-----------  ---------  -------------  -----------  -------  -----------  -------  -----------  --------  -----  ---------  ------  -----------  ---------------
lightonocr2  0.000      1.00           0.000        0.000    1.00         0.557    1.00         0.307     0.912  1.00       0.804   0.000        1.00
surya2       0.152      0.00           0.000        0.100    0.00         0.557    1.00         0.307     0.881  0.00       0.771   0.000        1.00
```

Each `--run` is `LABEL=PATH`, where `PATH` is a workspace dir, a bare dir of
`.jsonl`, or a single `.jsonl` file. The label is yours to choose — the model
name is not recorded in the JSONL. Documents join across runs by
`metadata["Source-File"]`, so the same PDFs must be fed to every model you
compare.

`evaluate` accepts any non-empty, unique label, but sticking to
`[A-Za-z0-9._-]` keeps them portable: `paperscale embed` takes the same
`--run LABEL=PATH` and puts the label in a filesystem path, so it *enforces*
that charset. A label like `legal/2024` works here and is rejected there.

### Metrics

A single run scores every column except `peer_*`, which compares models against
each other and is skipped below two runs.

| Column | Meaning | Better |
|---|---|---|
| `corr_rate` | Fraction of tokens a spell checker had to change | lower |
| `uncorr_rate` | Fraction it could **not** fix — garbage beyond repair | lower |
| `garbage` | Fraction of tokens that look like OCR noise: repeated characters, alpha-digit soup, vowel-less runs, mid-word case flips | lower |
| `peer_f1` / `peer_ned` | Agreement with the *other* models on the same page (bag-of-words F1, and 1 − normalized edit distance) | higher |
| `tl_f1` / `tl_ned` | Agreement with the PDF's own embedded text layer (`pdftotext`) | higher |
| `reject_rate` | Fallback pages ÷ total pages — how often the quality gate rejected the model's output | lower |
| `ppl_raw` / `ppl_corr` | Perplexity under an external LM, before and after spell correction (`--pplx` only) | lower |

A `*_win` column is the fraction of documents where that model scored best,
restricted to documents **all** models produced — so an extra PDF in one run
cannot inflate its win rate. It reads `n/a` with fewer than two runs.

With exactly **two** runs the `peer_*` columns are necessarily identical for both
models — agreement is symmetric, so `f1(a, b)` is `f1(b, a)` — and both win every
document. The column only starts discriminating at three or more runs, where a
model can agree with the consensus more than its rivals do. It is a
majority-vote signal, not a correctness one: three models sharing a systematic
error will all score well on it.

Means are **doc-weighted, not page-weighted**: each document contributes its own
mean, so a 400-page report does not drown out a 1-page invoice.

`tl_*` is a calibration signal on a subset, not full coverage. A document is
skipped entirely if any of its pages fell back (those pages are *filled* with
`pdftotext` output, so comparing them to the text layer would be circular), if
its PDF is no longer on disk, or — per page — if the text layer is effectively
blank, which is what a scanned image looks like. The run logs how many of each
it skipped. Expect `-` in these columns when evaluating scanned corpora.

`--dictionary words.txt` adds domain vocabulary (one word per line, repeatable)
to the spell checker, which matters on legal, medical, or technical corpora
where real jargon would otherwise be counted as a correction.

### Resume

Re-running the same command **resumes**: any document whose text is unchanged
keeps its cached scores, and only new or modified work is computed. This is
tracked per document via a checksum of its page text, so:

- Interrupting a long evaluation loses at most the documents in flight.
- Re-running OCR under the same label rescores exactly the documents whose
  output actually changed.
- Adding a third `--run` scores the new model and fills in only the peer pairs
  that involve it — the existing models are not re-scored against each other.
- Deleting a PDF from a run drops its rows, so it stops affecting the means.

`--no-resume` discards every cached score for the listed runs and starts clean.

### Perplexity (optional)

`--pplx` adds a language-model perplexity column, scored against a separate
OpenAI-compatible server. Each page is sent to `/v1/completions` with
`prompt_logprobs: 0` and `max_tokens: 1` — prefill only, no generation — and
scored twice: as-is, and after spell correction. The gap between `ppl_raw` and
`ppl_corr` separates "this model produced unusual text" from "this model
produced misspelled text".

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --no-enable-prefix-caching

poetry run paperscale evaluate \
  --run lightonocr2=./ws-lighton --run surya2=./ws-surya \
  --pplx --pplx-url http://localhost:8000 --pplx-model Qwen/Qwen3-8B
```

Serve the scorer with prefix caching **off** — a cached prefill can skip the
`prompt_logprobs` this depends on.
Scoring is GPU-bound, so raise `--pplx-concurrency` only until the server's
reported prefill throughput stops climbing. `--pplx-chunk-tokens` caps the
tokens per request (default 32000); smaller chunks let vLLM batch far more of
them, at the cost of cross-page conditioning at each boundary. Changing
`--pplx-model` invalidates stored perplexity scores automatically — they are not
comparable across scorers.

Every metric also lands in the SQLite file (one table per metric, keyed by
model/doc/page), so you can dig past the leaderboard:

```bash
sqlite3 evaluation.sqlite \
  "SELECT doc, page, score FROM garbage_fraction WHERE model='surya2' ORDER BY score DESC LIMIT 10;"
```

See `poetry run paperscale evaluate --help` for every flag.

## Embed

`paperscale embed` turns an OCR run's text into vectors. It reads the same
`results/*.jsonl` that `evaluate` does, chunks each document to fit the embedding
model's context, sends the chunks to an external vLLM server, and writes **chunk
vectors plus one pooled document vector** to a `.npz` tree, a LanceDB database,
or both. It needs the `embed` extra (numpy, lancedb, pyarrow):

```bash
poetry install --extras embed

vllm serve Qwen/Qwen3-Embedding-8B --port 8000 --runner pooling

poetry run paperscale embed \
  --run lightonocr2=./ws-lighton \
  --embed-model qwen3-embedding-8b \
  --embed-url http://127.0.0.1:8000 \
  --out ./vectors
```

Like the extras elsewhere, it is only paid for when used: every numpy, lancedb
and pyarrow import sits inside the subcommand, so an OCR-only install still
reaches `paperscale embed --help`.

`--run LABEL=PATH` is repeatable and takes the same paths `evaluate` does — a
workspace dir, a bare dir of `.jsonl`, or a single `.jsonl` file. The label is
recorded with every vector and, when one invocation embeds more than one run,
becomes a directory in the output tree, so `embed` requires it to match
`[A-Za-z0-9._-]+`.

`--embed-model` is **required**, which is where it departs from `--ocr-model`.
Two OCR models produce text that is roughly comparable; two embedding models
produce vectors that are meaningless against each other, and both sinks bake the
model into the output's identity — so a default would silently pick the semantics
of a whole corpus.

`--embed-url` is a base URL **without** `/v1` (mirroring `--pplx-url`).
paperscale asks it for the served model id and `max_model_len` on `/v1/models`,
counts tokens on `/v1/tokenize`, and embeds on `/v1/embeddings`.

Both sinks carry both artifacts, because there are two consumers and neither is
primary. Retrieval reads the **chunk vectors** — ranking is max-over-chunks,
which is query-dependent, so a stored document vector cannot do that job.
Classification reads the **document vectors**, which are exactly the `(n_docs,
dim)` feature matrix `sklearn`/`xgboost`/torch want. LanceDB is the working store
for both; the `.npz` tree is for portability and archive — no database, plain
files, numpy the only thing needed to read them.

### Models

| `--embed-model` | Served model | Native width | Document instruction | Query instruction |
|---|---|---|---|---|
| `qwen3-embedding-0.6b` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | *(none)* | `Instruct: {task_description}\nQuery:{query}` |
| `qwen3-embedding-4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | *(none)* | *(as above)* |
| `qwen3-embedding-8b` | `Qwen/Qwen3-Embedding-8B` | 4096 | *(none)* | *(as above)* |
| `nemotron-3-embed-1b` | `nvidia/Nemotron-3-Embed-1B-BF16` | 2048 | `passage: ` | `query: ` |
| `nemotron-3-embed-8b` | `nvidia/Nemotron-3-Embed-8B-BF16` | 4096 | `passage: ` | `query: ` |

The adapter carries the handful of facts no serving engine reports: the card's
context length (32768 for all five), the native width, the usable Matryoshka
floor (32), and the two instruction strings. Everything else is asked of the
server — the served model id, exact token counts, and the output width.

**paperscale applies the document instruction and never applies the query one.**
Nemotron requires `passage: ` on documents and `query: ` on queries; Qwen3 says
documents need no instruction, so its document side is the empty string — a
recorded decision, not a missing value. Both strings are written into every sink
so the consumer can build queries that match. Omitting Qwen3's query-side
instruction costs roughly 1–5% of retrieval performance by Qwen's own
measurement; paperscale never makes that choice for you, it just hands you the
string.

**Serving.** vLLM must load the model as a pooling model — recent builds spell
that `--runner pooling`, older ones `--task embed`; check `vllm serve --help` for
yours. Nothing else is required. In particular no `--hf-overrides
'{"is_matryoshka": true}'`: paperscale requests native-width vectors and slices
them itself, so `dimensions` never appears in a request.

vLLM is the only supported server, and the narrowing is a correctness one rather
than a preference: `embed` relies on the server **erroring** on oversized input,
because a server that truncates silently would leave stored character offsets
describing text that was never embedded, and nothing downstream could detect it.
The reasoning for each engine that was ruled out is in `docs/design/embed.md`.

The native width doubles as the wrong-model check: every response arrives at
native width (the slice is client-side), so a server holding a different size of
the same family is caught on the first response and the invocation stops, naming
both numbers. It cannot catch two models of *equal* width — Qwen3-Embedding-8B
and Nemotron-3-Embed-8B are both 4096 — which is why the served model id from
`/v1/models` is what the dashboard header shows, not the string you typed.

### Flags

| Flag | Default | Notes |
|---|---|---|
| `--run LABEL=PATH` | *required* | repeatable; label must match `[A-Za-z0-9._-]+` |
| `--out PATH` | `./vectors` | the `.npz` tree, the manifest, and the failures file |
| `--embed-model NAME` | *required* | selects the adapter, from the table above |
| `--embed-url URL` | `http://localhost:8000` | base URL of the vLLM server, no `/v1` |
| `--embed-dim N` | `768` | stored width, sliced client-side; rejected outside `[32, native]` |
| `--context-length N` | `min(card, server)` | rejected above the server's `max_model_len`; warns above the card's |
| `--api-key KEY` | unset | bearer token for the embedding server |
| `--lancedb PATH` | unset | the path is the opt-in: presence enables the LanceDB sink |
| `--no-npz` | off | disables the `.npz` sink; rejected without `--lancedb` |
| `--concurrency N` | `64` | embedding requests in flight — a starting point, see below |
| `--request-tokens N` | `32000` | token budget per request, raised to the chunk budget when below it — a starting point |
| `--max-request-retries N` | `8` | bounds the bad-response/timeout axis only — a starting point |
| `--no-resume` | off | re-embed and overwrite; **deletes nothing** |
| `--tui` | off | live dashboard (needs the `tui` extra) |
| `--tui-poll-interval S` | `5.0` | seconds between `/metrics` scrapes |
| `--disk-logging PATH` | unset | where the full log goes while `--tui` is on |

Three of those numbers — `--concurrency 64`, `--request-tokens 32000` and
`--max-request-retries 8` — are **calibrated starting points, not measurements of
this workload**. They are borrowed from `evaluate`'s perplexity scorer
(`--pplx-concurrency 64`, `--pplx-chunk-tokens 32000`) and the OCR side's
`--max_page_retries 8`, which drive the same hardware in the same prefill-only
way. Perplexity scoring is not embedding, so treat them as a sane place to start
and measure from there. The ~60 s window on the queue advisory below sits on the
same footing.

**The instrument for tuning concurrency is `vllm:num_requests_waiting`**, from
the server's own `/metrics`. If it sits at zero the server is idle and
`--concurrency` can go up; if it stays above zero the queue is backing up and the
client is only adding latency. With `--tui` it is the right-hand half of the
`in-flight` row (`<in flight from here>/<waiting on the server>`), and when the
queue stays non-empty for about a minute the event pane says so:

```
queue depth sustained; --concurrency 64 may be too high for this server
```

That is advisory: it names the flag and changes nothing, so throughput never
moves for reasons you cannot see.

Requests are bounded by a token budget rather than a count of chunks, because
greedy packing means one chunk may be a short page and the next forty-five dense
ones — "16 chunks per request" could be anywhere from a few hundred to half a
million prefill tokens. `--request-tokens` is **raised** to the chunk budget when
it sits below it (the 32000 default is ~704 tokens under the floor for both
pinned families), with a log line, so a full-size chunk is always sendable.
Requests mix documents, and a request that fails is re-issued one document at a
time before any document is recorded as failed.

### Dimensions and context length

`--embed-dim` (default 768) is a **client-side Matryoshka slice**: paperscale
asks for native-width vectors, cuts each chunk vector to the stored width,
re-normalizes it, and only then pools. That order is what lets a consumer holding
one sink recompute the document vector from the stored chunk vectors and get the
same answer. A width outside `[32, native]` is **rejected, not clamped** — asking
a 2048-wide model for 4096 means the mental model is wrong (usually: you meant
the 8B sibling), and quietly serving 2048 would hide it.

`--context-length` defaults to `min(model card, server max_model_len)`. The card
is authoritative about the model and the server is authoritative about the
deployment, and neither is authoritative about both: a default `vllm serve`
advertises 262144 for Nemotron and 40960 for Qwen3-8B against a documented 32768,
while an operator who launched with `--max-model-len 8192` would otherwise have
every long document hard-fail.

- **Above the server's `max_model_len`** the value is rejected at startup. The
  request could not succeed, so failing once beats failing per document.
- **Above the model card** it is permitted, and warns naming both numbers. What
  the warning means is that above the card, quality is **unmeasured by the
  vendor** — the card's number is where the vendor stopped measuring, not a
  measured cliff. The judgement is yours; paperscale records the numbers rather
  than enforcing a quality opinion under cover of a safety check.

The chunk budget follows from that: `validated context length - tokens(document
instruction) - 64`, where the 64 is margin against `/v1/tokenize` and
`/v1/embeddings` disagreeing by a token on special tokens. For a default serve
that is 32704 (Qwen3) or about 32701 (Nemotron), which is roughly 45 dense pages
— so chunking is the minority path, not the usual one. There is deliberately no
`--chunk-tokens`: the chunk budget is derived rather than chosen, it is recorded
in provenance, and letting it move between invocations would silently change what
a chunk is.

### What it writes

```
vectors/paperscale-embed.json                      # the invocation manifest
vectors/paperscale-embed-failures.txt              # written when documents failed
vectors/home/cc/corpus/law/case.pdf.npz            # eight arrays, nothing else
vectors/home/cc/corpus/law/case.pdf.json           # the per-document sidecar
```

The tree mirrors the source paths, exactly as the markdown export does. With two
or more `--run`s the label is inserted first (`vectors/<label>/home/cc/...`); with
one it is not. That choice is recorded in the manifest as the `layout`, and an
invocation that would change it over an existing output stops and says so —
otherwise every document would be re-embedded into a parallel subtree and the old
one orphaned.

Each `.npz` holds eight arrays and no metadata (an `.npz` cannot carry a header
without `allow_pickle`, which means asking a reader to execute the file):
`chunk_vectors` `(n_chunks, dim)` and `document_vector` `(dim,)` as `float32`,
plus `start_char`, `end_char`, `first_page`, `last_page`, `token_count` as
`int32` and `is_partial_page` as `bool`, each `(n_chunks,)`.

```python
import numpy as np

z = np.load("vectors/home/cc/corpus/law/case.pdf.npz")
z["document_vector"]        # (768,) float32, unit length -- (0,) if the document had no text
z["chunk_vectors"]          # (n_chunks, 768) float32, unit rows
z["start_char"][0], z["end_char"][0]   # slice chunk 0 back out of the record's own text
```

Chunk text is not stored: the record already holds it and the offsets are exact,
so `record["text"][start:end]` reconstructs a chunk. `chunk_index` and `n_chunks`
are not stored either — they are `np.arange(len(...))` and `len(...)`, and storing
a derived value invites the two copies to disagree. The sidecar next to each
`.npz` carries the four per-document facts that vary: `source_file`,
`source_digest`, `run_label`, `created`. Everything invariant — model id, stored
and native widths, both instructions, the pooling and chunker names, and the
chunk budget — lives once in `paperscale-embed.json`, and a later invocation that
disagrees with any of it stops before writing a single document.

`--lancedb PATH` writes two tables instead, `documents` (one row per document)
and `chunks` (one row per chunk), both naming the vector column `vector` so
`.search()` needs no column name. `document_name` joins them and `run_label`
separates runs sharing one database. The same invariant facts are the tables'
Arrow schema metadata, written once and compared on reopen. No vector index is
built: brute-force search is exact and needs none, while the IVF_PQ index
`create_index` would build is a lossy trade that belongs to whoever queries the
store.

A document whose OCR text is empty is a **recorded outcome, not a failure**: the
`.npz` gets every array at length zero (test it with
`z["document_vector"].size == 0`), and LanceDB gets one `documents` row with
`n_chunks = 0` and a NULL vector, which vector search skips. Recording it is what
stops it being retried on every future invocation.

An invocation that ends with any failed document **exits non-zero** and lists the
names in `<out>/paperscale-embed-failures.txt`, one per line. That file is a
convenience, not state — a failed document simply has no output and is retried
next time. The end-of-run report counts documents by outcome (embedded, skipped,
empty, failed, oversize); a non-zero `oversize` means a chunker or context-length
bug rather than a corpus problem.

### Document names

The name of every output is derived from the record's `metadata["Source-File"]`:
the path normalized and made relative, **with its extension kept rather than
replaced**, so `case.pdf` becomes `case.pdf.npz` beside `case.pdf.json` and
`case.tiff` cannot overwrite it. A tarball source
(`archive.tar.gz::inner/doc.pdf`) becomes `archive/inner/doc.pdf`, matching the
markdown export. When no usable path survives — `Source-File` missing, sanitizing
to nothing, or a component past the filesystem's 255-byte limit — the name is a
16-character digest of the raw string instead. Two documents in one run whose
names would collide are fatal at startup, listing every colliding group with both
raw `Source-File` values, because that name is also the LanceDB key and the
resume key.

**`Source-File` is whatever the OCR run recorded, and it was never normalized.**
`_expand_pdf_inputs` calls neither `abspath` nor `realpath`, so a corpus OCR'd as
`docs/a.pdf` from one working directory and as `/home/cc/docs/a.pdf` from another
yields two different document names for the same PDF — and nothing records the
OCR-time working directory, so no post-hoc processing can equate them. In
practice resume re-reads the same JSONL, so names are stable unless the corpus is
re-OCR'd from somewhere else. Mirroring the markdown export's layout was chosen
over a `--source-root` flag deliberately: having the two trees line up —
`vectors/home/cc/corpus/x.pdf.npz` beside `markdown/home/cc/corpus/x.md` — is
worth more than a flag on every invocation papering over an upstream
inconsistency. The raw `source_file` string is recorded in both sinks, so
anything reconciling by hand still has it.

### Resume

Re-running the same command **resumes**. A document is done when **every enabled
sink holds it**, so the resume set is the intersection of what the sinks know,
read once at startup — there is no per-document lookup during the run, and the
progress bar counts only what is left, so a fully resumed invocation reads `0/0`.

- **Enabling a second sink re-embeds the corpus.** Run once with `.npz`, then add
  `--lancedb`, and the intersection is empty: LanceDB holds nothing, so every
  document is embedded again. That is correct and it is expensive, which is why
  the manifest records the enabled sinks and the run says so before it starts
  rather than after a day of GPU time.
- **A crash between the two sinks heals itself.** The intersection says "not
  done", the next invocation re-embeds the document and writes it to both, where
  one write is a no-op and the other completes.
- **`--no-resume` deletes nothing here**, unlike the OCR pipeline's, which
  `rmtree`s the workspace's `results`, `done_flags` and `worker_locks`. A
  workspace is scratch; an embed output is the deliverable, quite possibly
  already being read by something, and one pair of LanceDB tables can hold
  several runs. So `--no-resume` re-embeds every document and overwrites it in
  place — the end state matches a wipe except that outputs whose documents have
  left the input are left alone.

What resume does **not** do is the one thing worth reading twice:

**Re-OCR-ing a corpus leaves stale vectors, and `embed` will not notice.**

Resume asks one question about each Document: *have I seen this name before?* It does not look at
the text. If you re-run OCR over the same PDFs -- with a different model, different settings, or a
newer version -- the Documents keep their names, and `embed` will skip every one of them. The
vectors in your output will continue to describe the *old* text, and nothing in the tool will tell
you.

`embed` does check two things before it starts, and stops the run if either disagrees: that the
embedding model and its settings match what built the output, and that the output layout has not
changed. Neither of those notices changed *text*.

After a re-OCR, either embed into a fresh output directory, or pass `--no-resume` to re-embed
everything in place.

See `poetry run paperscale embed --help` for every flag.

## Development

```bash
poetry run pytest -q                       # tests (stdlib unittest, run via pytest)
poetry run ruff check src/ tests/          # lint
poetry run pyrefly check src/paperscale    # type-check
```

The end-to-end tests render a real PDF page and drive a worker against a fake
in-process server, so poppler must be installed for them to run (they skip
otherwise).
