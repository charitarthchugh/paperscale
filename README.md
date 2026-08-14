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
the server's `/metrics` endpoint, and document outcome counters. Both commands
take it — the OCR pipeline and `paperscale evaluate` — and it is off by default:
without `--tui` the output is exactly what it always was.

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

## Development

```bash
poetry run pytest -q                       # tests (stdlib unittest, run via pytest)
poetry run ruff check src/ tests/          # lint
poetry run pyrefly check src/paperscale    # type-check
```

The end-to-end tests render a real PDF page and drive a worker against a fake
in-process server, so poppler must be installed for them to run (they skip
otherwise).
