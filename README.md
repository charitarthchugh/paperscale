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

## Development

```bash
poetry run pytest -q                       # tests (stdlib unittest, run via pytest)
poetry run ruff check src/ tests/          # lint
poetry run pyrefly check src/paperscale    # type-check
```

The end-to-end tests render a real PDF page and drive a worker against a fake
in-process server, so poppler must be installed for them to run (they skip
otherwise).
