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
  reproduces the original YAML-front-matter + rotation behavior).
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
- `--ocr-model {markdown,olmocr}` — which OCR adapter drives prompting/parsing.
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
