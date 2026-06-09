# paperscale

OCR documents to Markdown at scale using vision language models (VLMs) served
over an OpenAI-compatible API. paperscale renders each PDF page to an image,
sends it to the model, validates the Markdown, and assembles the pages into one
document. Every page attempt is journaled to a local ledger, so a crashed or
interrupted run resumes exactly where it stopped without re-OCRing finished
pages.

## Install

paperscale renders PDF pages with **poppler**, so the `pdftoppm` and `pdfinfo`
binaries must be on your PATH:

```bash
sudo pacman -S poppler          # Arch
sudo apt install poppler-utils  # Debian/Ubuntu
brew install poppler            # macOS
```

paperscale is a Poetry project. Install the package and its console scripts:

```bash
poetry install
```

This installs two commands into the environment: `paperscale` (the OCR runner)
and `paperscale-mock-api` (a deterministic mock inference server for local
development and tests). Run them with `poetry run paperscale ...` or activate the
environment first.

For the mock server you also need the optional extra:

```bash
poetry install -E mock-api      # or: pip install 'paperscale[mock-api]'
```

## Quickstart (against the mock server)

Start the deterministic mock inference API in one terminal:

```bash
poetry run paperscale-mock-api serve --port 8009 --model mock-vlm
```

OCR a PDF in another terminal:

```bash
poetry run paperscale run \
  --input document.pdf \
  --output document.md \
  --base-url http://127.0.0.1:8009 \
  --model mock-vlm \
  --job-id mydoc
```

`run` prints a one-line summary and exits `0` when every page succeeds. If the
run stops early — a page fails, the provider trips the circuit breaker, or the
process is killed — resume it:

```bash
poetry run paperscale resume mydoc          # reuses the original job's settings
poetry run paperscale status mydoc          # succeeded/pending/failed/ambiguous counts
```

## Running against a real vLLM server

The runner speaks the OpenAI chat-completions protocol, so any OpenAI-compatible
server works. Point `--base-url` at the server (paperscale appends `/v1` when you
omit it) and pass the served model id. Validate reachability and model
availability before starting a run:

```bash
poetry run paperscale doctor provider \
  --base-url http://127.0.0.1:8000 \
  --model your-served-model \
  --capacity local-vllm-small
```

`doctor` exits `0` when the server is reachable and serves the requested model,
`1` otherwise. Then run as above with the real `--base-url` and `--model`.

The mock server is the deterministic baseline for plumbing, ledger, retry, and
assembly behavior; a real vLLM/SmolVLM server is for model compatibility and OCR
quality. See [docs/mock-api.md](docs/mock-api.md) for the differences.

## Capacity and profiles

`--capacity` selects a provider pressure profile that drives concurrency limits,
retry backoff, and the circuit breaker:

| Capacity | In-flight | Circuit threshold | Use |
| --- | --- | --- | --- |
| `local-vllm-small` (default) | 2 | 3 | a single local GPU |
| `local-vllm-large` | 8 | 5 | a larger local deployment |
| `remote-openai-compatible` | 4 | 4 | a hosted endpoint |

`--profile` selects the OCR prompt/parser. The default `generic_vlm_markdown`
(default model `generic-vlm-markdown`) suits most VLMs; `strict_json_ocr`,
`glm_ocr`, `lighton_ocr_2_1b`, and `deepseek_ocr_2` target specific models.

## Documentation

- [docs/cli.md](docs/cli.md) — every command, flag, example, and exit code.
- [docs/state-layout.md](docs/state-layout.md) — the on-disk `.paperscale/jobs/<job_id>` layout.
- [docs/mock-api.md](docs/mock-api.md) — the mock inference server.
- [docs/ledger-recovery.md](docs/ledger-recovery.md) — ledger safety, recovery, and capacity behavior.

## Development

```bash
poetry run pytest        # run the test suite (stdlib unittest, run via pytest)
poetry run ruff check .  # lint
poetry run pyrefly check src/paperscale  # type-check
```
