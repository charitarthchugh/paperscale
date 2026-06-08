# Paperscale mock inference API

`paperscale-mock-api` is a development-only FastAPI/Uvicorn server for deterministic end-to-end provider tests. It mimics the small OpenAI/vLLM-compatible surface Paperscale needs; it does not run a model or judge OCR quality.

## Install the optional server dependencies

The console script is import-safe in a normal Paperscale install, but serving the mock requires the optional `mock-api` extra:

```bash
pip install 'paperscale[mock-api]'
# or, inside this repo:
poetry install -E mock-api
```

## Start the server

```bash
poetry run paperscale-mock-api serve \
  --host 127.0.0.1 \
  --port 8009 \
  --model mock-vlm
```

Useful options:

- `--scenario ok_markdown` selects the initial deterministic response mode.
- `--max-image-bytes 10485760` caps decoded data-URL image size.
- `--max-in-flight 8` bounds concurrent inference requests; `0` forces `429` overloads.
- `--latency-ms 250` adds artificial latency.
- `--bearer-token secret` requires `Authorization: Bearer secret` on all endpoints.

## OpenAI SDK example

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8009/v1", api_key="test-key")
response = client.responses.create(
    model="mock-vlm",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Convert this page to Markdown"},
                {"type": "input_image", "image_url": "data:image/png;base64,..."},
            ],
        }
    ],
    temperature=0.0,
    top_p=1.0,
    max_output_tokens=4096,
)
print(response.output_text)
```

## Paperscale provider config shape

Use the existing OpenAI-compatible provider against the local base URL:

```python
from openai import OpenAI
from paperscale.profiles.builtin import get_builtin_profile
from paperscale.providers.openai_chat import OpenAIChatProvider

profile = get_builtin_profile("generic_vlm_markdown")
request = profile.build_request("doc-1:page-1", image_bytes, "image/png", model="mock-vlm")
provider = OpenAIChatProvider(
    client=OpenAI(base_url="http://127.0.0.1:8009/v1", api_key="test-key")
)
page = provider.send(request)
```

The mock also exposes vLLM-style `POST /v1/chat/completions` for future chat-completions adapters.

## Debug endpoints

- `GET /v1/models` lists the configured served model.
- `GET /__paperscale/requests` returns normalized request log entries with endpoint, model, image hash, prompt hash, scenario, and request id.
- `POST /__paperscale/reset` clears request logs and scenario counters.
- `POST /__paperscale/scenario` with `{"scenario": "repeated_ngram"}` changes the active scenario.

Errors use OpenAI-style JSON:

```json
{"error": {"message": "...", "type": "invalid_request_error", "code": "..."}}
```

## Scenarios

| Scenario | Behavior |
| --- | --- |
| `ok_markdown` | Stable Markdown containing model, endpoint, image hash, prompt hash, and request id. |
| `json_layout` | JSON string with `markdown` plus synthetic layout `regions`, useful for GLM-style parsers. |
| `empty_output` | Empty string to exercise retryable empty-output handling. |
| `refusal` | Refusal-like text. |
| `repeated_ngram` | Repeated tokens to trigger repetition guards. |
| `malformed_frontmatter` | Markdown with intentionally malformed frontmatter. |
| `truncated` | Shortened deterministic Markdown. |
| `rate_limit` | Always returns HTTP `429`. |
| `rate_limit_then_ok` | First request after selecting the scenario returns `429`; later requests succeed. |
| `server_error` | Always returns HTTP `500`. |
| `slow` | Sleeps for `--latency-ms`, or 250 ms by default, then returns stable Markdown. |

## Mock vs real vLLM/SmolVLM

Use this mock as the deterministic baseline for scheduler, ledger, provider, parser, retry, and end-to-end plumbing tests. Use optional real vLLM/SmolVLM tests when you need model-server compatibility or OCR quality signals. The mock intentionally does not implement full OpenAI/vLLM protocol coverage, streaming, GPU behavior, or quality evaluation.
