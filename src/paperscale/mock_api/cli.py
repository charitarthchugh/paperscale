from __future__ import annotations

import argparse
import sys
from typing import Sequence

SCENARIOS = (
    "empty_output",
    "json_layout",
    "malformed_frontmatter",
    "ok_markdown",
    "rate_limit",
    "rate_limit_then_ok",
    "refusal",
    "repeated_ngram",
    "server_error",
    "slow",
    "truncated",
)

_MISSING_EXTRA_MESSAGE = (
    "paperscale-mock-api requires the optional mock API dependencies. "
    "Install with `pip install 'paperscale[mock-api]'` or run `poetry install -E mock-api`."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperscale-mock-api",
        description="Run the Paperscale deterministic OpenAI/vLLM-compatible mock inference API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the mock inference API server")
    serve.add_argument("--host", default="127.0.0.1", help="bind host")
    serve.add_argument("--port", default=8009, type=int, help="bind port")
    serve.add_argument("--model", default="mock-vlm", help="served model id")
    serve.add_argument("--scenario", default="ok_markdown", choices=SCENARIOS, help="initial response scenario")
    serve.add_argument("--max-image-bytes", default=10 * 1024 * 1024, type=int, help="maximum decoded image size")
    serve.add_argument(
        "--max-in-flight",
        default=8,
        type=int,
        help="maximum concurrent inference requests; 0 always returns 429",
    )
    serve.add_argument("--latency-ms", default=0, type=int, help="artificial per-request latency")
    serve.add_argument("--bearer-token", help="optional bearer token required by all endpoints")
    serve.set_defaults(handler=_handle_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from paperscale.mock_api.app import MockApiConfig, create_app
    except ModuleNotFoundError as exc:
        if exc.name in {"fastapi", "uvicorn"} or str(exc) in {"fastapi", "uvicorn"}:
            print(_MISSING_EXTRA_MESSAGE, file=sys.stderr)
            return 2
        raise

    config = MockApiConfig(
        served_model=args.model,
        scenario=args.scenario,
        max_image_bytes=args.max_image_bytes,
        max_in_flight=args.max_in_flight,
        latency_ms=args.latency_ms,
        bearer_token=args.bearer_token,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
