from __future__ import annotations

import argparse
from pathlib import Path
import sys

from paperscale.assembly import assemble_document_markdown, load_page_markdown_artifacts, write_document_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperscale",
        description="Paperscale document-to-Markdown tooling for v1 OCR workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble", help="assemble page Markdown into document Markdown")
    assemble.add_argument("--input", required=True, help="JSONL file containing page Markdown artifacts")
    assemble.add_argument("--output", required=True, help="output Markdown file")
    assemble.add_argument("--title", default=None, help="document title")
    assemble.add_argument("--enforce-quality", action="store_true", help="reject low-quality fragments")
    assemble.set_defaults(func=_assemble_command)

    for name in ("run", "status", "resume", "reconcile", "fsck", "repair-index"):
        cmd = subparsers.add_parser(name, help=f"placeholder {name} command")
        cmd.set_defaults(func=_placeholder_command)

    doctor = subparsers.add_parser("doctor", help="diagnose provider configuration")
    doctor_subparsers = doctor.add_subparsers(dest="doctor_command", required=True)
    provider = doctor_subparsers.add_parser("provider", help="check provider reachability and compatibility")
    provider.set_defaults(func=_placeholder_command)

    return parser


def _assemble_command(args: argparse.Namespace) -> int:
    artifacts = load_page_markdown_artifacts(args.input)
    markdown = assemble_document_markdown(
        artifacts,
        title=args.title,
        enforce_quality=args.enforce_quality,
    )
    write_document_markdown(args.output, markdown)
    print(f"assembled {len(artifacts)} pages to {args.output}")
    return 0


def _placeholder_command(args: argparse.Namespace) -> int:
    del args
    print("not implemented in this slice")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
