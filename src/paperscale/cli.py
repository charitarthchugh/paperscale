"""Paperscale command line interface for v1 document-to-Markdown OCR workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from paperscale.assembly import PageMarkdownArtifact, assemble_document_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperscale",
        description="Local-first VLM OCR runner for public v1 document-to-Markdown workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble_parser = subparsers.add_parser(
        "assemble",
        help="assemble completed page OCR Markdown artifacts into a document",
        description="Assemble completed page OCR Markdown artifacts into one document-to-Markdown output.",
    )
    assemble_parser.add_argument("--input", required=True, type=Path, help="JSONL file of completed page artifacts")
    assemble_parser.add_argument("--output", required=True, type=Path, help="Markdown output path")
    assemble_parser.add_argument("--title", help="optional document title to prepend as an H1")
    assemble_parser.add_argument(
        "--enforce-quality",
        action="store_true",
        help="reject empty, mojibake, or repeated OCR fragments before writing output",
    )
    assemble_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="mark the assembled Markdown as partial",
    )
    assemble_parser.set_defaults(handler=_handle_assemble)

    for name, description in {
        "run": "run page OCR for a document-to-Markdown workload",
        "status": "show compact-index workload status",
        "resume": "resume pending work using compact indexes",
        "reconcile": "surface ambiguous attempts and duplicate-call risk",
        "fsck": "repair-only filesystem consistency check",
        "repair-index": "repair-only compact index rebuild",
    }.items():
        command_parser = subparsers.add_parser(name, help=description, description=description)
        command_parser.set_defaults(handler=_handle_not_yet_integrated)

    doctor_parser = subparsers.add_parser("doctor", help="diagnose provider configuration")
    doctor_subparsers = doctor_parser.add_subparsers(dest="doctor_command", required=True)
    provider_parser = doctor_subparsers.add_parser(
        "provider",
        help="validate provider reachability and OCR profile compatibility",
        description="Validate provider reachability and OCR profile compatibility without starting OCR work.",
    )
    provider_parser.set_defaults(handler=_handle_provider_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))


def _handle_assemble(args: argparse.Namespace) -> int:
    pages = _read_page_artifacts(args.input)
    markdown = assemble_document_markdown(
        pages,
        title=args.title,
        enforce_quality=args.enforce_quality,
        partial=args.allow_partial,
    )
    _atomic_write_text(args.output, markdown)
    print(f"assembled {len(pages)} pages into {args.output}")
    return 0


def _handle_not_yet_integrated(args: argparse.Namespace) -> int:
    command = args.command
    raise SystemExit(
        f"paperscale {command} is reserved for the document-to-Markdown OCR pipeline and awaits "
        "state/ledger/provider integration; no provider call was started."
    )


def _handle_provider_doctor(args: argparse.Namespace) -> int:
    del args
    print("provider diagnostics are not yet integrated in this slice")
    return 0


def _read_page_artifacts(path: Path) -> list[PageMarkdownArtifact]:
    pages: list[PageMarkdownArtifact] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            pages.append(_page_from_json(payload, line_number=line_number))
    return pages


def _page_from_json(payload: dict[str, Any], *, line_number: int) -> PageMarkdownArtifact:
    try:
        document_id = payload["document_id"]
        page_number = payload["page_number"]
        markdown = payload["markdown"]
    except KeyError as exc:
        raise ValueError(f"page artifact line {line_number} is missing required field {exc.args[0]!r}") from exc

    if not isinstance(document_id, str) or not document_id:
        raise ValueError(f"page artifact line {line_number} has invalid document_id")
    if not isinstance(page_number, int):
        raise ValueError(f"page artifact line {line_number} has invalid page_number")
    if not isinstance(markdown, str):
        raise ValueError(f"page artifact line {line_number} has invalid markdown")
    return PageMarkdownArtifact(document_id=document_id, page_number=page_number, markdown=markdown)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
        _fsync_parent(path.parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_parent(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
