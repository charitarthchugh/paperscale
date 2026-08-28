"""Read whole Records out of an OCR Run's Dolma JSONL.

`evaluation/runs.py` already reads these files, but `load_run` flattens each
Record into `PageText` rows and drops every zero-length span on the way. embed
cannot use that. A page whose `natural_text` was None emits a **zero-width**
span, and to the packer that page is a real entry: it costs 0 tokens, it can
never force a Chunk break, and it still has to land inside some Chunk's
`first_page`/`last_page` range or the Chunk silently claims a page range with a
hole in it (design 5.1, 17.2 item 4). So nothing here interprets a Record --
text slicing, page numbering and name derivation all belong to the caller, and
what comes out of `iter_records` is the parsed line and nothing else.

Input resolution is `load_run`'s, on purpose: an operator who has been pointing
`paperscale evaluate` at a workspace should be able to point `paperscale embed`
at the same string. The rule is re-expressed here rather than called, because
`_iter_jsonl_paths` is private to `evaluation` and embed has no business
reaching into another subcommand's internals; the two are coupled by intent, so
a change to one is a change to both. `DuplicateSourceFileError` *is* imported --
it is public, and two spellings of one failure would let a caller catch one and
miss the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from paperscale.evaluation.runs import DuplicateSourceFileError

__all__ = ["DuplicateSourceFileError", "iter_records", "resolve_jsonl_paths"]


def resolve_jsonl_paths(path: str | Path) -> list[Path]:
    """Resolve one `--run` value into the concrete `.jsonl` files behind it.

    Three shapes: an OCR workspace, a bare directory of `*.jsonl`, or a single
    file. The workspace glob is tried **first** because a workspace keeps its
    Records under `results/` and has no `*.jsonl` at its own top level -- take
    the shapes in the other order and the most common input of the three
    resolves to zero files without an error.

    A directory that matches neither glob yields `[]` rather than raising. That
    is `_iter_jsonl_paths`'s behaviour and is kept only for that reason; the
    empty Run then reads as a `0/0` bar, which is indistinguishable from a fully
    resumed Invocation.
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"run path does not exist: {p}")
    results = sorted((p / "results").glob("*.jsonl"))
    if results:
        return results
    return sorted(p.glob("*.jsonl"))


def iter_records(path: str | Path) -> Iterator[dict]:
    """Yield every Record of one Run, whole, in file order.

    Raises `DuplicateSourceFileError` when two Records in this Run carry the
    same `Source-File`. They would derive one Document name, so the second would
    overwrite the first in every Sink and the corpus would come out one Document
    short with nothing reporting it -- the same ambiguity `load_run` refuses,
    for the same reason. The check is scoped to a single call because a Run is a
    single call; two Runs may legitimately hold the same PDF, which is what the
    `(run_label, document_name)` Resume key exists for.

    Records with no `Source-File` are deliberately **not** compared. `names.py`
    gives those the path digest as their whole name and `check_collisions`
    reports the residue with every raw value in the colliding group, so a
    nameless Record has one fatal-at-startup path instead of two that word the
    same problem differently.
    """
    seen: dict[str, Path] = {}
    for jsonl in resolve_jsonl_paths(path):
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                source_file = record.get("metadata", {}).get("Source-File") or ""
                if source_file:
                    first = seen.get(source_file)
                    if first is not None:
                        # Name both shards: a Run spanning results/output_*.jsonl can duplicate
                        # across files, and "which file" is the first thing the operator asks.
                        where = str(first) if first == jsonl else f"{first} and {jsonl}"
                        raise DuplicateSourceFileError(f"duplicate Source-File {source_file!r} in {where}")
                    seen[source_file] = jsonl
                yield record
