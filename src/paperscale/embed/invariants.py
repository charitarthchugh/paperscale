"""The facts both Sinks agree on, and the one comparison that reports a disagreement.

Its own module because it belongs to neither Sink. It lived in `npz_sink` while the
`.npz` tree was the only thing that had a manifest, and `lance_sink` imported it from
there -- so the LanceDB Sink depended on the npz Sink for a vocabulary that was never
about `.npz` files, and `npz_sink` changed for two unrelated reasons: the on-disk array
format, and what two Invocations must agree about.

Nothing here knows how either Sink stores anything. That is the point: a third Sink
would import this module and not a sibling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class Invariants:
    """The nine manifest facts a second Invocation must agree with, or stop.

    This is the check the Adapter's `native_dim` assertion cannot make. That assertion
    catches a server serving a model of the wrong *width*; it says nothing about an
    Invocation appended to a tree an earlier Invocation built with a **different model of
    the same width** -- `qwen3-embedding-8b` over `nemotron-3-embed-8b`, both 4096. With
    content detection removed (standing decision 7) Resume will not catch that either: the
    names match, so every Document is skipped and the tree ends up half one model's
    vectors and half another's, in one search index, silently.

    `layout` is in the block for a related reason (design 11.3). The run-label directory
    appears only when one Invocation embeds more than one Run, so `bare` and `labelled`
    are two layouts over one output directory. The failure a change causes is not silent
    *mixing* but silent **duplication**: the new paths match nothing, every Document is
    re-embedded into a parallel subtree, and the old tree is orphaned.

    `sinks` is deliberately **not** here -- the enabled-Sink set is allowed to change, and
    a change warns rather than stops (design 8.2).
    """

    model_id: str
    stored_dim: int
    native_dim: int
    document_instruction: str
    query_instruction: str
    pooling: str
    chunker: str
    chunk_budget_tokens: int
    layout: str


class SinkInvariantError(RuntimeError):
    """This output tree was built by an Invocation whose settings differ from this one's."""


# What a fact reads as when the output records no value for it at all. A recorded fact and a
# missing one are both disagreements, and this is what the message shows for the second.
ABSENT = "<absent>"


def compare_invariant_facts(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    subject: str,
    holder: str,
    nothing_yet: str,
    remedy: str,
    notes: Mapping[str, str] | None = None,
) -> None:
    """Report **every** disagreeing fact with **both** values, then stop.

    Every fact, not the first: an output built with a different model usually differs in
    several at once (`model_id`, both Instructions, often `native_dim`), and reporting one at
    a time turns one operator decision into four failed runs.

    Both Sinks compare invariant facts and both stop on disagreement, so the comparison and
    the shape of the message live here once rather than twice. What differs between them is
    only what they *name*: `subject` is the artifact that disagrees (the manifest, or one
    `.lance` table), `holder` is how the message refers to it (`this tree`, `this table`),
    `nothing_yet` says what has not happened yet, and `remedy` names the fix -- which is not
    the same fix, since the `.npz` manifest can be replaced by embedding into a fresh `--out`
    and LanceDB table metadata is write-once. `notes` adds a line per fact whose fix is not
    obvious from the two values alone, and only when that fact is one of the ones that
    disagreed; the caller supplies its indentation.

    A fact the output does not record at all is a disagreement, not a match -- otherwise a
    table with no metadata at all would read as agreeing with everything.
    """
    diffs = [
        (fact, recorded[fact] if fact in recorded else ABSENT, value) for fact, value in current.items() if fact not in recorded or recorded[fact] != value
    ]
    if not diffs:
        return

    lines = [f"  {fact}: {holder} has {was!r}, this Invocation has {now!r}" for fact, was, now in diffs]
    changed = {fact for fact, _was, _now in diffs}
    lines.extend(note for fact, note in (notes or {}).items() if fact in changed)
    raise SinkInvariantError(
        f"{subject} was built by a different Invocation: {len(diffs)} invariant fact(s) disagree; {nothing_yet}\n" + "\n".join(lines) + f"\n  {remedy}"
    )
