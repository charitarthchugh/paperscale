"""Resume: one question per Document -- "do I know this name?"

**State is derived from the outputs, never recorded** (design 11.1). There is no manifest of names
and no flag files, and both arguments for one dissolved. *"One cheap lookup instead of a stat per
Document"*: measured on btrfs over 20,000 Documents (40,000 files), `os.walk` collects every name in
29 ms while reading a 1.3 MB JSON manifest of the same names takes 3 ms -- the manifest buys 26
milliseconds, once per Invocation. *"It survives a Sink on a remote"*: both Sinks are local. What is
left is the argument against it -- a manifest is a second source of truth that a crash can
desynchronise from the first, and making it trustworthy needs exactly the careful write ordering it
was supposed to spare us. **Derived state cannot drift, because the evidence *is* the work.**

So this module holds no filesystem and no database code at all. Each Sink derives its own set (one
`os.walk` over `<out>` for `.npz`, one `SELECT document_name, run_label` for LanceDB, both read once
at startup and never inside the Document loop) and Resume only combines them. The combination is the
whole decision, and it is the **intersection** -- see `derive_resume_state`.

There is no content-change detection anywhere in `embed` (standing decision 7), which is a real cost
to the operator and an invisible one. `RE_OCR_WARNING` below is the paragraph that tells them; the
README reproduces it verbatim, so this is the single authoritative copy.

Design authority: `docs/design/embed.md` section 11, plus 14.6 for the `--no-resume` divergence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "LAYOUT_BARE",
    "LAYOUT_LABELLED",
    "RE_OCR_WARNING",
    "KnownSink",
    "LayoutChangeError",
    "check_layout",
    "derive_resume_state",
    "invocation_layout",
    "sink_set_warning",
]

# The two values of the `layout` invariant manifest fact. They are constants rather than bare
# literals because the string is compared across Invocations: a typo in one of the two producers
# (the manifest writer and this guard) would read as a layout change and stop every second run.
LAYOUT_BARE = "bare"
LAYOUT_LABELLED = "labelled"

# Placed here, and only here, because design 19 requires the README to carry it verbatim: two
# copies of a paragraph this load-bearing drift apart, and the drift would be in the one document
# that describes a failure the tool deliberately cannot detect.
RE_OCR_WARNING = """\
**Re-OCR-ing a corpus leaves stale vectors, and `embed` will not notice.**

Resume asks one question about each Document: *have I seen this name before?* It does not look at
the text. If you re-run OCR over the same PDFs -- with a different model, different settings, or a
newer version -- the Documents keep their names, and `embed` will skip every one of them. The
vectors in your output will continue to describe the *old* text, and nothing in the tool will tell
you.

`embed` does check two things before it starts, and stops the run if either disagrees: that the
embedding model and its settings match what built the output, and that the output layout has not
changed. Neither of those notices changed *text*.

After a re-OCR, either embed into a fresh output directory, or pass `--no-resume` to re-embed
everything in place.
"""


class LayoutChangeError(RuntimeError):
    """A second Invocation would write a different `.npz` layout over an existing output."""


class KnownSink(Protocol):
    """The only thing Resume needs from a Sink.

    Structural on purpose: `resume` importing `npz_sink` would drag numpy into an OCR-only
    install through the `embed` package, and importing `lance_sink` would drag lancedb in. It
    also lets a test stand in a Sink that is four lines long.
    """

    def known(self) -> set[tuple[str, str]]:
        """`{(run_label, document_name)}` this Sink already holds."""


def invocation_layout(run_count: int) -> str:
    """Which `.npz` layout this Invocation writes, from the number of Runs it embeds.

    The run-label directory exists to keep two Runs' identical Document names apart, so it appears
    only when there are two Runs to keep apart. One Run mirrors straight into `<out>`.
    """
    return LAYOUT_BARE if run_count <= 1 else LAYOUT_LABELLED


def check_layout(*, out: Path | str, recorded: str, current: str) -> None:
    """Stop before writing a `.npz` layout the output directory was not built with.

    With derived state the failure a layout change causes is not silent *mixing* -- it is silent
    **duplication**. Every Document's path moves by one directory level, so nothing matches, the
    whole corpus is embedded again into a parallel subtree, and the tree the Consumer is reading
    is orphaned in place. Nothing in the run looks wrong while it happens; it just costs a day.

    Relayouting on demand was rejected: moving a Consumer's files to spare them a flag is a large
    act for a small convenience.

    `.npz`-specific by design. LanceDB has no filesystem layout and its `run_label` column already
    separates Runs, so the same change is a no-op there.
    """
    if recorded == current:
        return
    raise LayoutChangeError(
        f"output layout changed: {out} was built with layout {recorded!r}, this Invocation would write {current!r}.\n"
        "  Resume state is derived from the output paths, so this does not mix the two trees -- it duplicates them: "
        "under the new layout every Document matches nothing,\n"
        "  is embedded again into a parallel subtree, and the existing tree is orphaned.\n"
        f"  Embed the same set of --run inputs as before (one Run gives {LAYOUT_BARE!r}, two or more give {LAYOUT_LABELLED!r}), "
        "or point --out at a fresh directory."
    )


def sink_set_warning(recorded: list[str], enabled: list[str], *, corpus_size: int | None = None) -> str | None:
    """The line to print when the enabled-Sink set differs from the one that built this output.

    A change **warns and never stops** -- the set is deliberately outside the manifest's invariant
    block, because adding a Sink is a legitimate thing to want. It is also expensive in a way that
    is invisible until the run is over: a Document is done only when *every* enabled Sink holds it,
    so a Sink enabled today knows nothing, the intersection is empty, and the whole corpus is
    embedded again. Saying so before the first request is the entire point -- the design's shape is
    *make the expensive silent thing loud*, because standing decision 7 removed the mechanism that
    would otherwise notice.

    Returns `None` when the sets match (order and duplicates ignored -- they are a set on the wire
    and a list only because JSON has no set).
    """
    was = set(recorded)
    now = set(enabled)
    if was == now:
        return None

    parts = [f"the enabled Sink set changed: this output was built with {sorted(was)}, this Invocation enables {sorted(now)}."]
    added = sorted(now - was)
    dropped = sorted(was - now)
    if added:
        scale = "the whole corpus" if corpus_size is None else f"{corpus_size:,} Documents"
        parts.append(
            f"{_sink_labels(added)} {'is' if len(added) == 1 else 'are'} new; a Document is done only when every enabled Sink "
            f"holds it, so this will re-embed {scale}."
        )
    if dropped:
        parts.append(
            f"{_sink_labels(dropped)} {'is' if len(dropped) == 1 else 'are'} no longer enabled; the existing output is left "
            "exactly as it is and stops being updated."
        )
    return " ".join(parts)


def derive_resume_state(sinks: list[KnownSink], *, no_resume: bool) -> set[tuple[str, str]]:
    """`{(run_label, document_name)}` to skip: the **intersection** of every enabled Sink's set.

    **A Document is done when every enabled Sink holds it.** With one Sink the intersection is that
    Sink's own set, so the rule needs no special case for the common configuration.

    The intersection exploits a difference between the two Sinks rather than fighting it: they have
    different tolerances for a double write -- LanceDB upserts harmlessly, the `.npz` pair rewrites
    harmlessly through rename. A crash between the two leaves a Document in one and not the other,
    the intersection says "not done", the next Invocation embeds it again and writes both, and one
    of those writes is a no-op while the other completes. **The gap heals itself, and it heals
    without anyone detecting that it happened.**

    Two consequences, both accepted knowingly:

    * **The batched Sink sets the pace.** LanceDB lags the `.npz` tree by up to one batch (64
      Documents), so a crash re-embeds up to that many Documents the `.npz` Sink already holds.
      That is the price of batching, paid in GPU time.
    * **Enabling a Sink later re-embeds the corpus.** `sink_set_warning` is what makes that loud
      before it happens rather than after.

    `no_resume` returns the empty set and **deletes nothing**, which diverges from the OCR
    pipeline's `--no-resume` on purpose (design 11.5/14.6): `_wipe_workspace_progress`
    (`pipeline.py:1147`) `rmtree`s an OCR workspace because a workspace is scratch, while an embed
    output is the deliverable and may already be open in a Consumer, and one pair of LanceDB tables
    holds several Runs -- so wiping would mean a scoped delete built from unnormalized paths, which
    is exactly where the `O'Brien` quoting hazard lives. Both Sinks are idempotent, so ignoring
    prior state is sufficient: everything is embedded again and overwritten in place.
    """
    if no_resume:
        # The divergence is invisible in the outputs (they are simply overwritten), so it is said
        # out loud at the moment it applies. Design 14.6: "state it, or it reads as a bug".
        logger.info(
            "--no-resume: ignoring prior progress. Every Document is embedded again and its outputs are overwritten in place; nothing on disk is deleted."
        )
        return set()

    if not sinks:
        # Unreachable through the CLI, which rejects --no-npz without --lancedb. Worth an explicit
        # answer anyway, because the mathematical one is the dangerous one: an intersection over no
        # sets is "everything", which would skip the entire corpus and report a successful no-op.
        return set()

    # Every Sink is read, even once the intersection is empty. Short-circuiting would make the
    # number of reads depend on the data, and a Sink's known() is also the read that tells it what
    # it already holds.
    per_sink = [(type(sink).__name__, set(sink.known())) for sink in sinks]

    state = set(per_sink[0][1])
    for _, known in per_sink[1:]:
        state &= known

    # The count is worth a line on its own (design 12.1 step 11), and the per-Sink counts beside it
    # are what distinguishes "nothing to do" from "a Sink was enabled today and knows nothing".
    logger.info(
        "Resume: %s. %d Document(s) are held by every enabled Sink and will be skipped.",
        "; ".join(f"{name} knows {len(known)}" for name, known in per_sink),
        len(state),
    )
    return state


def _sink_labels(names: list[str]) -> str:
    """Name a Sink the way the operator turned it on, so the warning names the fix."""
    flags = {"lancedb": "--lancedb", "npz": "the .npz Sink"}
    return " and ".join(flags.get(name, repr(name)) for name in names)
