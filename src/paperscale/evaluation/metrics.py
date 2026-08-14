"""Pure per-page text-quality metrics. No I/O, no run/db knowledge.

All functions operate on a single string (or a pair of strings). Functions that
can be undefined for a page (no scorable tokens) return ``None`` so the caller
drops the row rather than averaging in a misleading 0.0.
"""

from __future__ import annotations

import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

_WORD = re.compile(r"\w+", re.UNICODE)
_VOWELS = set("aeiouyAEIOUY")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [text](url) -> text
_MD_CHARS = re.compile(r"[#*_>`~]")
_WS = re.compile(r"\s+")


def _is_garbage(tok: str) -> bool:
    # >=4 identical consecutive chars (aaaa, ----)
    if re.search(r"(.)\1{3,}", tok):
        return True
    core = tok.strip("[](){}.,;:!?\"'")
    if len(core) > 40:  # absurdly long unbroken run
        return True
    letters = [c for c in core if c.isalpha()]
    # alpha-digit soup: mixes letters and digits within one token
    if letters and any(c.isdigit() for c in core):
        return True
    # vowel-less alphabetic run (OCR garble like "brwn"/"jmps"); skip ALL-CAPS so
    # real acronyms (PDF, XML, CSV) aren't flagged. _VOWELS includes 'y'.
    if len(letters) > 2 and not core.isupper() and not any(c in _VOWELS for c in letters):
        return True
    # mid-word case alternation: >=2 lower->upper transitions inside the token
    transitions = sum(1 for a, b in zip(core, core[1:]) if a.islower() and b.isupper())
    if transitions >= 2:
        return True
    return False


def garbage_token_fraction(text: str) -> float | None:
    """Fraction of whitespace tokens that look like OCR garbage. None if empty."""
    tokens = text.split()
    if not tokens:
        return None
    return sum(1 for t in tokens if _is_garbage(t)) / len(tokens)


def normalize_markdown(text: str) -> str:
    """OmniDocBench-style structural normalization (no lowercasing).

    Strip link syntax to its anchor text, drop markdown emphasis/heading/quote/code
    markers, and collapse all whitespace. Used before edit-distance comparison so
    formatting differences don't dominate.
    """
    text = _LINK.sub(r"\1", text)
    text = _MD_CHARS.sub("", text)
    return _WS.sub(" ", text).strip()


def bow_f1(a: str, b: str) -> float:
    """Multiset bag-of-words F1 between two texts (order-insensitive)."""
    ca, cb = Counter(_WORD.findall(a.lower())), Counter(_WORD.findall(b.lower()))
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(ca.values())
    recall = overlap / sum(cb.values())
    return 2 * precision * recall / (precision + recall)


def one_minus_ned(a: str, b: str) -> float:
    """1 - normalized edit distance on normalized markdown (order-sensitive)."""
    na, nb = normalize_markdown(a), normalize_markdown(b)
    if not na and not nb:
        return 1.0
    return Levenshtein.normalized_similarity(na, nb)


def missing_peer_pairs(
    models: list[str], stored: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Which unordered model pairs on one page still need scoring.

    Peer agreement is the one metric that resumes on the *pair* rather than the
    doc: adding a third run leaves a-b valid but needs a-c and b-c. The stored
    peer_agreement rows are their own bookkeeping here (no eval_doc entry) --
    unlike textlayer, this metric always writes rows for what it computes, so
    row presence is a truthful record of what is done.

    ``models``  models present on this page, in any order.
    ``stored``  directed ``(model, peer)`` pairs already in the DB for this page.
    Returns sorted ``(m, peer)`` tuples with ``m < peer``, so the caller gets a
    deterministic work list and each unordered pair appears at most once.
    """
    ordered = sorted(models)
    missing: list[tuple[str, str]] = []
    for i, m in enumerate(ordered):
        for peer in ordered[i + 1:]:
            # Both directions must be present. A half-written pair would let the
            # leaderboard average m's score against peer while peer has no matching
            # row, skewing peer's mean; recomputing re-emits both and self-heals.
            if (m, peer) not in stored or (peer, m) not in stored:
                missing.append((m, peer))
    return missing


def peer_rows_for_page(
    item: tuple[tuple[str, int], dict[str, str], list[tuple[str, str]]],
) -> list[tuple]:
    """Peer-agreement DB rows for one page:
    ``((doc, page), {model: text}, [(m, peer), ...])`` ->
    ``[(model, peer, doc, page, bow_f1, one_minus_ned), ...]``.

    The caller supplies the pairs to score (see `missing_peer_pairs`) so a resumed
    run recomputes only what is absent. Both metrics are symmetric, so each pair is
    computed once and emitted in both directions. Top-level so it pickles into
    worker processes.
    """
    (doc, page), by_model, pairs = item
    rows: list[tuple] = []
    for m, peer in pairs:
        f1 = bow_f1(by_model[m], by_model[peer])
        ned = one_minus_ned(by_model[m], by_model[peer])
        rows.append((m, peer, doc, page, f1, ned))
        rows.append((peer, m, doc, page, f1, ned))
    return rows
