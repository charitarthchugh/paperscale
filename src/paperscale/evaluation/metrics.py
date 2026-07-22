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
