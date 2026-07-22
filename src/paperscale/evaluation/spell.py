"""Shared symspell dictionary + correction utilities.

Used by the required ``correction_rate`` metric (how much a spell checker has to
change the text) and by the opt-in ``pplx`` corrected-perplexity pass, so both
run over the *same* dictionary. symspell is a core dependency.
"""

from __future__ import annotations

import re

from symspellpy import SymSpell, Verbosity

_TOKEN_RE = re.compile(r"[A-Za-z]+")
_MAX_EDIT_DISTANCE = 2


def build_dictionary(extra_words: frozenset[str] = frozenset()) -> SymSpell:
    """Build a SymSpell dictionary from the 50k most common English words.

    Frequencies come from ``wordfreq``; ``extra_words`` are added with a low
    count so they pass lookups without shadowing real corrections.
    """
    from wordfreq import top_n_list, word_frequency

    sym = SymSpell(max_dictionary_edit_distance=_MAX_EDIT_DISTANCE)
    for word in top_n_list("en", 50_000):
        count = max(1, int(word_frequency(word, "en") * 1_000_000_000))
        sym.create_dictionary_entry(word, count)
    for word in extra_words:
        sym.create_dictionary_entry(word, 1)
    return sym


def correct_text(text: str, sym: SymSpell) -> str:
    """Spell-correct only alphabetic tokens that fail the dictionary lookup.

    Numbers, punctuation, whitespace, and already-known words are left exactly
    as-is, preserving char offsets between tokens so page boundaries stay
    meaningful.
    """

    def repl(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if tok.lower() in sym.words:  # known -> untouched (case preserved)
            return tok
        sugg = sym.lookup(tok, Verbosity.CLOSEST, max_edit_distance=_MAX_EDIT_DISTANCE)
        return sugg[0].term if sugg else tok

    return _TOKEN_RE.sub(repl, text)


def correction_counts(text: str, sym: SymSpell) -> tuple[int, int, int] | None:
    """Return ``(n_tokens, n_corrected, n_uncorrectable)`` over alphabetic tokens.

    Mirrors ``correct_text``'s decisions exactly:
    - a token in the dictionary is left alone (neither corrected nor uncorrectable);
    - a non-dictionary token with an edit-distance<=2 suggestion is *corrected*;
    - a non-dictionary token with no suggestion is *uncorrectable*.

    Returns ``None`` when the page has no alphabetic tokens (row skipped, not 0.0).
    """
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return None
    corrected = uncorrectable = 0
    for tok in tokens:
        if tok.lower() in sym.words:
            continue
        if sym.lookup(tok, Verbosity.CLOSEST, max_edit_distance=_MAX_EDIT_DISTANCE):
            corrected += 1
        else:
            uncorrectable += 1
    return len(tokens), corrected, uncorrectable
