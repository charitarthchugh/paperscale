"""Deterministic quality gates for VLM OCR Markdown fragments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# Phrases only an assistant declining a task produces — safe to match anywhere.
_REFUSAL_PATTERNS = (
    r"\bas an ai\b",
    r"\bi am unable to help\b",
    r"\bi can(?:'|)t help\b",
    r"\bi cannot help\b",
    r"\bi cannot provide\b",
    r"\bi cannot comply\b",
)
# Polite phrases that also appear verbatim in real documents (deposition
# transcripts, letters). Only a refusal when they dominate a SHORT output; in a
# full transcribed page they are quoted speech, not the model refusing.
_POLITE_REFUSAL_PATTERNS = (
    r"\bi(?:'|)m sorry\b",
    r"\bsorry,? but\b",
)
# A genuine refusal is brief and is the whole response; transcribed pages that
# merely quote an apology run far longer than this.
_REFUSAL_MAX_LEN = 600


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """A deterministic issue found in an OCR Markdown fragment."""

    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Quality verdict for a Markdown fragment."""

    accepted: bool
    severity: str
    issues: list[QualityIssue]


@dataclass(frozen=True, slots=True)
class VerificationFinding:
    """Persistable verifier finding attached to a page artifact."""

    accepted: bool
    kind: str
    retry_class: str
    warnings: list[str] = field(default_factory=list)


# Every check ``assess_markdown_fragment`` can raise, in the order it runs them.
# This is the vocabulary accepted by ``--disable-quality-check``.
QUALITY_CHECK_CODES = (
    "empty_output",
    "mojibake",
    "control_characters",
    "refusal_boilerplate",
    "malformed_frontmatter",
    "repeated_character",
    "repeated_ngram",
    "repeated_tail",
    "truncation_indicator",
    "length_anomaly",
)

#: Accepted by ``--disable-quality-check`` alongside the individual codes.
DISABLE_ALL_CHECKS = "all"


def expand_disabled_checks(names: list[str]) -> frozenset[str]:
    """Turn ``--disable-quality-check`` values into the set of codes to skip."""
    if DISABLE_ALL_CHECKS in names:
        return frozenset(QUALITY_CHECK_CODES)
    return frozenset(names)


class DeterministicQualityVerifier:
    """Local, no-extra-model verifier for v1 page Markdown fragments."""

    def __init__(self, optional_slm: object | None = None, disabled_checks: frozenset[str] = frozenset()) -> None:
        self.optional_slm = optional_slm
        self.disabled_checks = disabled_checks

    def classify(self, markdown: str) -> VerificationFinding:
        report = assess_markdown_fragment(markdown, disabled_checks=self.disabled_checks)
        if report.accepted:
            return VerificationFinding(True, "ok", "none", warnings=[])
        issue = report.issues[0]
        kind = _public_kind(issue.code)
        retry_class = "terminal" if kind == "refusal" else "retryable"
        return VerificationFinding(False, kind, retry_class, warnings=[])


def assess_markdown_fragment(markdown: str, disabled_checks: frozenset[str] = frozenset()) -> QualityReport:
    """Assess whether a page Markdown fragment is coherent enough to assemble.

    ``disabled_checks`` names codes from :data:`QUALITY_CHECK_CODES` to skip, so a
    corpus that trips a gate systematically can be processed without it rather
    than by loosening ``--max_page_error_rate`` for every failure mode at once.
    """

    issues: list[QualityIssue] = []
    text = markdown.strip()

    def enabled(code: str) -> bool:
        return code not in disabled_checks

    if not text:
        if enabled("empty_output"):
            issues.append(QualityIssue("empty_output", "OCR output is empty or whitespace only."))
            return QualityReport(accepted=False, severity="error", issues=issues)
        return QualityReport(accepted=True, severity="ok", issues=issues)

    replacement_count = text.count("\ufffd")
    if enabled("mojibake") and (replacement_count >= 3 or replacement_count / max(len(text), 1) > 0.02):
        issues.append(QualityIssue("mojibake", "OCR output contains too many Unicode replacement characters."))

    if enabled("control_characters") and _CONTROL_CHAR_RE.search(text):
        issues.append(QualityIssue("control_characters", "OCR output contains control characters."))

    if enabled("refusal_boilerplate") and _has_refusal_boilerplate(text):
        issues.append(QualityIssue("refusal_boilerplate", "OCR output contains refusal boilerplate."))

    if enabled("malformed_frontmatter") and _has_malformed_frontmatter(text):
        issues.append(QualityIssue("malformed_frontmatter", "OCR output has malformed frontmatter or schema preamble."))

    if enabled("repeated_character") and _has_repeated_character_run(text):
        issues.append(QualityIssue("repeated_character", "OCR output contains an abnormal repeated-character run."))

    if enabled("repeated_ngram") and _has_repeated_ngram_loop(text):
        issues.append(QualityIssue("repeated_ngram", "OCR output appears to repeat the same phrase."))

    if enabled("repeated_tail") and _has_repeated_tail_loop(text):
        issues.append(QualityIssue("repeated_tail", "OCR output ends in a repeating loop."))

    if enabled("truncation_indicator") and _has_truncation_indicator(text):
        issues.append(QualityIssue("truncation_indicator", "OCR output appears truncated."))

    if enabled("length_anomaly") and _has_length_anomaly(text):
        issues.append(QualityIssue("length_anomaly", "OCR output length looks anomalous."))

    accepted = not any(issue.severity == "error" for issue in issues)
    return QualityReport(accepted=accepted, severity="ok" if accepted else "error", issues=issues)


def _public_kind(code: str) -> str:
    if code == "refusal_boilerplate":
        return "refusal"
    return code


# Characters that legitimately tile in documents and so do not signal a model
# loop: form-field blanks (____), dotted leaders (....), rules and separators
# (--- *** === ~~~), bullets, and en/em dashes. A long contiguous run of these is
# page furniture, not degeneration. Genuine character loops repeat letters,
# digits, or other punctuation; space-separated loops (e.g. "— — —") are caught
# by the n-gram gate instead, so excluding the dashes here does not hide them.
_FILL_RUN_CHARS = frozenset("_.-–—·•*=~")


def _has_repeated_character_run(text: str) -> bool:
    current = ""
    run_length = 0
    for char in text:
        if char == current:
            run_length += 1
        else:
            current = char
            run_length = 1
        if not char.isspace() and char not in _FILL_RUN_CHARS and run_length >= 24:
            return True
    return False


def _has_repeated_ngram_loop(text: str) -> bool:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text) if token.strip()]
    if len(tokens) >= 6:
        most_common = max((tokens.count(token) for token in set(tokens)), default=0)
        if most_common >= 6 and most_common / len(tokens) >= 0.75:
            return True
    if len(tokens) < 24:
        return False

    for ngram_size in (2, 3, 4, 5):
        if len(tokens) < ngram_size * 8:
            continue
        ngrams = zip(*(tokens[offset:] for offset in range(ngram_size)), strict=False)
        counts = Counter(tuple(ngram) for ngram in ngrams)
        _ngram, count = counts.most_common(1)[0]
        if count >= 12 and (count * ngram_size) / len(tokens) >= 0.35:
            return True
    return False


# Degeneration that runs to the end of the output: the model got stuck emitting
# one unit until it ran out of budget. _has_repeated_ngram_loop cannot see this
# once the unit grows: it scores ``count * ngram_size / len(tokens)`` with
# ngram_size capped at 5, so the score tops out at ``5 / period`` and a looped
# *sentence* (13+ tokens) stays under the 0.35 threshold no matter how much of
# the page it eats — measured at 0.35 even when the loop is 90% of the output.
# Walking back from the end at a fixed period instead makes period length
# irrelevant. The period/repeat/length constants are OvisOCR2's vendor cleaner
# (https://huggingface.co/ATH-MaaS/OvisOCR2), reused here as a *detector* so the
# page is retried rather than silently rewritten.
_TAIL_LOOP_MAX_PERIOD = 200
_TAIL_LOOP_MIN_REPEATS = 5
_TAIL_LOOP_MIN_CHARS = 100
# The loop must also dominate the output, reusing the n-gram gate's 0.35 share.
# This is what separates a genuine loop from legitimately tiling content — a form
# whose trailing table rows are identical blanks sits near 0.23 and is kept.
_TAIL_LOOP_MIN_SHARE = 0.35
# The loop need not reach the final character. A model that loops table rows and
# then closes the tag leaves "</table>" behind it, and OvisOCR2 emits HTML tables,
# so that is the common shape — anchoring at len(text) misses it entirely. Probe a
# ladder of end positions so a trailer of ordinary text after the loop cannot hide
# it. Stepping back one character at a time does not work: on prose a character
# coincidentally lines up at the period within a few steps, which stops the search
# inside the trailer rather than at the end of the loop.
_TAIL_LOOP_TRAILERS = (0, 8, 24, 64, 160, 400)


def _has_repeated_tail_loop(text: str) -> bool:
    length = len(text)
    if length < _TAIL_LOOP_MIN_CHARS:
        return False
    # All three thresholds grow monotonically with the length of the repeating run,
    # so for a given period and end position only ``span`` characters decide the
    # answer: if that slice repeats at ``period``, every threshold is met, and if it
    # does not, no longer run ends there either. Testing it as one slice equality
    # keeps the scan in C — walking character by character in Python costs ~300ms on
    # a page whose tail repeats just under the share threshold.
    share_floor = math.ceil(_TAIL_LOOP_MIN_SHARE * length)
    for trailer in _TAIL_LOOP_TRAILERS:
        end = length - trailer
        if end < _TAIL_LOOP_MIN_CHARS:
            break  # trailers only grow, so no later one leaves room either
        for period in range(1, _TAIL_LOOP_MAX_PERIOD + 1):
            span = max(_TAIL_LOOP_MIN_CHARS, _TAIL_LOOP_MIN_REPEATS * period, share_floor)
            if span > end:
                break  # span only grows with period, so no larger period fits either
            tail = text[end - span : end]
            if tail[period:] == tail[:-period]:
                return True
    return False


def _has_refusal_boilerplate(text: str) -> bool:
    lowered = text.lower()
    if any(re.search(pattern, lowered) for pattern in _REFUSAL_PATTERNS):
        return True
    if len(text.strip()) <= _REFUSAL_MAX_LEN:
        return any(re.search(pattern, lowered) for pattern in _POLITE_REFUSAL_PATTERNS)
    return False


def _has_malformed_frontmatter(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return False
    lines = stripped.splitlines()
    if len(lines) == 1:
        return True
    closing_index = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if closing_index is None:
        return True
    frontmatter = lines[1:closing_index]
    return any(line.count("[") != line.count("]") or line.count("{") != line.count("}") for line in frontmatter)


def _has_truncation_indicator(text: str) -> bool:
    lowered = text.rstrip().lower()
    return lowered.endswith("...") or lowered.endswith("[...]") or "truncated" in lowered


def _has_length_anomaly(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < 5 or len(stripped) > 50_000
