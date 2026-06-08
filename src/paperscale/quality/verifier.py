"""Deterministic quality gates for VLM OCR Markdown fragments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_REFUSAL_PATTERNS = (
    r"\bas an ai\b",
    r"\bi am unable to help\b",
    r"\bi can(?:'|)t help\b",
    r"\bi cannot help\b",
    r"\bi(?:'|)m sorry\b",
    r"\bsorry,? but\b",
    r"\bcannot provide\b",
    r"\bi cannot comply\b",
)


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


class DeterministicQualityVerifier:
    """Local, no-extra-model verifier for v1 page Markdown fragments."""

    def __init__(self, optional_slm: object | None = None) -> None:
        self.optional_slm = optional_slm

    def classify(self, markdown: str) -> VerificationFinding:
        report = assess_markdown_fragment(markdown)
        if report.accepted:
            return VerificationFinding(True, "ok", "none", warnings=[])
        issue = report.issues[0]
        kind = _public_kind(issue.code)
        retry_class = "terminal" if kind == "refusal" else "retryable"
        return VerificationFinding(False, kind, retry_class, warnings=[])


def assess_markdown_fragment(markdown: str) -> QualityReport:
    """Assess whether a page Markdown fragment is coherent enough to assemble."""

    issues: list[QualityIssue] = []
    text = markdown.strip()

    if not text:
        issues.append(QualityIssue("empty_output", "OCR output is empty or whitespace only."))
        return QualityReport(accepted=False, severity="error", issues=issues)

    replacement_count = text.count("\ufffd")
    if replacement_count >= 3 or replacement_count / max(len(text), 1) > 0.02:
        issues.append(QualityIssue("mojibake", "OCR output contains too many Unicode replacement characters."))

    if _CONTROL_CHAR_RE.search(text):
        issues.append(QualityIssue("control_characters", "OCR output contains control characters."))

    if _has_refusal_boilerplate(text):
        issues.append(QualityIssue("refusal_boilerplate", "OCR output contains refusal boilerplate."))

    if _has_malformed_frontmatter(text):
        issues.append(QualityIssue("malformed_frontmatter", "OCR output has malformed frontmatter or schema preamble."))

    if _has_repeated_character_run(text):
        issues.append(QualityIssue("repeated_character", "OCR output contains an abnormal repeated-character run."))

    if _has_repeated_ngram_loop(text):
        issues.append(QualityIssue("repeated_ngram", "OCR output appears to repeat the same phrase."))

    if _has_truncation_indicator(text):
        issues.append(QualityIssue("truncation_indicator", "OCR output appears truncated."))

    if _has_length_anomaly(text):
        issues.append(QualityIssue("length_anomaly", "OCR output length looks anomalous."))

    accepted = not any(issue.severity == "error" for issue in issues)
    return QualityReport(accepted=accepted, severity="ok" if accepted else "error", issues=issues)


def _public_kind(code: str) -> str:
    if code == "refusal_boilerplate":
        return "refusal"
    return code


def _has_repeated_character_run(text: str) -> bool:
    current = ""
    run_length = 0
    for char in text:
        if char == current:
            run_length += 1
        else:
            current = char
            run_length = 1
        if not char.isspace() and run_length >= 24:
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


def _has_refusal_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _REFUSAL_PATTERNS)


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
