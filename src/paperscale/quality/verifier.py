from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re


@dataclass(frozen=True)
class QualityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    severity: str
    issues: list[QualityIssue] = field(default_factory=list)


_REFUSAL_PATTERNS = (
    r"\bas an ai\b",
    r"\bi can(?:'|)t help\b",
    r"\bi cannot help\b",
    r"\bi(?:'|)m sorry\b",
    r"\bsorry,? but\b",
    r"\brefuse(?:d|s)?\b",
    r"\bcannot provide\b",
)


def assess_markdown_fragment(markdown: str) -> QualityReport:
    text = markdown or ""
    stripped = text.strip()
    issues: list[QualityIssue] = []

    if not stripped:
        issues.append(QualityIssue("empty_output", "fragment is empty"))
        return QualityReport(accepted=False, severity="error", issues=issues)

    if "\ufffd" in text:
        issues.append(QualityIssue("mojibake", "replacement characters detected"))

    if _looks_like_refusal(text):
        issues.append(QualityIssue("refusal_boilerplate", "refusal boilerplate detected"))

    if _has_malformed_frontmatter(text):
        issues.append(QualityIssue("malformed_frontmatter", "frontmatter is not closed"))

    if _looks_repeated(text):
        issues.append(QualityIssue("repeated_ngram", "repeated phrase detected"))

    if _has_truncation_marker(text):
        issues.append(QualityIssue("truncation_indicator", "truncation marker detected"))

    if _has_length_anomaly(text):
        issues.append(QualityIssue("length_anomaly", "fragment length looks anomalous"))

    accepted = not any(issue.code in {"empty_output", "mojibake", "refusal_boilerplate", "malformed_frontmatter", "repeated_ngram"} for issue in issues)
    severity = "ok" if accepted else "error"
    return QualityReport(accepted=accepted, severity=severity, issues=issues)


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _REFUSAL_PATTERNS)


def _has_malformed_frontmatter(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return False
    return stripped.count("\n---") == 0 and not re.search(r"^---\s*$", stripped.splitlines()[-1])


def _looks_repeated(text: str) -> bool:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(tokens) < 12:
        return False
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return False
    counts = Counter(trigrams)
    most_common = counts.most_common(1)[0][1]
    repeated_ratio = most_common / max(1, len(trigrams))
    return most_common >= 6 or repeated_ratio >= 0.25


def _has_truncation_marker(text: str) -> bool:
    lowered = text.lower().rstrip()
    return lowered.endswith("...") or "truncated" in lowered or lowered.endswith("[...]")


def _has_length_anomaly(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) < 5 or len(stripped) > 50_000
