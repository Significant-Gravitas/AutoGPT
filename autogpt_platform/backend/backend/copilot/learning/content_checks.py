"""Deterministic content checks for a skill bundle before it becomes ready.

Rejects known credential patterns and caller-supplied seeded secret values
across every file in the bundle (SKILL.md plus references and snippets).
A failure names the pattern class and an *ordinal* location (section and
step number, line number) — never source text, headings, or file names —
so the diagnostic cannot itself leak a secret.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel

# Typed-input placeholders a reusable recipe may legitimately contain:
# ``{{API_KEY}}``, ``${SLACK_TOKEN}``, ``<API_KEY>``. Only an upper-case
# identifier qualifies; arbitrary bracketed material is scanned as-is.
_PLACEHOLDER_RE = re.compile(
    r"(\{\{[A-Z][A-Z0-9_]{1,63}\}\}|\$\{[A-Z][A-Z0-9_]{1,63}\}|<[A-Z][A-Z0-9_]{1,63}>)"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{24,}")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|auth)\b\s*[:=]\s*"
            r"['\"]?(?!\{\{)[A-Za-z0-9+/_=-]{16,}"
        ),
    ),
    (
        "connection_string_password",
        re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@]{6,}@"),
    ),
]

PATTERN_CLASSES: tuple[str, ...] = tuple(name for name, _ in _PATTERNS)
SEEDED_PATTERN_CLASS = "seeded_secret"

_STEP_RE = re.compile(r"^\s*(\d+)[.)]\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
_SAFE_FILE_RE = re.compile(r"^[A-Za-z0-9._/-]{1,80}$")


class ContentCheckFailure(BaseModel):
    """Where a check failed, in ordinals only."""

    pattern_class: str
    step: str
    file: str = "SKILL.md"

    def describe(self) -> str:
        return (
            f"Blocked by content check: {self.pattern_class} in {self.file} "
            f"at {self.step}"
        )


def check_skill_bundle(
    files: dict[str, str],
    *,
    seeded_values: Iterable[str] = (),
    allowed_pattern_classes: Iterable[str] = (),
) -> ContentCheckFailure | None:
    """Return the first failure across the bundle, or ``None`` when clean.

    ``seeded_values`` are known secret values (for example from tool
    output) that must not appear anywhere; ``allowed_pattern_classes`` is
    an audited, scoped allowance for a legitimate non-secret false positive
    (the seeded check can never be allowed).
    """
    allowed = set(allowed_pattern_classes) - {SEEDED_PATTERN_CLASS}
    seeds = [value for value in seeded_values if value and len(value) >= 8]
    for index, (file_name, text) in enumerate(files.items(), start=1):
        label = _safe_file_label(file_name, index)
        failure = _check_text(text, label, seeds, allowed)
        if failure is not None:
            return failure
    return None


def check_skill_content(
    content: str,
    *,
    seeded_values: Iterable[str] = (),
    allowed_pattern_classes: Iterable[str] = (),
) -> ContentCheckFailure | None:
    return check_skill_bundle(
        {"SKILL.md": content},
        seeded_values=seeded_values,
        allowed_pattern_classes=allowed_pattern_classes,
    )


def safe_diagnostic(text: str, *, limit: int = 500) -> str:
    """Text that may be stored or shown: withheld when it matches a secret
    pattern. Model-written reasons and provider error strings can echo
    input, so they pass through here before reaching the ledger or UI."""
    failure = check_skill_content(text)
    if failure is not None:
        return f"(diagnostic withheld: matched {failure.pattern_class})"
    return text[:limit]


def check_metadata(fields: dict[str, str]) -> ContentCheckFailure | None:
    """Content checks over model-derived metadata (name, description,
    triggers, summary) before anything is persisted under those values."""
    for index, (label, value) in enumerate(fields.items(), start=1):
        failure = _check_text(value, f"metadata field {index} ({label})", [], set())
        if failure is not None:
            return failure
    return None


_KNOWN_BUNDLE_FOLDERS = ("references", "scripts", "assets")


def _safe_file_label(file_name: str, index: int) -> str:
    """A file label that cannot carry secret material: the ordinal plus, at
    most, the conventional bundle folder it sits in — never the name."""
    if file_name.rsplit("/", 1)[-1] == "SKILL.md":
        return "SKILL.md"
    folder = file_name.split("/", 1)[0] if "/" in file_name else ""
    if folder in _KNOWN_BUNDLE_FOLDERS and _SAFE_FILE_RE.match(file_name):
        return f"bundle file {index} ({folder}/)"
    return f"bundle file {index}"


def _check_text(
    text: str, file_label: str, seeds: list[str], allowed: set[str]
) -> ContentCheckFailure | None:
    section = 0
    step: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _HEADING_RE.match(line):
            section += 1
            step = None
        step_match = _STEP_RE.match(line)
        if step_match:
            step = int(step_match.group(1))
        location = _location(section, step, line_number)
        for seed in seeds:
            if seed in line:
                return ContentCheckFailure(
                    pattern_class=SEEDED_PATTERN_CLASS, step=location, file=file_label
                )
        scrubbed = _PLACEHOLDER_RE.sub("", line)
        for pattern_class, pattern in _PATTERNS:
            if pattern_class in allowed:
                continue
            if pattern.search(scrubbed):
                return ContentCheckFailure(
                    pattern_class=pattern_class, step=location, file=file_label
                )
    return None


def _location(section: int, step: int | None, line_number: int) -> str:
    parts = [f"section {section}" if section else "preamble"]
    if step is not None:
        parts.append(f"step {step}")
    parts.append(f"line {line_number}")
    return " › ".join(parts)
