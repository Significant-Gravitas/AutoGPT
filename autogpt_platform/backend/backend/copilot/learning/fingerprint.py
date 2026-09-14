"""Normalised behaviour fingerprints for proposed skill changes.

A suppression record identifies the *behaviour* a user removed, not the
exact bytes, so trivially different formatting or reordered steps cannot
evade it. The fingerprint hashes the bag of content tokens across the
procedure's step lines. Negations and numbers are kept: "do not retry"
and "retry", or a 10-row and a 1000-row threshold, are different
behaviours.

This is deliberately conservative and makes no semantic claim: a genuine
paraphrase with new vocabulary will not hash equal. That is why an exact
match is *suppressed*, a high-overlap match is *uncertain* and routed to a
labelled proposal, and — as the first-release boundary — a restore also
turns automatic improvements off for that skill, so nothing reappears
without an explicit decision.
"""

from __future__ import annotations

import hashlib
import re

_STOPWORDS = frozenset(
    "the a an and or to of in on for with then that this is are be as at by "
    "it its into from your you we our use using via each any all".split()
)
_STEP_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.+)$")
_TOKEN_RE = re.compile(r"[a-z0-9_]{2,}")

UNCERTAIN_EQUIVALENCE_THRESHOLD = 0.6


def behavior_tokens(body: str) -> list[str]:
    """Sorted, de-duplicated content tokens of the procedure's step lines.

    Falls back to the whole body when the skill has no list-shaped steps so
    a prose-only recipe still fingerprints.
    """
    steps = [
        match.group(1)
        for match in (_STEP_LINE_RE.match(line) for line in body.splitlines())
        if match
    ]
    text = "\n".join(steps) if steps else body
    tokens = {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS}
    return sorted(tokens)


def behavior_fingerprint(skill_name: str, body: str) -> str:
    digest = hashlib.sha256()
    digest.update(skill_name.strip().lower().encode("utf-8"))
    digest.update(b"\n")
    digest.update(" ".join(behavior_tokens(body)).encode("utf-8"))
    return digest.hexdigest()


def token_overlap(a: list[str], b: list[str]) -> float:
    """Jaccard overlap of two token bags (0 when either is empty)."""
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def evidence_fingerprint(refs: list[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(refs)).encode("utf-8"))
    return digest.hexdigest()
