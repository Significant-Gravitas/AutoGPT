"""Stale-fact deprecation heuristics for the sanitize step.

Surfaces candidate "this fact may no longer be true" rows to the
sanitize prompt so the LLM can issue a ``DreamDemotion`` when general
knowledge or the consolidated phase 1 facts contradict the candidate.
The heuristics here are *non-blocking*: they raise a fact's score, the
sanitizer makes the final call. Per ``dream/p0-spec.md`` §P0.3a.

Why heuristics + LLM and not heuristics alone:
  * Stable facts that look stale by heuristic (e.g. "user's birthday
    is March 5" — contains a "specific date" and is old) must not be
    demoted. The sanitizer reads phase 1 facts alongside the candidate
    and rejects the demotion when the world hasn't changed.
  * The heuristic side is cheap (regex + age math) so we can afford to
    score every fact in the dream window, not just a sample.

Edge case test fixtures (the 50-fact "stale and wrong" vs "old but
stable" set) live in ``backend/copilot/eval/`` as part of P0.6 — not
in this module.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fetch import FactRow

logger = logging.getLogger(__name__)


# A fact at or above this score is surfaced to the sanitize prompt as a
# demotion candidate. The threshold is conservative — false negatives
# (a stale fact slipping through) cost one more dream cycle to catch,
# false positives spam the sanitizer with noise to ignore.
STALENESS_THRESHOLD = 0.6


# Each heuristic adds this much to the score when it fires. Tuned so:
#   * Birthday / preference (age only) → 0.4, stays below threshold.
#   * Old model-version claim (age + version) → 0.8, above.
#   * Current pricing or leadership claim (volatile domain alone) → 0.6,
#     above — these go stale fast even when freshly recorded.
#   * "Latest config" alone (recency keyword only) → 0.3, below — too
#     little signal to demote without context.
# Threshold is 0.6 so a SINGLE volatile-domain hit is enough; everything
# else needs at least two corroborating signals.
_AGE_WEIGHT = 0.4
_VERSION_WEIGHT = 0.4
_RECENCY_KEYWORD_WEIGHT = 0.3
_VOLATILE_DOMAIN_WEIGHT = 0.6


# Matches version numbers, year stamps, and named model versions that
# usually rev within a year. ``\b`` guards keep e.g. "v2" matching a
# version but skipping the "v2" inside the word "advert2".
_VERSION_REGEX = re.compile(
    r"\bv?\d+\.\d+\b"  # 1.2, v3.4
    r"|\b(?:19|20)\d{2}\b"  # 1999, 2024
    r"|\b(?:GPT|gpt|Claude|claude|Gemini|gemini)-?\d+(?:\.\d+)?(?:o)?\b"  # GPT-4, Claude 3.5
    r"|\bgpt-image-\d+\b"
)


# Words signalling that the fact is asserting current truth.  These age
# fast — "X is the latest" was true last month, not now.
_RECENCY_KEYWORDS = re.compile(
    r"\b(?:best|current|latest|as of|now|today|recent|trending|leading|top-rated)\b",
    re.IGNORECASE,
)


# Domains where facts go stale fastest: pricing, leadership, capability
# claims about specific tools.  Stable-fact counterparts (birthdays,
# preferences) don't trigger these.
_VOLATILE_DOMAIN = re.compile(
    r"\b(?:price|costs?|priced?|fee|fees|charge|"
    r"CEO|founder|head of|chairman|leader|president|director|"
    r"market share|valuation|revenue|funding|"
    r"supports?|integrates?|works with|powered by"
    r")\b",
    re.IGNORECASE,
)


def score_staleness(fact: "FactRow", *, now: datetime | None = None) -> float:
    """Return a 0.0–1.0 score; ≥ ``STALENESS_THRESHOLD`` is a candidate.

    Each heuristic adds an independent contribution; the score is
    clamped at 1.0. Returns 0.0 for facts with empty text — the
    sanitizer can't do anything with them so flagging is noise.

    A fact with ONLY the age signal stays at 0.5 and does not cross
    the threshold; a true-but-old preference like "user prefers dark
    mode" doesn't get demoted just because it's old.
    """
    text = fact.fact or ""
    if not text.strip():
        return 0.0

    now = now or datetime.now(timezone.utc)

    score = 0.0
    if _is_old(fact, now):
        score += _AGE_WEIGHT
    if _VERSION_REGEX.search(text):
        score += _VERSION_WEIGHT
    if _RECENCY_KEYWORDS.search(text):
        score += _RECENCY_KEYWORD_WEIGHT
    if _VOLATILE_DOMAIN.search(text):
        score += _VOLATILE_DOMAIN_WEIGHT

    return min(score, 1.0)


def identify_stale_candidates(
    facts: list["FactRow"], *, now: datetime | None = None
) -> list[tuple["FactRow", float]]:
    """Filter to facts at or above the staleness threshold.

    Returns ``(fact, score)`` tuples ordered by descending score so the
    sanitize prompt can render the most-suspect candidates first when
    the candidate count exceeds the prompt budget.
    """
    candidates: list[tuple[FactRow, float]] = []
    for fact in facts:
        score = score_staleness(fact, now=now)
        if score >= STALENESS_THRESHOLD:
            candidates.append((fact, score))
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates


def _is_old(fact: "FactRow", now: datetime) -> bool:
    """True when ``fact.created_at`` is more than 90 days in the past.

    Missing or unparseable timestamps return False — we'd rather under-
    flag a fact than over-flag every legacy edge that lost its
    ``created_at`` in a migration.
    """
    if not fact.created_at:
        return False
    parsed = _parse_iso(fact.created_at)
    if parsed is None:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (now - parsed).days > 90


def _parse_iso(raw: str) -> datetime | None:
    """Best-effort ISO 8601 parse; returns None on any failure.

    Graphiti stores ``created_at`` as a Cypher ``datetime`` which
    round-trips through ``toString(...)`` as e.g. ``"2026-05-12T09:33Z"``
    or ``"2026-05-12T09:33:14.123Z"``. Python's ``fromisoformat``
    accepts both since 3.11.
    """
    try:
        # Cypher's `Z` suffix isn't accepted by fromisoformat on older
        # Pythons; normalize before parsing.
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
