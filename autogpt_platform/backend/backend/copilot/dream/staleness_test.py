"""Staleness heuristic tests — the critical guardrail is "stable facts
must not score above threshold". Stale fact false positives are noise
for the sanitizer; stable fact false positives delete real memory.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from .fetch import FactRow
from .staleness import STALENESS_THRESHOLD, identify_stale_candidates, score_staleness


def _now() -> datetime:
    return datetime(2026, 5, 21, tzinfo=timezone.utc)


def _fact(
    text: str, *, created_days_ago: int = 1, scope: str = "real:global"
) -> FactRow:
    created = _now() - timedelta(days=created_days_ago)
    return FactRow(
        uuid="f1",
        source=None,
        target=None,
        name=None,
        fact=text,
        scope=scope,
        confidence=0.8,
        status="active",
        created_at=created.isoformat().replace("+00:00", "Z"),
    )


def test_empty_fact_text_scores_zero():
    assert score_staleness(_fact(""), now=_now()) == 0.0
    assert score_staleness(_fact("   \n"), now=_now()) == 0.0


def test_user_birthday_old_but_stable_stays_below_threshold():
    """The headline edge case from spec §P0.3a — birthdays must not
    be demoted just because they're old."""
    fact = _fact("user's birthday is March 5", created_days_ago=400)
    score = score_staleness(fact, now=_now())
    assert score < STALENESS_THRESHOLD


def test_old_preference_stays_below_threshold():
    """'user prefers dark mode' — old and a preference — must survive."""
    fact = _fact("user prefers dark mode in IDEs", created_days_ago=300)
    score = score_staleness(fact, now=_now())
    assert score < STALENESS_THRESHOLD


def test_old_version_claim_crosses_threshold():
    """Age + version regex hits both — the canonical "stale and wrong"
    pattern: "GPT-4 is the best model" 18 months later."""
    fact = _fact("GPT-4 is the best LLM available", created_days_ago=400)
    score = score_staleness(fact, now=_now())
    assert score >= STALENESS_THRESHOLD


def test_current_pricing_claim_crosses_threshold():
    """Pricing facts go stale fast even if the assertion is recent."""
    fact = _fact("Claude Opus costs $15 per million input tokens")
    score = score_staleness(fact, now=_now())
    assert score >= STALENESS_THRESHOLD


def test_leadership_claim_crosses_threshold():
    """Company-leader facts are high churn."""
    fact = _fact("Sundar Pichai is the CEO of Google")
    score = score_staleness(fact, now=_now())
    assert score >= STALENESS_THRESHOLD


def test_recency_keyword_alone_does_not_cross_threshold():
    """A single signal (just "latest") shouldn't trip — needs context."""
    fact = _fact("This is the latest config", created_days_ago=1)
    score = score_staleness(fact, now=_now())
    # 0.4 < 0.6
    assert score < STALENESS_THRESHOLD


def test_age_signal_alone_does_not_cross_threshold():
    """An old fact about nothing volatile shouldn't get demoted."""
    fact = _fact("user works on weekends", created_days_ago=400)
    score = score_staleness(fact, now=_now())
    assert score < STALENESS_THRESHOLD


def test_missing_created_at_does_not_trip_age_signal():
    """Legacy edges with no created_at must not be flagged as old —
    a fact whose ONLY potential signal is age scores zero when
    created_at is missing."""
    fact = FactRow(
        uuid="f",
        source=None,
        target=None,
        name=None,
        fact="user works on weekends",
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at=None,
    )
    score = score_staleness(fact, now=_now())
    assert score == 0.0


@pytest.mark.parametrize(
    "text",
    [
        "Claude 4 supports prompt caching",
        "GPT-5 has a 1M context window",
        "Anthropic's latest model is Opus 4.5",
        "claude-sonnet-4 is currently best for code",
    ],
)
def test_model_version_claims_with_recency_keyword_flagged(text):
    """Model + version + recency keyword = high-confidence stale candidate."""
    fact = _fact(text, created_days_ago=200)
    score = score_staleness(fact, now=_now())
    assert score >= STALENESS_THRESHOLD


def test_identify_stale_candidates_returns_descending_score_order():
    """Sanitize prompt needs most-suspect first when budget is tight."""
    facts = [
        _fact("user's birthday is March 5", created_days_ago=400),  # below
        _fact("GPT-4 is the best LLM", created_days_ago=400),  # very high
        _fact("Claude Opus costs $15/M tokens"),  # high
    ]
    candidates = identify_stale_candidates(facts, now=_now())
    assert len(candidates) == 2  # birthday filtered out
    assert candidates[0][1] >= candidates[1][1]


def test_identify_stale_candidates_returns_empty_when_no_candidates():
    """Stable-only fact set produces an empty candidate list."""
    facts = [
        _fact("user prefers dark mode"),
        _fact("user's birthday is March 5", created_days_ago=400),
    ]
    assert identify_stale_candidates(facts, now=_now()) == []
