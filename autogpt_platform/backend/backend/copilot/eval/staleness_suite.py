"""Staleness benchmark — measures precision/recall of P0.3a heuristics.

Runs ``score_staleness`` over a labeled fixture, reports:
  * precision   = correctly-flagged stale / total flagged
  * recall      = correctly-flagged stale / total actually stale
  * stability   = preserved stable / total actually stable
    (= 1 - false positive rate; the high-cost side — false positives
    here are the "delete real user memory" failure mode)

Stability is the most important number; the harness fails the suite
when stability dips below 0.85 (8/10 stable facts preserved). False
negatives (missed stale facts) cost an extra dream cycle to catch;
false positives delete real memory.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

from pydantic.dataclasses import dataclass

from backend.copilot.dream.fetch import FactRow
from backend.copilot.dream.staleness import STALENESS_THRESHOLD, score_staleness

from .datasets import StalenessFixture


@dataclass(frozen=True)
class StalenessSuiteResult:
    n: int
    precision: float
    recall: float
    stability: float
    threshold: float
    failures: list[str]
    """Human-readable description of each misclassification — surfaces
    in the JSON ``results.json`` ``suites.staleness.failures`` field so
    a regression in the heuristics is debuggable from the report alone
    (no replay needed)."""


def _fixture_to_fact(fixture: StalenessFixture, now: datetime) -> FactRow:
    created = now - timedelta(days=fixture.created_days_ago)
    return FactRow(
        uuid=f"fixture-{hash(fixture.fact_text) & 0xFFFFFFFF:08x}",
        source=None,
        target=None,
        name=None,
        fact=fixture.fact_text,
        scope="real:global",
        confidence=0.8,
        status="active",
        created_at=created.isoformat().replace("+00:00", "Z"),
    )


def run_staleness_suite(
    fixtures: Sequence[StalenessFixture], *, now: datetime | None = None
) -> StalenessSuiteResult:
    now = now or datetime.now(timezone.utc)

    true_positive = 0  # flagged AND actually stale
    false_positive = 0  # flagged BUT actually stable
    true_negative = 0  # not flagged AND actually stable
    false_negative = 0  # not flagged BUT actually stale
    failures: list[str] = []

    for fixture in fixtures:
        fact = _fixture_to_fact(fixture, now)
        score = score_staleness(fact, now=now)
        flagged = score >= STALENESS_THRESHOLD

        if flagged and fixture.expected_stale:
            true_positive += 1
        elif flagged and not fixture.expected_stale:
            false_positive += 1
            failures.append(
                f"FP score={score:.2f}: {fixture.fact_text!r} "
                f"(stable fact incorrectly flagged)"
            )
        elif not flagged and not fixture.expected_stale:
            true_negative += 1
        else:
            false_negative += 1
            failures.append(
                f"FN score={score:.2f}: {fixture.fact_text!r} "
                f"(stale fact not flagged)"
            )

    total_flagged = true_positive + false_positive
    total_stale = true_positive + false_negative
    total_stable = true_negative + false_positive

    precision = true_positive / total_flagged if total_flagged else 1.0
    recall = true_positive / total_stale if total_stale else 1.0
    stability = true_negative / total_stable if total_stable else 1.0

    return StalenessSuiteResult(
        n=len(fixtures),
        precision=round(precision, 3),
        recall=round(recall, 3),
        stability=round(stability, 3),
        threshold=STALENESS_THRESHOLD,
        failures=failures,
    )
