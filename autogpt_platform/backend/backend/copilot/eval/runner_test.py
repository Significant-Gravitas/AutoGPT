"""Eval runner tests — verify schema + suite invariants.

The critical contract is the report schema (downstream dashboards read
this JSON) + the staleness suite's stability number (a regression here
means we're about to delete real user memory).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from backend.copilot.dream.schemas import DreamPassResult, DreamPassUsage, PhaseUsage

from .datasets import ALL_STALENESS_FIXTURES, STABLE_FACT_FIXTURES, STALE_FACT_FIXTURES
from .runner import DreamEvalConfig, run_dream_eval
from .staleness_suite import run_staleness_suite


def _now() -> datetime:
    return datetime(2026, 5, 21, tzinfo=timezone.utc)


def _pass(
    *,
    elapsed: float,
    cost: float | None = 0.005,
    skipped: bool = False,
    error: str | None = None,
    path: str = "sync_baseline",
) -> DreamPassResult:
    return DreamPassResult(
        user_id="u",
        pass_id="p",
        execution_path=path,  # type: ignore[arg-type]
        elapsed_seconds=elapsed,
        skipped=skipped,
        error=error,
        usage=(
            DreamPassUsage(
                phases=[PhaseUsage(phase="consolidate", model="m", cost_usd=cost)],
                total_cost_usd=cost,
            )
            if not skipped
            else None
        ),
    )


# ---------------------------------------------------------------------------
# staleness suite
# ---------------------------------------------------------------------------


def test_staleness_suite_preserves_all_stable_facts_in_fixture():
    """Stability == 1.0 — every stable fact must stay below threshold.
    A drop here means false positives are deleting real memory."""
    result = run_staleness_suite(STABLE_FACT_FIXTURES, now=_now())
    assert result.stability == 1.0
    assert result.failures == []


def test_staleness_suite_flags_at_least_two_thirds_of_stale_facts():
    """Recall ≥ 0.67 — the heuristics don't have to catch every stale
    fact (sanitizer also catches contradictions via consolidated facts)
    but should catch most. False negatives just cost one more cycle."""
    result = run_staleness_suite(STALE_FACT_FIXTURES, now=_now())
    assert result.recall >= 0.67


def test_staleness_suite_reports_specific_failures_when_misclassified():
    """A misclassification surfaces in `failures` with the fact text +
    score — needed to debug regressions from results.json alone."""
    from .datasets import StalenessFixture

    bad_fixture = [
        StalenessFixture("user prefers dark mode", 300, True),  # FN — stable
    ]
    result = run_staleness_suite(bad_fixture, now=_now())
    assert len(result.failures) == 1
    assert "FN" in result.failures[0]
    assert "user prefers dark mode" in result.failures[0]


# ---------------------------------------------------------------------------
# cost + latency suite
# ---------------------------------------------------------------------------


def test_cost_latency_suite_aggregates_pass_counts_and_path_breakdown():
    passes = [
        _pass(elapsed=1.0, path="sync_baseline"),
        _pass(elapsed=2.0, path="sync_baseline"),
        _pass(elapsed=3.0, path="anthropic_batch"),
        _pass(elapsed=0, skipped=True),
    ]
    from .cost_latency_suite import run_cost_latency_suite

    result = run_cost_latency_suite(passes)
    assert result.n == 4
    assert result.skipped == 1
    assert result.by_execution_path == {"sync_baseline": 2, "anthropic_batch": 1}
    assert result.latency["n"] == 3  # skipped excluded
    assert result.latency["p50_seconds"] == 2.0


def test_cost_latency_suite_returns_none_total_when_any_pass_has_unknown_cost():
    """Surfacing the missing-cost gap is more useful than a wrong total.
    Downstream billing dashboards read this — silent zero would
    underreport actual spend."""
    passes = [
        _pass(elapsed=1.0, cost=0.01),
        _pass(elapsed=1.0, cost=None),
    ]
    from .cost_latency_suite import run_cost_latency_suite

    result = run_cost_latency_suite(passes)
    assert result.cost["total_usd"] is None


def test_cost_latency_suite_handles_empty_pass_list_without_raising():
    """Eval scaffolding may run before any real passes exist."""
    from .cost_latency_suite import run_cost_latency_suite

    result = run_cost_latency_suite([])
    assert result.n == 0
    assert result.cost["total_usd"] is None or result.cost["total_usd"] == 0
    assert result.latency["n"] == 0


# ---------------------------------------------------------------------------
# top-level runner
# ---------------------------------------------------------------------------


def test_runner_emits_full_report_with_both_suites_by_default():
    """Default config should produce both staleness + cost_latency suites
    in the report — the schema downstream readers expect."""
    result = run_dream_eval(
        run_id="test-run-1",
        passes=[_pass(elapsed=0.5)],
        now=_now(),
    )
    assert result.run_id == "test-run-1"
    assert "staleness" in result.suites
    assert "cost_latency" in result.suites
    # Round-trip via JSON to confirm the report is serializable
    parsed = json.loads(result.to_json())
    assert parsed["suites"]["staleness"]["n"] == len(ALL_STALENESS_FIXTURES)
    assert parsed["suites"]["cost_latency"]["n"] == 1


def test_runner_skips_disabled_suites():
    """Config flag must actually skip the suite — needed for fast CI runs
    that only check the cheap stats."""
    result = run_dream_eval(
        run_id="r",
        config=DreamEvalConfig(run_staleness=False, run_cost_latency=True),
        now=_now(),
    )
    assert "staleness" not in result.suites
    assert "cost_latency" in result.suites


def test_runner_report_is_serializable_to_pretty_json():
    """results.json is consumed by humans + scripts — both need pretty
    indented JSON that round-trips through json.loads cleanly."""
    result = run_dream_eval(run_id="r", now=_now())
    rendered = result.to_json(indent=2)
    parsed = json.loads(rendered)
    assert parsed["run_id"] == "r"
    assert "python_version" in parsed
    assert "ts" in parsed


# ---------------------------------------------------------------------------
# end-to-end fixture coverage gate
# ---------------------------------------------------------------------------


def test_full_fixture_meets_quality_floor_for_p0_ship():
    """Gate: P0 cannot ship if the staleness heuristics regress below
    the quality floor on the full fixture. Stability is the high-cost
    side — preserve stable facts."""
    result = run_staleness_suite(ALL_STALENESS_FIXTURES, now=_now())
    assert result.stability >= 0.85, (
        f"Stability dropped to {result.stability}; would delete real memory. "
        f"Failures:\n" + "\n".join(result.failures)
    )
    assert result.recall >= 0.67, (
        f"Recall dropped to {result.recall}; heuristics missing too many "
        f"stale facts.\nFailures:\n" + "\n".join(result.failures)
    )


@pytest.mark.parametrize("fixture", ALL_STALENESS_FIXTURES)
def test_each_fixture_row_classifies_to_its_expected_label(fixture):
    """Per-row visibility: if one fixture flips, the failing test name
    tells you exactly which fact regressed."""
    from backend.copilot.dream.staleness import STALENESS_THRESHOLD, score_staleness

    from .staleness_suite import _fixture_to_fact

    fact = _fixture_to_fact(fixture, _now())
    score = score_staleness(fact, now=_now())
    flagged = score >= STALENESS_THRESHOLD
    assert flagged == fixture.expected_stale, (
        f"Misclassified: {fixture.fact_text!r} got score={score:.2f}, "
        f"expected_stale={fixture.expected_stale}"
    )
