"""Tier-0 gate: the registry index must beat the recorded find_block results on
the real query set, and the pinned named cases must resolve."""

import pytest

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.registry import build_entries
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY

from .benchmark import NAMED_CASES, Report, evaluate, format_report, load_cases

# Floors from the plan (section 3): keep >= 90% of what find_block found,
# answer > 90% of queries with something.  Labels are "what the turn did
# next", so absolute recall is capped by label noise; the gate is parity
# with today's recorded results, within a small tolerance for block
# description drift.
MIN_RETAINED = 0.90
MAX_NO_RESULT_RATE = 0.10
PARITY_TOLERANCE = 0.03


@pytest.fixture(scope="module")
def report() -> Report:
    # Score against the same catalogue the recorded results came from: a
    # machine without provider secrets disables the Google, Twitter, Notion
    # and Reddit blocks, and thirteen labelled answers vanish with them.
    index = CapabilityIndex(
        build_entries(TOOL_REGISTRY, TOOL_GROUPS, include_disabled_blocks=True)
    )
    result = evaluate(index, load_cases())
    print("\n" + format_report(result))
    return result


def test_dataset_is_labelled():
    cases = load_cases()
    assert len(cases) > 500
    assert sum(c.label_kind == "block" for c in cases) > 100


def test_registry_matches_today_on_blocks(report: Report):
    today, registry = report.today["block"], report.registry["block"]
    assert registry.rate("hit5") >= today.rate("hit5") - PARITY_TOLERANCE
    assert registry.rate("hit1") >= today.rate("hit1") - PARITY_TOLERANCE
    assert report.retained_rate() >= MIN_RETAINED
    assert registry.rate("miss") < today.rate("miss")


def test_registry_reaches_tools_and_mcp_that_find_block_could_not(report: Report):
    # Tool labels are mostly what the model fell back to after a miss
    # (find_library_agent, web_search), so only reach is asserted here.
    assert report.registry["tool"].rate("hit5") > 0
    assert report.registry["tool"].rate("miss") < report.today["tool"].rate("miss")
    assert report.registry["mcp"].rate("hit5") >= 0.8


def test_registry_returns_something_far_more_often(report: Report):
    today, registry = report.today["all"], report.registry["all"]
    assert registry.rate("miss") < today.rate("miss")
    assert registry.rate("miss") < MAX_NO_RESULT_RATE


def test_named_cases(report: Report):
    assert report.named_failures() == {}, format_report(report)
    assert set(report.named) == set(NAMED_CASES)
