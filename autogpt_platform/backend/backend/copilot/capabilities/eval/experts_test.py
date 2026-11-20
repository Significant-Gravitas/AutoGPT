"""Expert-hire gate: ``find_capability`` must surface the roster expert a user
asks for and keep experts off searches for something to run.  Floors are the
first measured run (2026-10-02) less a query or two, so a ranking or entry
change that loses ground fails here; ``python -m
backend.copilot.capabilities.eval.experts`` prints the full report and every
failing query."""

from collections import Counter

import pytest

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.sources import expert_entries

from .experts import (
    GROUPS,
    TOP_K,
    Report,
    evaluate,
    format_report,
    load_cases,
    load_roster,
    platform_index,
)

# The slack is the catalogue: which tools load differs by environment, and
# leaps scored 47% from the CLI and 40% under pytest on the same commit.
# Lexical search cannot bridge a symptom to a role ("more followers" never
# names social media), so leaps hold a floor rather than a target.
MIN_HIT5 = {"plain": 0.95, "near": 0.85, "leap": 0.35}
MIN_HIRE_PHRASED_HIT5 = {"plain": 0.95, "near": 0.90, "leap": 0.60}
MIN_EXPERTS_ONLY_HIT5 = {"plain": 0.95, "near": 0.90, "leap": 0.65}
MAX_MISS_EXPERT_FIRST = 0.07
MAX_MISS_EXPERT_IN_TOP5 = 0.15
# Must hit at 5 whatever the floors allow: Toran's own query (SECRT-2814), and
# requests naming a service, which restrict the list to that service's entries.
PINNED = {
    "hire expert social media manager": {"Jules"},
    "someone to run my LinkedIn": {"Jules", "Maya"},
    "find wasted spend in google ads": {"Marco"},
}


@pytest.fixture(scope="module")
def platform() -> CapabilityIndex:
    return platform_index()


@pytest.fixture(scope="module")
def report(platform: CapabilityIndex) -> Report:
    result = evaluate(platform, load_roster(), load_cases())
    print("\n" + format_report(result))
    return result


def test_dataset_covers_every_template_and_group():
    cases = load_cases()
    names = {t.name for t in load_roster()}
    plain = Counter(n for c in cases if c.group == "plain" for n in c.expect)
    assert {n for c in cases for n in c.expect} <= names
    assert min(plain[n] for n in names) >= 2
    assert Counter(c.group for c in cases) == {
        "plain": 66,
        "near": 40,
        "leap": 30,
        "miss": 30,
    }
    assert all(not c.expect for c in cases if c.group == "miss")


def test_platform_registry_alone_finds_no_expert(report: Report):
    assert all(report.today[g].hit5 == 0 for g in GROUPS)


@pytest.mark.parametrize("group", MIN_HIT5)
def test_expert_hit_rate(report: Report, group: str):
    assert report.registry[group].rate("hit5") >= MIN_HIT5[group], format_report(report)
    assert report.hire_phrased[group].rate("hit5") >= MIN_HIRE_PHRASED_HIT5[group]
    assert report.experts_only[group].rate("hit5") >= MIN_EXPERTS_ONLY_HIT5[group]


def test_searches_for_something_to_run_keep_experts_off_the_top(report: Report):
    miss = report.registry["miss"]
    assert miss.rate("fp1") <= MAX_MISS_EXPERT_FIRST, format_report(report)
    assert miss.rate("fp5") <= MAX_MISS_EXPERT_IN_TOP5, format_report(report)


def test_every_template_is_reachable_by_its_role(report: Report):
    assert report.unreachable == []


@pytest.mark.parametrize("query", PINNED)
def test_pinned_requests_find_their_expert(platform: CapabilityIndex, query: str):
    index = platform.with_entries(expert_entries(load_roster(), hired=[]))
    top = [hit.entry for hit in index.search(query).hits[:TOP_K]]
    assert {e.name for e in top if e.kind == "expert"} & PINNED[query], top
