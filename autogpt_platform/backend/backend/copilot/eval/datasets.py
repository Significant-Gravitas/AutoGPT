"""Curated fixtures used by the dream-eval staleness benchmark.

The 12-row split here is the *unit-test sized* version of the 50-fact
fixture called for in ``dream/p0-spec.md`` §3. Twelve covers every
heuristic axis (age, version regex, recency keyword, volatile domain)
plus the headline edge cases (birthday / preference). The full 50-row
set comes in when the eval harness graduates from unit-of-record to a
real cloud-benchmark suite; until then, twelve is what we need to
gate "did P0.3a regress" in CI.

Adding a row:
  * Real-world looking — phrases the assistant might actually persist.
  * One reason per row in the docstring of ``expected_stale=True`` —
    helps future readers understand WHY each fact should be demoted.
"""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class StalenessFixture:
    """One labeled fact for the staleness benchmark.

    ``expected_stale`` is the ground-truth label: True means the
    sanitizer SHOULD demote this fact; False means a healthy P0.3a
    must preserve it.
    """

    fact_text: str
    created_days_ago: int
    expected_stale: bool


# Stable-fact half — must score below threshold. False positives here
# delete real user memory, so this set is the high-cost side of the
# precision/recall tradeoff.
STABLE_FACT_FIXTURES: list[StalenessFixture] = [
    StalenessFixture("user's birthday is March 5", 400, False),
    StalenessFixture("user prefers dark mode in IDEs", 300, False),
    StalenessFixture("user lives in Boston", 250, False),
    StalenessFixture("user works on weekends", 400, False),
    StalenessFixture("user is allergic to peanuts", 600, False),
    StalenessFixture("user's first programming language was Python", 800, False),
]


# Stale-fact half — must score at or above threshold. False negatives
# leave wrong assertions in memory; the sanitizer's job downstream is
# to verify before demoting, so the cost of a flag-but-don't-demote
# is just prompt noise.
STALE_FACT_FIXTURES: list[StalenessFixture] = [
    # Age + version regex hit
    StalenessFixture("GPT-4 is the best LLM available", 400, True),
    StalenessFixture("Claude 2.0 is the most capable Anthropic model", 500, True),
    # Volatile domain — pricing / leadership / market claims
    StalenessFixture("Claude Opus costs $15 per million input tokens", 30, True),
    StalenessFixture("Sundar Pichai is the CEO of Google", 30, True),
    StalenessFixture("OpenAI's valuation is $80 billion", 60, True),
    StalenessFixture("ChatGPT is the leading consumer AI product", 200, True),
]


ALL_STALENESS_FIXTURES: list[StalenessFixture] = (
    STABLE_FACT_FIXTURES + STALE_FACT_FIXTURES
)
