"""Per-plan limits from the ``usage-plan-limits`` flag: what is accepted, and
what a bad value falls back to."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import usage_limits
from backend.copilot.usage_limits import (
    DEFAULT_PLAN_LIMITS,
    PlanLimits,
    get_plan_limits,
    parse_plan_limits,
    plan_limits_for,
)

PRO = {"daily": 4_000_000, "weekly": 10_000_000, "reserve": 300_000}


@pytest.fixture(autouse=True)
def _fresh_flag_cache():
    usage_limits._fetch_plan_limits_flag.cache_clear()
    yield
    usage_limits._fetch_plan_limits_flag.cache_clear()


def _flag(value: Any):
    return patch.object(
        usage_limits, "get_feature_flag_value", AsyncMock(return_value=value)
    )


def test_the_defaults_are_the_approved_launch_numbers():
    pro, top = DEFAULT_PLAN_LIMITS["PRO"], DEFAULT_PLAN_LIMITS["MAX"]

    assert (pro.daily, pro.weekly, pro.reserve) == (3_200_000, 8_000_000, 250_000)
    assert (top.weekly, top.reserve) == (68_000_000, 2_000_000)
    # Max's daily limit is a third of its week, and its week 8.5 times Pro's.
    assert top.daily == 22_666_667 and top.daily * 3 >= top.weekly
    assert top.weekly * 2 == pro.weekly * 17
    # A reset costs a quarter of the monthly price: $12.50 and $80.
    assert (pro.reset_credits, top.reset_credits) == (1_250, 8_000)
    assert set(DEFAULT_PLAN_LIMITS) == {"PRO", "MAX"}


def test_a_valid_entry_is_read():
    parsed = parse_plan_limits({"PRO": {**PRO, "reset_credits": 1_500}})

    assert parsed == {"PRO": PlanLimits(**PRO, reset_credits=1_500)}


@pytest.mark.parametrize(
    "entry",
    [
        {**PRO, "daily": "4000000"},  # a string
        {**PRO, "daily": 4_000_000.0},  # a float
        {**PRO, "daily": True},  # a bool
        {**PRO, "daily": 4},  # dollars, not microdollars
        {**PRO, "weekly": 20_000_000_000},  # absurdly large
        {**PRO, "daily": 11_000_000},  # above weekly
        {**PRO, "reserve": 5_000_000},  # above daily
        {**PRO, "reserve": 1_300_000},  # 8 days of it exceed weekly
        {**PRO, "reserve": -1},
        {**PRO, "reset_credits": 50},  # under a dollar
        {**PRO, "reset_credits": -5},
        {**PRO, "weeekly": 1},  # a typo'd key
        {"daily": 4_000_000},  # weekly missing
        "3200000",
        None,
    ],
)
def test_an_invalid_entry_is_dropped_whole_and_the_others_kept(entry: Any):
    parsed = parse_plan_limits(
        {"PRO": entry, "MAX": {"daily": 50_000_000, "weekly": 90_000_000}}
    )

    assert "PRO" not in parsed
    assert parsed["MAX"].weekly == 90_000_000


@pytest.mark.parametrize("plan", ["NO_TIER", "TRIAL", "GOLD", "pro"])
def test_no_plan_and_trial_can_never_be_set_from_the_flag(plan: str):
    assert parse_plan_limits({plan: PRO}) == {}


@pytest.mark.parametrize("payload", [[], "x", 5, True])
def test_a_payload_that_is_not_an_object_changes_nothing(payload: Any):
    assert parse_plan_limits(payload) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, False, {}, {"PRO": {"daily": 1}}, "junk"])
async def test_without_a_usable_flag_the_defaults_apply(answer: Any):
    with _flag(answer):
        assert await get_plan_limits() == DEFAULT_PLAN_LIMITS


@pytest.mark.asyncio
async def test_a_flag_entry_replaces_that_plans_default_only():
    with _flag({"PRO": PRO, "BUSINESS": {"daily": 9_000_000, "weekly": 9_000_000}}):
        limits = await get_plan_limits()

    assert limits["PRO"] == PlanLimits(**PRO)
    assert limits["MAX"] == DEFAULT_PLAN_LIMITS["MAX"]
    assert limits["BUSINESS"].reserve == 0


@pytest.mark.asyncio
async def test_a_failing_flag_read_falls_back_to_the_defaults():
    with patch.object(
        usage_limits,
        "get_feature_flag_value",
        AsyncMock(side_effect=RuntimeError("posthog down")),
    ):
        assert await get_plan_limits() == DEFAULT_PLAN_LIMITS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier", ["BASIC", "BUSINESS", "ENTERPRISE", "NO_TIER", "TRIAL"]
)
async def test_plans_without_their_own_limits_have_none(tier: str):
    with _flag(None):
        assert await plan_limits_for(tier) is None
