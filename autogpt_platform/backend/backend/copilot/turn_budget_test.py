from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
)
from backend.copilot.turn_budget import TurnBudget, resolve_turn_budget

_MOD = "backend.copilot.turn_budget"
NOW = datetime(2026, 10, 14, 15, 0, tzinfo=UTC)
DAILY_RESET = NOW.replace(hour=0) + timedelta(days=1)
WEEKLY_RESET = datetime(2026, 10, 19, tzinfo=UTC)


def _status(
    daily_used: int,
    weekly_used: int,
    *,
    tier: SubscriptionTier = SubscriptionTier.PRO,
    daily_limit: int = 3_000_000,
    weekly_limit: int = 6_000_000,
) -> CoPilotUsageStatus:
    return CoPilotUsageStatus(
        daily=UsageWindow(used=daily_used, limit=daily_limit, resets_at=DAILY_RESET),
        weekly=UsageWindow(
            used=weekly_used, limit=weekly_limit, resets_at=WEEKLY_RESET
        ),
        tier=tier,
    )


@pytest.fixture
def limits():
    """Patch the rate-limit reads; tests set ``remaining`` and ``status``."""
    with (
        patch(
            f"{_MOD}.get_global_rate_limits",
            AsyncMock(return_value=(3_000_000, 6_000_000, SubscriptionTier.PRO)),
        ) as global_limits,
        patch(f"{_MOD}.get_remaining_usd_budget", AsyncMock()) as remaining,
        patch(f"{_MOD}.get_usage_status", AsyncMock()) as status,
        patch(f"{_MOD}.config.claude_agent_max_budget_usd", 10.0),
    ):
        yield global_limits, remaining, status


@pytest.mark.asyncio
async def test_no_user_gets_the_task_cap():
    with patch(f"{_MOD}.config.claude_agent_max_budget_usd", 7.5):
        budget = await resolve_turn_budget(None)
    assert budget == TurnBudget(usd=7.5, limiting="task")


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [float("inf"), -1.0])
async def test_uncapped_or_unreadable_allowance_gets_the_task_cap(limits, remaining):
    # inf: no cap configured. -1: Redis brown-out, which the pre-turn gate has
    # already failed closed on -- don't shrink the turn on a guess.
    _, remaining_mock, _ = limits
    remaining_mock.return_value = remaining
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=10.0, limiting="task")


@pytest.mark.asyncio
async def test_task_cap_binds_when_the_account_has_more_left(limits):
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 80.0
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=10.0, limiting="task")
    status_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_daily_window_binds_with_its_reset(limits):
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 0.5
    status_mock.return_value = _status(daily_used=2_500_000, weekly_used=1_000_000)
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=0.5, limiting="daily", resets_at=DAILY_RESET)


@pytest.mark.asyncio
async def test_no_floor_a_cent_left_is_a_cent(limits):
    # The old resolver topped a near-empty allowance up to $0.50. The turn now
    # gets what is left and lands with a checkpoint instead.
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 0.01
    status_mock.return_value = _status(daily_used=2_990_000, weekly_used=1_000_000)
    budget = await resolve_turn_budget("u-1")
    assert budget.usd == 0.01
    assert budget.limiting == "daily"


@pytest.mark.asyncio
async def test_weekly_window_binds_when_it_is_tighter(limits):
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 0.25
    status_mock.return_value = _status(daily_used=1_000_000, weekly_used=5_750_000)
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=0.25, limiting="weekly", resets_at=WEEKLY_RESET)


@pytest.mark.asyncio
async def test_both_windows_empty_names_the_one_that_resets_last(limits):
    # Telling the user "midnight" while the weekly cap holds them until Monday
    # was the old bug: the later reset is when work can actually resume.
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 0.0
    status_mock.return_value = _status(daily_used=3_000_000, weekly_used=6_000_000)
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=0.0, limiting="weekly", resets_at=WEEKLY_RESET)


@pytest.mark.asyncio
async def test_unreadable_windows_still_name_a_window(limits):
    _, remaining_mock, status_mock = limits
    remaining_mock.return_value = 0.5
    status_mock.side_effect = RateLimitUnavailable("brown-out")
    budget = await resolve_turn_budget("u-1")
    assert budget == TurnBudget(usd=0.5, limiting="daily")


@pytest.fixture
def trial_limits(limits):
    global_limits, remaining_mock, status_mock = limits
    global_limits.return_value = (20_000_000, 20_000_000, SubscriptionTier.TRIAL)
    yield remaining_mock, status_mock


@pytest.mark.asyncio
async def test_trial_allowance_binds_without_a_reset(trial_limits):
    # A lifetime trial folds its allowance into the weekly gauge; it never
    # refills, so there is no reset to offer.
    remaining_mock, status_mock = trial_limits
    remaining_mock.return_value = 0.05
    status_mock.return_value = _status(
        daily_used=1_000_000,
        weekly_used=19_950_000,
        tier=SubscriptionTier.TRIAL,
        daily_limit=20_000_000,
        weekly_limit=20_000_000,
    )
    budget = await resolve_turn_budget("trial-user")
    assert budget == TurnBudget(usd=0.05, limiting="trial", hard_cap=True)


@pytest.mark.asyncio
async def test_trial_with_a_tighter_daily_window_resets_daily(trial_limits):
    remaining_mock, status_mock = trial_limits
    remaining_mock.return_value = 0.1
    status_mock.return_value = _status(
        daily_used=2_400_000,
        weekly_used=5_000_000,
        tier=SubscriptionTier.TRIAL,
        daily_limit=2_500_000,
        weekly_limit=20_000_000,
    )
    budget = await resolve_turn_budget("trial-user")
    assert budget == TurnBudget(
        usd=0.1, limiting="daily", resets_at=DAILY_RESET, hard_cap=True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan")])
async def test_trial_with_nothing_left_starts_no_turn(trial_limits, remaining):
    remaining_mock, _ = trial_limits
    remaining_mock.return_value = remaining
    with pytest.raises(ValueError, match="Trial budget"):
        await resolve_turn_budget("trial-user")


@pytest.mark.asyncio
async def test_a_trial_limited_by_the_task_cap_is_still_a_hard_cap(trial_limits):
    # More left than one task may spend: the task cap binds, but the money is
    # still the trial's, so nothing may be spent past it.
    remaining_mock, _ = trial_limits
    remaining_mock.return_value = 15.0
    budget = await resolve_turn_budget("trial-user")
    assert budget.limiting == "task"
    assert budget.hard_cap is True


@pytest.mark.asyncio
async def test_a_paid_budget_is_not_a_hard_cap(limits):
    _, remaining_mock, _ = limits
    remaining_mock.return_value = 15.0
    assert (await resolve_turn_budget("u-1")).hard_cap is False
