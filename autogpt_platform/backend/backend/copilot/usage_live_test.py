"""The gate's rules on the billing clock: the daily limit, the week's pool, the
daily reserve inside it, and a paid reset that starts the week over with the
daily limit lifted."""

from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.copilot import usage_live
from backend.copilot.usage_clock import ResolvedClock
from backend.copilot.usage_counters import Recorded
from backend.copilot.usage_live import (
    UsageSnapshot,
    build_snapshot,
    record_live,
    resets_in_window,
    usable_reserve,
)
from backend.copilot.usage_windows import windows_for_period
from backend.data.pro_activation import UsageActivationState

START = 1_793_824_256  # Wed 4 Nov 2026 20:30:56 UTC
END = START + 30 * 86_400
WEEKLY, DAILY, RESERVE = 68_000_000, 22_666_667, 2_000_000
# A 7.5-day window holds 7.5 reserves: the pool is 68 - 15 = 53 dollars.
POOL = 53_000_000


def _state(
    tier: str = "MAX", *, resets: int = 0, reset_week: int = START
) -> UsageActivationState:
    return UsageActivationState(
        user_id="u1",
        tier=SubscriptionTier(tier),
        ready=True,
        reset_week_start=reset_week if resets else None,
        reset_count=resets,
    )


def _snapshot(
    day_used: int = 0,
    week_used: int = 0,
    *,
    resets: int = 0,
    reset_week: int = START,
    tier: str = "MAX",
    daily: int = DAILY,
    weekly: int = WEEKLY,
    reserve: int = RESERVE,
    now: int = START + 3600,
) -> UsageSnapshot:
    clock = ResolvedClock(
        windows=windows_for_period(START, END, START, now), source="stripe"
    )
    return build_snapshot(
        _state(tier, resets=resets, reset_week=reset_week),
        clock,
        daily_limit=daily,
        weekly_limit=weekly,
        reserve=reserve,
        day_used=day_used,
        week_used=week_used,
    )


def test_a_fresh_window_has_the_whole_day_and_the_pool_is_weekly_less_reserves():
    fresh = _snapshot()

    assert fresh.pool == POOL and fresh.reserve == RESERVE
    assert fresh.blocked_by() is None
    assert fresh.left() == DAILY


def test_the_last_short_day_has_a_proportional_reserve():
    half_day = _snapshot(now=START + 7 * 86_400 + 60)

    assert half_day.reserve == RESERVE // 2
    assert half_day.pool == POOL


@pytest.mark.parametrize(
    "day_used,week_used,blocked,left",
    [
        # Under both limits.
        (5_000_000, 3_000_000, None, DAILY - 5_000_000),
        # Today's limit reached with room in the week: stopped until tomorrow.
        (DAILY, 20_666_667, "daily", 0),
        # The pool is gone but today's reserve is untouched: a turn may start.
        (0, POOL, None, RESERVE),
        (1_500_000, POOL, None, 500_000),
        # Pool gone and today's reserve used: the weekly limit has run out.
        (RESERVE, POOL, "weekly", 0),
        (RESERVE + 1, POOL + 1, "weekly", 0),
        # Both limits reached: the weekly one, which refills last.
        (DAILY, POOL, "weekly", 0),
    ],
)
def test_what_stops_a_turn_and_how_much_is_left(
    day_used: int, week_used: int, blocked: str | None, left: int
):
    snapshot = _snapshot(day_used, week_used)

    assert snapshot.blocked_by() == blocked
    assert snapshot.left() == left


def test_a_background_pass_runs_on_the_pool_alone():
    reserve_only = _snapshot(0, POOL)
    day_full = _snapshot(DAILY, 10_000_000)

    # Not on the reserve, which is the customer's.
    assert reserve_only.blocked_by(skip_daily=True) == "weekly"
    # The daily limit doesn't apply to it, as before.
    assert day_full.blocked_by(skip_daily=True) is None


def test_a_paid_reset_is_one_whole_weekly_limit_with_no_daily_limit():
    fresh = _snapshot(resets=1)

    assert fresh.reset_active
    assert fresh.daily_limit == -1 and fresh.reserve == 0
    # No reserves are kept back: the pool is the whole weekly limit.
    assert fresh.pool == fresh.weekly_limit == WEEKLY
    assert fresh.blocked_by() is None and fresh.left() == WEEKLY


@pytest.mark.parametrize(
    "day_used,week_used,blocked,left",
    [
        # More than a normal day in one day: nothing stops it.
        (DAILY + 9_000_000, DAILY + 9_000_000, None, WEEKLY - DAILY - 9_000_000),
        # A microdollar short of the weekly limit: still going.
        (WEEKLY - 1, WEEKLY - 1, None, 1),
        # Exactly one weekly limit spent since the reset: stopped, and until
        # the weekly reset, not tomorrow (no reserve arrives before it).
        (WEEKLY, WEEKLY, "weekly", 0),
        (0, WEEKLY, "weekly", 0),
    ],
)
def test_after_a_paid_reset_exactly_one_weekly_limit_can_be_spent(
    day_used: int, week_used: int, blocked: str | None, left: int
):
    snapshot = _snapshot(day_used, week_used, resets=1)

    assert snapshot.blocked_by() == blocked
    assert snapshot.left() == left


def test_a_paid_reset_keeps_the_windows_own_reset_time():
    plain, reset = _snapshot(), _snapshot(resets=2)

    assert reset.windows == plain.windows
    assert reset.resets == 2


def test_a_paid_reset_ends_with_its_weekly_window():
    """The next window is a normal one: daily limit and reserve are back."""
    next_window = _snapshot(resets=1, now=START + 8 * 86_400)

    assert not next_window.reset_active
    assert next_window.daily_limit == DAILY and next_window.reserve == RESERVE


def test_a_reset_of_another_window_does_not_apply():
    stale = _snapshot(resets=1, reset_week=START - 7 * 86_400)

    assert not stale.reset_active and stale.daily_limit == DAILY


@pytest.mark.parametrize("tier", ["TRIAL", "NO_TIER"])
def test_a_paid_reset_never_counts_for_a_trial_or_without_a_plan(tier: str):
    snapshot = _snapshot(RESERVE, POOL, resets=1, tier=tier)

    assert not snapshot.reset_active
    assert snapshot.blocked_by() == "weekly"


def test_a_paid_reset_is_not_a_plan():
    no_allowance = _snapshot(0, 0, resets=1, daily=0, weekly=0, reserve=0)

    assert not no_allowance.reset_active
    assert no_allowance.blocked_by() == "weekly"


def test_a_paid_reset_does_not_lift_a_daily_limit_that_is_switched_off():
    off = _snapshot(0, 0, resets=1, daily=0)

    assert off.daily_limit == 0 and off.blocked_by() == "daily"


def test_a_paid_reset_gives_an_uncapped_week_nothing():
    uncapped = _snapshot(resets=1, weekly=-1)

    assert not uncapped.reset_active and uncapped.daily_limit == DAILY


def test_resets_in_window_counts_only_this_windows_resets():
    windows = windows_for_period(START, END, START, START + 3600)

    assert resets_in_window(_state(resets=2), windows) == 2
    assert resets_in_window(_state(resets=2, reset_week=START + 1), windows) == 0
    assert resets_in_window(_state(), windows) == 0


def test_without_a_reserve_the_rules_are_the_weekly_and_daily_limits_alone():
    plain = _snapshot(
        100, 299_999_999, daily=150_000_000, weekly=300_000_000, reserve=0
    )
    full = _snapshot(100, 300_000_000, daily=150_000_000, weekly=300_000_000, reserve=0)

    assert plain.pool == 300_000_000 and plain.reserve == 0
    assert plain.blocked_by() is None and plain.left() == 1
    assert full.blocked_by() == "weekly"


@pytest.mark.parametrize(
    "daily,weekly,blocked,left",
    [
        (-1, -1, None, None),  # no caps at all
        (-1, 5_000_000, "weekly", 0),  # only the week is capped
        (2_000_000, -1, "daily", 0),  # only the day is capped
    ],
)
def test_a_negative_limit_is_no_cap(
    daily: int, weekly: int, blocked: str | None, left: int | None
):
    snapshot = _snapshot(9_000_000, 9_000_000, daily=daily, weekly=weekly)

    assert snapshot.reserve == 0  # nothing to keep a reserve out of
    assert snapshot.blocked_by() == blocked
    assert snapshot.left() == left


@pytest.mark.parametrize(
    "reserve,daily,weekly,usable",
    [
        (2_000_000, 22_666_667, 68_000_000, 2_000_000),
        (2_000_000, -1, 68_000_000, 0),
        (2_000_000, 22_666_667, -1, 0),
        (2_000_000, 0, 0, 0),
        (9_000_000, 3_000_000, 68_000_000, 3_000_000),
        (-5, 3_000_000, 68_000_000, 0),
    ],
)
def test_a_reserve_needs_both_windows_capped(
    reserve: int, daily: int, weekly: int, usable: int
):
    assert usable_reserve(reserve, daily, weekly) == usable


def _recording():
    recorded = Recorded(day_used=1, week_used=1)
    return patch.object(usage_live, "record_spend", AsyncMock(return_value=recorded))


async def _record(state: UsageActivationState, *, skip_daily: bool = False) -> None:
    windows = windows_for_period(START, END, START, START + 3600)
    await record_live(
        state,
        None,
        windows,
        4_000_000,
        daily_limit=DAILY,
        weekly_limit=WEEKLY,
        reserve=RESERVE,
        skip_daily=skip_daily,
        now=START + 3600,
    )


@pytest.mark.asyncio
async def test_a_cost_is_counted_with_todays_reserve():
    with _recording() as spend:
        await _record(_state())

    assert spend.await_args.kwargs["reserve"] == RESERVE
    assert spend.await_args.kwargs["resets"] == 0
    assert spend.await_args.kwargs["counts_daily"] is True


@pytest.mark.asyncio
async def test_after_a_paid_reset_a_cost_is_counted_on_its_fresh_counters():
    with _recording() as spend:
        await _record(_state(resets=2))

    # The counters since the second reset, and every dollar against the pool.
    assert spend.await_args.kwargs["resets"] == 2
    assert spend.await_args.kwargs["reserve"] == 0


@pytest.mark.asyncio
async def test_a_background_pass_does_not_count_against_the_day():
    with _recording() as spend:
        await _record(_state(), skip_daily=True)

    assert spend.await_args.kwargs["counts_daily"] is False


@pytest.mark.asyncio
async def test_a_trials_cost_ignores_a_reset():
    with _recording() as spend:
        await _record(_state("TRIAL", resets=1))

    assert spend.await_args.kwargs["resets"] == 0
