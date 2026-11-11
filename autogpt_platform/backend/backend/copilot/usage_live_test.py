"""The gate's rules on the billing clock: the daily limit, the week's pool, the
daily reserve inside it, and a paid extra week once the week has run out."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.copilot import usage_live
from backend.copilot.usage_clock import ResolvedClock
from backend.copilot.usage_counters import Recorded
from backend.copilot.usage_live import (
    UsageSnapshot,
    build_snapshot,
    record_live,
    usable_reserve,
)
from backend.copilot.usage_windows import windows_for_period
from backend.data.pro_activation import UsageActivationState

START = 1_793_824_256  # Wed 4 Nov 2026 20:30:56 UTC
END = START + 30 * 86_400
WEEKLY, DAILY, RESERVE = 68_000_000, 22_666_667, 2_000_000
# A 7.5-day window holds 7.5 reserves: the pool is 68 - 15 = 53 dollars.
POOL = 53_000_000


def _state(tier: str = "MAX", *, extra: int = 0) -> UsageActivationState:
    return UsageActivationState(
        user_id="u1",
        tier=SubscriptionTier(tier),
        ready=True,
        extra_remaining=extra,
        extra_total=max(extra, 68_000_000) if extra else 0,
        extra_expires_at=START + 20 * 86_400 if extra else None,
        extra_count=1 if extra else 0,
    )


def _snapshot(
    day_used: int = 0,
    week_used: int = 0,
    *,
    extra: int = 0,
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
        _state(tier, extra=extra),
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
    "day_used,week_used,extra,blocked,left",
    [
        # Under both limits.
        (5_000_000, 3_000_000, 0, None, DAILY - 5_000_000),
        # Today's limit reached with room in the week: stopped until tomorrow.
        (DAILY, 20_666_667, 0, "daily", 0),
        # The pool is gone but today's reserve is untouched: a turn may start.
        (0, POOL, 0, None, RESERVE),
        (1_500_000, POOL, 0, None, 500_000),
        # Pool gone and today's reserve used: the weekly limit has run out.
        (RESERVE, POOL, 0, "weekly", 0),
        (RESERVE + 1, POOL + 1, 0, "weekly", 0),
        # Both limits reached: the weekly one, which refills last.
        (DAILY, POOL, 0, "weekly", 0),
        # A paid extra week takes over once the weekly limit has run out...
        (RESERVE, POOL, 30_000_000, None, 30_000_000),
        # ...and the daily limit doesn't apply to it...
        (DAILY + 9_000_000, POOL, 30_000_000, None, 30_000_000),
        # ...but it is not drawn on while today's reserve or the pool has room,
        (1_000_000, POOL, 30_000_000, None, 1_000_000 + 30_000_000),
        # nor can it lift the daily limit while the week still has room.
        (DAILY, 20_666_667, 30_000_000, "daily", 0),
    ],
)
def test_what_stops_a_turn_and_how_much_is_left(
    day_used: int, week_used: int, extra: int, blocked: str | None, left: int
):
    snapshot = _snapshot(day_used, week_used, extra=extra)

    assert snapshot.blocked_by() == blocked
    assert snapshot.left() == left


def test_a_background_pass_runs_on_the_pool_alone():
    reserve_only = _snapshot(0, POOL, extra=30_000_000)
    day_full = _snapshot(DAILY, 10_000_000)

    # Not on the reserve, which is the customer's, nor on a paid extra week.
    assert reserve_only.blocked_by(skip_daily=True) == "weekly"
    # The daily limit doesn't apply to it, as before.
    assert day_full.blocked_by(skip_daily=True) is None


@pytest.mark.parametrize("tier", ["TRIAL", "NO_TIER"])
def test_an_extra_week_never_counts_for_a_trial_or_without_a_plan(tier: str):
    snapshot = _snapshot(RESERVE, POOL, extra=30_000_000, tier=tier)

    assert snapshot.extra_remaining == 0
    assert snapshot.blocked_by() == "weekly"


def test_an_extra_week_is_not_a_plan():
    no_allowance = _snapshot(0, 0, extra=30_000_000, daily=0, weekly=0, reserve=0)

    assert no_allowance.extra_remaining == 0
    assert no_allowance.blocked_by() == "weekly"


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


def _recording(excess: int):
    recorded = Recorded(day_used=1, week_used=1, weekly_excess=excess)
    return patch.object(usage_live, "record_spend", AsyncMock(return_value=recorded))


def _database(consume: AsyncMock):
    return patch.object(
        usage_live,
        "usage_clock_db",
        return_value=MagicMock(consume_extra_allowance=consume),
    )


async def _record(
    state: UsageActivationState, *, skip_daily: bool = False, sells: bool = True
) -> None:
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
        sells_extra_weeks=sells,
    )


@pytest.mark.asyncio
async def test_a_plan_that_sells_none_asks_only_when_it_holds_one():
    """A Business account that kept an extra week from its Max days still
    draws on it; one that holds none costs no database call."""
    consume = AsyncMock(return_value=0)
    with _recording(1_500_000), _database(consume):
        await _record(_state("BUSINESS", extra=0), sells=False)
        consume.assert_not_awaited()
        await _record(_state("BUSINESS", extra=30_000_000), sells=False)
        consume.assert_awaited_once()


def test_the_debit_goes_through_a_client_that_never_retries():
    """The debit isn't idempotent: repeated after a reply lost in transit, it
    would be taken twice."""
    client = MagicMock()
    with (
        patch("backend.data.db_accessors.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ) as factory,
    ):
        assert usage_live.usage_clock_db() is client

    factory.assert_called_once_with(should_retry=False)


@pytest.mark.asyncio
async def test_a_cost_beyond_the_pool_is_taken_from_the_extra_weeks():
    consume = AsyncMock(return_value=1_500_000)
    with _recording(1_500_000) as spend, _database(consume):
        await _record(_state(extra=30_000_000))

    assert spend.await_args.kwargs["pool"] == POOL
    assert spend.await_args.kwargs["reserve"] == RESERVE
    consume.assert_awaited_once_with("u1", 1_500_000, START + 3600)


@pytest.mark.asyncio
async def test_an_extra_week_bought_since_the_state_was_read_is_still_drawn_on():
    consume = AsyncMock(return_value=1_500_000)
    with _recording(1_500_000), _database(consume):
        await _record(_state(extra=0))

    consume.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_cost_within_the_pool_touches_no_extra_week():
    consume = AsyncMock()
    with _recording(0), _database(consume):
        await _record(_state(extra=30_000_000))

    consume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_background_pass_never_draws_on_an_extra_week():
    consume = AsyncMock()
    with _recording(4_000_000) as spend, _database(consume):
        await _record(_state(extra=30_000_000), skip_daily=True)

    assert spend.await_args.kwargs["counts_daily"] is False
    consume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trial_never_draws_on_an_extra_week():
    consume = AsyncMock()
    with _recording(4_000_000), _database(consume):
        await _record(_state("TRIAL", extra=30_000_000))

    consume.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_debit_is_tried_once_and_does_not_fail_the_recording():
    consume = AsyncMock(side_effect=RuntimeError("database down"))
    with _recording(1_500_000), _database(consume):
        await _record(_state(extra=30_000_000))

    # Once: the debit isn't idempotent, and the cost is already counted.
    assert consume.await_count == 1
