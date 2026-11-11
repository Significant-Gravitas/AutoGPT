"""The limiter on the billing clock (SECRT-3441): what each public function
does from the launch second, with the clock, the counters and the state
pinned. The calendar behaviour before launch is ``rate_limit_test.py``."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import SubscriptionTier as PrismaTier
from redis.exceptions import RedisError

from backend.copilot import at_reset, budget_gate, rate_limit, usage_limits, usage_live
from backend.copilot.rate_limit import (
    CoPilotUsagePublic,
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
    check_rate_limit,
    get_global_rate_limits,
    get_remaining_usd_budget,
    get_usage_status,
    record_cost_usage,
    reset_user_usage,
)
from backend.copilot.usage_clock import forget_recent_states
from backend.copilot.usage_counters import Recorded
from backend.data.pro_activation import UsageActivationState

DAY = 86_400
START = 1_793_824_256  # Wed 4 Nov 2026 20:30:56 UTC
END = START + 30 * DAY
NOW = START + 3600
WEEK_END = START + 7 * DAY + 12 * 3600  # Thu 12 Nov 08:30:56
USER = "user-billing-clock"
WEEKLY, DAILY, RESERVE, POOL = 68_000_000, 22_666_667, 2_000_000, 53_000_000
BASE_DAILY, BASE_WEEKLY = 2_500_000, 5_000_000


def _state(tier: str = "MAX", **fields) -> UsageActivationState:
    values = {
        "user_id": USER,
        "tier": PrismaTier(tier),
        "ready": True,
        "created_at": 1_750_000_000,
        "has_stripe_customer": True,
        "period_start": START,
        "period_end": END,
        "billing_anchor": START,
    }
    return UsageActivationState(**{**values, **fields})


class Account:
    """The pinned world of one test: its usage state, counters and clock."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.state = _state()
        self.counters = (0, 0)
        self.now = NOW
        self.live = True
        self.read_counters = AsyncMock(side_effect=lambda *_: self.counters)
        self.calendar_redis = AsyncMock(side_effect=AssertionError("calendar keys"))
        monkeypatch.setattr(
            rate_limit,
            "billing_clock_live",
            AsyncMock(side_effect=lambda *_: self.live),
        )
        monkeypatch.setattr(rate_limit, "now_seconds", lambda: self.now)
        monkeypatch.setattr(
            rate_limit, "get_ready_usage_state", AsyncMock(side_effect=self._read)
        )
        monkeypatch.setattr(usage_live, "read_counters", self.read_counters)
        monkeypatch.setattr(rate_limit, "get_redis_async", self.calendar_redis)

    async def _read(self, _user_id: str) -> UsageActivationState:
        return self.state

    def has(self, day_used: int, week_used: int, **state_fields) -> None:
        self.counters = (day_used, week_used)
        if state_fields:
            self.state = self.state.model_copy(update=state_fields)


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> Account:
    usage_limits._fetch_plan_limits_flag.cache_clear()
    rate_limit._fetch_cost_limits_flag.cache_clear()
    rate_limit._fetch_tier_multipliers_flag.cache_clear()
    forget_recent_states()
    return Account(monkeypatch)


async def _limits(account: Account, tier: str, *, live: bool) -> tuple[int, int]:
    account.state = _state(tier)
    account.live = live
    daily, weekly, _ = await get_global_rate_limits(USER, BASE_DAILY, BASE_WEEKLY)
    return daily, weekly


@pytest.mark.asyncio
async def test_pro_and_max_take_their_own_limits_from_launch(account: Account):
    assert await _limits(account, "PRO", live=False) == (3_125_000, 6_250_000)
    assert await _limits(account, "PRO", live=True) == (3_200_000, 8_000_000)
    assert await _limits(account, "MAX", live=False) == (26_666_750, 53_333_500)
    assert await _limits(account, "MAX", live=True) == (DAILY, WEEKLY)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier,expected",
    [
        ("BASIC", (2_500_000, 5_000_000)),
        ("BUSINESS", (150_000_000, 300_000_000)),
        ("ENTERPRISE", (150_000_000, 300_000_000)),
    ],
)
async def test_every_other_plan_keeps_its_limit_values(
    account: Account, tier: str, expected: tuple[int, int]
):
    before = await _limits(account, tier, live=False)
    after = await _limits(account, tier, live=True)

    assert before == after == expected


@pytest.mark.asyncio
async def test_an_uncapped_install_stays_uncapped_for_pro(account: Account):
    account.state = _state("PRO")

    assert (await get_global_rate_limits(USER, -1, -1))[:2] == (-1, -1)
    assert (await get_global_rate_limits(USER, -1, BASE_WEEKLY))[:2] == (-1, 8_000_000)


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_a_base_limit_of_zero_still_stops_all_spend(account: Account, tier: str):
    """Setting the base limits to 0 is the switch that stops everyone's
    spend; a plan's own limits must not undo it."""
    account.state = _state(tier)

    assert (await get_global_rate_limits(USER, 0, 0))[:2] == (0, 0)


async def _check(account: Account, **kwargs) -> None:
    await check_rate_limit(
        USER, DAILY, WEEKLY, expected_tier=account.state.tier.value, **kwargs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "day_used,week_used,extra",
    [
        (0, 0, 0),
        (DAILY - 1, 20_000_000, 0),
        (0, POOL, 0),  # week's pool gone, today's reserve untouched
        (RESERVE - 1, POOL + 5, 0),
        (RESERVE, POOL, 1),  # the weekly limit ran out: a paid extra week
        (DAILY * 2, POOL, 30_000_000),  # the daily limit doesn't apply to it
    ],
)
async def test_a_turn_may_start(
    account: Account, day_used: int, week_used: int, extra: int
):
    account.has(day_used, week_used, extra_remaining=extra, extra_total=WEEKLY)

    await _check(account)

    # Only the billing clock's counters were read.
    account.calendar_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_daily_limit_stops_a_turn_until_the_customers_next_day(
    account: Account,
):
    account.has(DAILY, 20_666_667)

    with pytest.raises(RateLimitExceeded) as stopped:
        await _check(account)

    assert stopped.value.window == "daily"
    # 24 hours from the weekly reset, not midnight UTC.
    assert stopped.value.resets_at == datetime.fromtimestamp(START + DAY, UTC)
    assert stopped.value.reading.cap == 22.666667


@pytest.mark.asyncio
async def test_the_weekly_limit_stops_a_turn_until_the_customers_weekly_reset(
    account: Account,
):
    account.has(RESERVE, POOL)

    with pytest.raises(RateLimitExceeded) as stopped:
        await _check(account)

    assert stopped.value.window == "weekly"
    assert stopped.value.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    # Reported against the whole weekly limit, reserves included.
    assert (stopped.value.reading.used, stopped.value.reading.cap) == (68.0, 68.0)


@pytest.mark.asyncio
async def test_an_extra_week_does_not_lift_the_daily_limit_while_the_week_has_room(
    account: Account,
):
    account.has(DAILY, 20_666_667, extra_remaining=30_000_000, extra_total=WEEKLY)

    with pytest.raises(RateLimitExceeded) as stopped:
        await _check(account)

    assert stopped.value.window == "daily"


@pytest.mark.asyncio
async def test_a_background_pass_uses_neither_the_reserve_nor_an_extra_week(
    account: Account,
):
    account.has(0, POOL, extra_remaining=30_000_000, extra_total=WEEKLY)

    with pytest.raises(RateLimitExceeded) as stopped:
        await _check(account, skip_daily=True)

    assert stopped.value.window == "weekly"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RedisError("down"), OSError(), ValueError("x")])
async def test_unreadable_counters_fail_the_check_closed(
    account: Account, failure: Exception
):
    account.read_counters.side_effect = failure

    with pytest.raises(RateLimitUnavailable):
        await _check(account)


@pytest.mark.asyncio
async def test_a_plan_change_between_the_limits_and_the_check_is_unavailable(
    account: Account,
):
    with pytest.raises(RateLimitUnavailable):
        await check_rate_limit(USER, DAILY, WEEKLY, expected_tier="PRO")


@pytest.mark.asyncio
async def test_a_state_without_a_clock_falls_back_to_the_calendar(account: Account):
    """A database manager still on the previous release sends no clock: the
    check runs on the calendar counters rather than failing."""
    account.state = UsageActivationState(user_id=USER, tier=PrismaTier.MAX, ready=True)
    redis = MagicMock(get=AsyncMock(return_value=None))
    account.calendar_redis.side_effect = None
    account.calendar_redis.return_value = redis

    await _check(account)

    assert redis.get.await_count == 2
    account.read_counters.assert_not_awaited()


async def _status(account: Account, **kwargs):
    return await get_usage_status(
        USER, DAILY, WEEKLY, tier=SubscriptionTier(account.state.tier.value), **kwargs
    )


@pytest.mark.asyncio
async def test_the_status_reports_the_customers_own_resets_and_reserve(
    account: Account,
):
    account.has(1_500_000, 10_600_000)

    status = await _status(account)

    assert status.on_billing_clock and status.clock_source == "stripe"
    assert status.daily.resets_at == datetime.fromtimestamp(START + DAY, UTC)
    assert status.weekly.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    # The week's meter is its pool; the reserve has its own.
    assert (status.weekly.used, status.weekly.limit) == (10_600_000, POOL)
    assert status.weekly_limit == WEEKLY
    assert status.reserve is not None
    assert (status.reserve.used, status.reserve.limit) == (1_500_000, RESERVE)
    assert status.reserve.resets_at == status.daily.resets_at
    assert status.blocked_by is None and status.extra is None
    assert status.extra_week_price == 8_000
    public = CoPilotUsagePublic.from_status(status)
    assert public.state == "ok" and public.reset_cost == 0
    assert public.reserve is not None and public.reserve.percent_used == 75.0
    assert public.weekly is not None and public.weekly.percent_used == 20.0
    assert public.extra_week_offer is not None
    assert public.extra_week_offer.price_credits == 8_000


@pytest.mark.asyncio
async def test_a_used_up_week_with_reserve_left_is_low_not_exhausted(account: Account):
    account.has(500_000, POOL)

    public = CoPilotUsagePublic.from_status(await _status(account))

    assert public.weekly is not None and public.weekly.percent_used == 100.0
    assert (public.state, public.limiting_window) == ("low", "weekly")


@pytest.mark.asyncio
async def test_a_used_up_week_is_exhausted_until_the_weekly_reset(account: Account):
    account.has(RESERVE, POOL)

    status = await _status(account)
    public = CoPilotUsagePublic.from_status(status)

    assert status.blocked_by == "weekly"
    assert (public.state, public.limiting_window) == ("exhausted", "weekly")
    assert public.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    # The small daily reserve comes back first.
    assert public.reserve is not None
    assert public.reserve.resets_at == datetime.fromtimestamp(START + DAY, UTC)


@pytest.mark.asyncio
async def test_a_used_up_day_is_exhausted_until_the_daily_reset(account: Account):
    account.has(DAILY, 20_666_667)

    public = CoPilotUsagePublic.from_status(await _status(account))

    assert (public.state, public.limiting_window) == ("exhausted", "daily")
    assert public.resets_at == datetime.fromtimestamp(START + DAY, UTC)


@pytest.mark.asyncio
async def test_an_extra_week_keeps_a_used_up_week_going_and_is_shown(account: Account):
    account.has(
        DAILY,
        POOL,
        extra_remaining=51_000_000,
        extra_total=68_000_000,
        extra_expires_at=START + 20 * DAY,
        extra_count=1,
    )

    status = await _status(account)
    public = CoPilotUsagePublic.from_status(status)

    assert status.blocked_by is None
    assert public.state != "exhausted"
    assert public.extra is not None
    assert (public.extra.percent_used, public.extra.count) == (25.0, 1)
    assert public.extra.expires_at == datetime.fromtimestamp(START + 20 * DAY, UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("tier,price", [("PRO", 1_250), ("BUSINESS", 0), ("BASIC", 0)])
async def test_only_plans_that_sell_an_extra_week_offer_one(
    account: Account, tier: str, price: int
):
    account.state = _state(tier)

    status = await get_usage_status(
        USER, 3_200_000, 8_000_000, tier=SubscriptionTier(tier)
    )

    assert status.extra_week_price == price
    assert (CoPilotUsagePublic.from_status(status).extra_week_offer is None) == (
        price == 0
    )


@pytest.mark.asyncio
async def test_a_plan_without_a_reserve_shows_none(account: Account):
    account.state = _state("BUSINESS")

    status = await get_usage_status(
        USER, 150_000_000, 300_000_000, tier=SubscriptionTier.BUSINESS
    )

    assert status.reserve is None
    assert status.weekly.limit == 300_000_000


@pytest.mark.asyncio
async def test_a_display_with_unreadable_counters_shows_no_usage_unless_it_must_know(
    account: Account,
):
    account.read_counters.side_effect = RedisError("down")

    shown = await _status(account)
    assert (shown.daily.used, shown.weekly.used) == (0, 0)
    assert shown.weekly.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    with pytest.raises(RateLimitUnavailable):
        await _status(account, fail_closed=True)
    account.has(0, 0, generation="generation-1")
    with pytest.raises(RateLimitUnavailable):
        await _status(account)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "day_used,week_used,extra,left_usd",
    [
        (0, 0, 0, 22.666667),
        (20_000_000, 52_000_000, 0, 1.0),  # the pool binds, reserve already used
        (0, POOL, 0, 2.0),  # only today's reserve
        (RESERVE, POOL, 0, 0.5),  # nothing left: the floor
        (RESERVE, POOL, 40_000_000, 40.0),  # an extra week
    ],
)
async def test_the_remaining_budget(
    account: Account, day_used: int, week_used: int, extra: int, left_usd: float
):
    account.has(day_used, week_used, extra_remaining=extra, extra_total=WEEKLY)

    left = await get_remaining_usd_budget(USER, DAILY, WEEKLY, expected_tier="MAX")

    assert left == pytest.approx(left_usd)


@pytest.mark.asyncio
async def test_the_remaining_budget_is_the_floor_when_counters_are_unreadable(
    account: Account,
):
    account.read_counters.side_effect = RedisError("down")

    assert await get_remaining_usd_budget(USER, DAILY, WEEKLY, floor_usd=-1.0) == -1.0


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    record = AsyncMock(return_value=Recorded(day_used=1, week_used=1, weekly_excess=0))
    monkeypatch.setattr(rate_limit, "record_live", record)
    monkeypatch.setattr(
        rate_limit,
        "capture_cost_context",
        AsyncMock(return_value=MagicMock(trial_id=None, generation=None)),
    )
    return record


@pytest.mark.asyncio
async def test_a_cost_is_recorded_in_the_window_it_finishes_in(
    account: Account, recorded: AsyncMock
):
    """A turn checked before a weekly reset and finished after it lands in
    the new window."""
    account.now = WEEK_END + 5

    await record_cost_usage(USER, 4_000_000)

    windows = recorded.await_args.args[2]
    assert (windows.week_start, windows.day_start) == (WEEK_END, WEEK_END)
    assert recorded.await_args.kwargs == {
        "daily_limit": DAILY,
        "weekly_limit": WEEKLY,
        "reserve": RESERVE,
        "skip_daily": False,
        "now": WEEK_END + 5,
        "sells_extra_weeks": True,
    }
    account.calendar_redis.assert_not_awaited()


@pytest.mark.asyncio
async def test_recording_reuses_the_state_the_check_just_read(
    account: Account, recorded: AsyncMock
):
    await _check(account)
    reads = rate_limit.get_ready_usage_state.await_count

    await record_cost_usage(USER, 4_000_000)
    await record_cost_usage(USER, 1_000_000, skip_daily=True)

    assert rate_limit.get_ready_usage_state.await_count == reads
    assert recorded.await_count == 2
    assert recorded.await_args.kwargs["skip_daily"] is True


@pytest.mark.asyncio
async def test_a_cost_after_the_period_ended_reads_the_state_again(
    account: Account, recorded: AsyncMock
):
    await _check(account)
    reads = rate_limit.get_ready_usage_state.await_count
    account.state = _state(period_start=END, period_end=END + 31 * DAY)
    account.now = END + 10

    await record_cost_usage(USER, 4_000_000)

    assert rate_limit.get_ready_usage_state.await_count == reads + 1
    assert recorded.await_args.args[2].week_start == END


@pytest.mark.asyncio
async def test_a_trials_cost_is_only_counted(account: Account, recorded: AsyncMock):
    account.state = _state(
        "TRIAL", trial_started_at=NOW - DAY, trial_ends_at=NOW + 6 * DAY
    )

    await record_cost_usage(USER, 4_000_000)

    assert recorded.await_args.kwargs["reserve"] == 0
    assert recorded.await_args.kwargs["weekly_limit"] == -1
    assert recorded.await_args.kwargs["sells_extra_weeks"] is False


@pytest.mark.asyncio
async def test_a_business_cost_is_recorded_without_a_reserve_or_extra_weeks_for_sale(
    account: Account, recorded: AsyncMock
):
    account.state = _state("BUSINESS")

    await record_cost_usage(USER, 4_000_000)

    assert recorded.await_args.kwargs["weekly_limit"] == 300_000_000
    assert recorded.await_args.kwargs["reserve"] == 0
    assert recorded.await_args.kwargs["sells_extra_weeks"] is False


@pytest.mark.asyncio
async def test_a_redis_failure_while_recording_does_not_fail_the_turn(
    account: Account, recorded: AsyncMock
):
    recorded.side_effect = RedisError("down")

    await record_cost_usage(USER, 4_000_000)

    # And it is not counted a second time on the calendar.
    account.calendar_redis.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("weekly", [False, True])
async def test_an_admin_reset_clears_the_current_windows(
    account: Account, monkeypatch: pytest.MonkeyPatch, weekly: bool
):
    clear = AsyncMock()
    monkeypatch.setattr(rate_limit, "clear_counters", clear)

    await reset_user_usage(USER, reset_weekly=weekly)

    windows = clear.await_args.args[2]
    assert (windows.week_start, windows.day_start) == (START, START)
    assert clear.await_args.kwargs == {"weekly": weekly}
    account.calendar_redis.assert_not_awaited()


@pytest.fixture
def plan_limits(account: Account, monkeypatch: pytest.MonkeyPatch) -> Account:
    limits = AsyncMock(return_value=(DAILY, WEEKLY, SubscriptionTier.MAX))
    monkeypatch.setattr(budget_gate, "get_global_rate_limits", limits)
    monkeypatch.setattr(at_reset, "get_global_rate_limits", limits)
    return account


@pytest.mark.asyncio
async def test_an_extra_week_pays_before_the_credit_balance_does(plan_limits: Account):
    """The order of payers: the plan's allowance, then an extra week, and only
    when both refuse does a block turn fall to the credit balance. The gate
    that decides the last step sees no refusal while an extra week is left."""
    plan_limits.has(RESERVE, POOL, extra_remaining=30_000_000, extra_total=WEEKLY)
    assert await budget_gate.refusal_for(USER) is None

    plan_limits.has(RESERVE, POOL, extra_remaining=0)
    pause = await budget_gate.refusal_for(USER)

    assert pause is not None and pause.reason == "weekly"
    assert pause.resets_at == datetime.fromtimestamp(WEEK_END, UTC)


@pytest.mark.asyncio
async def test_send_at_reset_waits_for_the_limit_that_stops_a_turn(
    plan_limits: Account,
):
    plan_limits.has(500_000, POOL)  # week used up, today's reserve still open
    with pytest.raises(at_reset.NoResetAhead) as none:
        await at_reset._blocking_reset(USER)
    assert none.value.reason == "not_out_of_budget"

    plan_limits.has(RESERVE, POOL)
    # The weekly reset, not the small reserve that arrives the next day.
    assert await at_reset._blocking_reset(USER) == datetime.fromtimestamp(WEEK_END, UTC)

    plan_limits.has(DAILY, 20_666_667)
    assert await at_reset._blocking_reset(USER) == datetime.fromtimestamp(
        START + DAY, UTC
    )


@pytest.mark.asyncio
async def test_a_cost_for_a_state_without_a_clock_is_counted_on_the_calendar(
    account: Account, recorded: AsyncMock, monkeypatch: pytest.MonkeyPatch
):
    """A database manager still on the previous release sends no clock: the
    cost must still be counted somewhere, so it goes to the calendar keys."""
    account.state = UsageActivationState(user_id=USER, tier=PrismaTier.MAX, ready=True)
    calendar = AsyncMock()
    monkeypatch.setattr(rate_limit, "_incr_counter_atomic", calendar)
    account.calendar_redis.side_effect = None

    await record_cost_usage(USER, 4_000_000)

    recorded.assert_not_awaited()
    assert calendar.await_count == 2  # the day's and the week's calendar counters


@pytest.mark.asyncio
async def test_a_trials_single_allowance_is_shown_as_before(
    account: Account, monkeypatch: pytest.MonkeyPatch
):
    """A trial's lifetime allowance decides on its own, on either clock: the
    status carries no reserve, no extra week and no billing-clock verdict."""
    account.state = _state(
        "TRIAL", trial_started_at=NOW - DAY, trial_ends_at=NOW + 6 * DAY
    )
    offer = MagicMock(total_cost_limit=20_000_000)
    trial = MagicMock(active=True, cost_microdollars=5_000_000, offer=offer)
    trial.ends_at = datetime.fromtimestamp(NOW + 6 * DAY, UTC)
    monkeypatch.setattr(
        rate_limit, "_subscription_trial", AsyncMock(return_value=trial)
    )

    status = await get_usage_status(
        USER, 100_000_000, 20_000_000, tier=SubscriptionTier.TRIAL
    )

    assert status.weekly_is_trial_allowance and not status.on_billing_clock
    assert (status.weekly.used, status.weekly.limit) == (5_000_000, 20_000_000)
    assert status.weekly.resets_at == trial.ends_at
    assert status.reserve is None and status.extra_week_price == 0
