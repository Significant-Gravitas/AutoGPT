"""Real Redis: the record script splits a cost between the day's reserve and
the week's pool atomically, and a counter never expires before its window."""

import asyncio
import time
from uuid import uuid4

import pytest
from redis.cluster import key_slot

from backend.copilot.usage_counters import (
    EXPIRY_GRACE_SECONDS,
    clear_counters,
    counter_keys,
    read_counters,
    record_spend,
)
from backend.copilot.usage_windows import UsageWindows, windows_for_period
from backend.data import usage_clock_integration_fixtures as fixtures
from backend.data.redis_client import get_redis_async

pytestmark = fixtures.pytestmark

DAY = 86_400
RESERVE, POOL = 2_000_000, 53_000_000


@pytest.fixture
def user() -> str:
    return f"counters-{uuid4()}"


@pytest.fixture
def now() -> int:
    return int(time.time())


@pytest.fixture
def windows(now: int) -> UsageWindows:
    """The first window of a 30-day period that started an hour ago."""
    return windows_for_period(now - 3600, now - 3600 + 30 * DAY, None, now)


async def _spend(user: str, windows: UsageWindows, cost: int, now: int, **kwargs):
    options = {"reserve": RESERVE, "pool": POOL, "counts_daily": True, **kwargs}
    return await record_spend(user, None, windows, cost, now=now, **options)


def test_a_users_counters_share_a_redis_cluster_slot(windows: UsageWindows):
    day_key, week_key = counter_keys("user-1", "generation-1", windows)

    assert key_slot(day_key.encode()) == key_slot(week_key.encode())
    assert day_key.endswith(f":d:{windows.day_start}")
    assert week_key.endswith(f":w:{windows.week_start}")
    assert counter_keys("user-1", None, windows) != (day_key, week_key)


@pytest.mark.asyncio
async def test_a_missing_counter_is_a_fresh_window(user: str, windows: UsageWindows):
    assert await read_counters(user, None, windows) == (0, 0)


@pytest.mark.asyncio
async def test_the_days_first_dollars_are_the_reserve_and_the_rest_the_pool(
    user: str, windows: UsageWindows, now: int
):
    first = await _spend(user, windows, 1_500_000, now)
    second = await _spend(user, windows, 1_500_000, now)

    # 1.5 of reserve, then 0.5 of reserve and 1.0 of pool.
    assert (first.day_used, first.week_used) == (1_500_000, 0)
    assert (second.day_used, second.week_used) == (3_000_000, 1_000_000)
    assert first.weekly_excess == second.weekly_excess == 0
    assert await read_counters(user, None, windows) == (3_000_000, 1_000_000)


@pytest.mark.asyncio
async def test_a_new_day_has_a_new_reserve_and_the_same_pool(
    user: str, windows: UsageWindows, now: int
):
    await _spend(user, windows, 5_000_000, now)
    tomorrow = windows_for_period(
        now - 3600, now - 3600 + 30 * DAY, None, windows.day_end
    )

    recorded = await _spend(user, tomorrow, 2_500_000, windows.day_end)

    assert recorded.day_used == 2_500_000
    assert recorded.week_used == 3_000_000 + 500_000


@pytest.mark.asyncio
async def test_the_part_of_a_cost_beyond_the_pool_is_reported_once(
    user: str, windows: UsageWindows, now: int
):
    await _spend(user, windows, RESERVE + POOL - 1_000_000, now)

    crossing = await _spend(user, windows, 4_000_000, now)
    beyond = await _spend(user, windows, 2_000_000, now)

    assert (crossing.weekly_excess, beyond.weekly_excess) == (3_000_000, 2_000_000)
    assert beyond.week_used == POOL + 5_000_000


@pytest.mark.asyncio
async def test_a_background_cost_counts_against_the_pool_only(
    user: str, windows: UsageWindows, now: int
):
    recorded = await _spend(user, windows, 4_000_000, now, counts_daily=False)

    assert (recorded.day_used, recorded.week_used) == (0, 4_000_000)


@pytest.mark.asyncio
async def test_without_a_weekly_cap_nothing_is_ever_beyond_the_pool(
    user: str, windows: UsageWindows, now: int
):
    recorded = await _spend(user, windows, 9_000_000_000, now, reserve=0, pool=-1)

    assert recorded.weekly_excess == 0
    assert recorded.week_used == 9_000_000_000


@pytest.mark.asyncio
async def test_costs_recorded_at_once_all_count_and_their_excess_adds_up(
    user: str, windows: UsageWindows, now: int
):
    costs = [1_000_000 + index for index in range(60)]

    recorded = await asyncio.gather(
        *(_spend(user, windows, cost, now, pool=30_000_000) for cost in costs)
    )

    day_used, week_used = await read_counters(user, None, windows)
    assert day_used == sum(costs)
    assert week_used == sum(costs) - RESERVE
    assert sum(r.weekly_excess for r in recorded) == week_used - 30_000_000


@pytest.mark.asyncio
async def test_a_counter_outlives_its_window_and_never_expires_inside_it(
    user: str, windows: UsageWindows, now: int
):
    await _spend(user, windows, 5_000_000, now)
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user, None, windows)

    assert await redis.expiretime(day_key) == windows.day_end + EXPIRY_GRACE_SECONDS
    assert await redis.expiretime(week_key) == windows.week_end + EXPIRY_GRACE_SECONDS


@pytest.mark.asyncio
async def test_a_held_windows_counters_keep_living(user: str, now: int):
    """A period that ended days ago and Stripe hasn't confirmed: the last
    window stays open, so its counters must not expire under it."""
    held = windows_for_period(now - 40 * DAY, now - 10 * DAY, None, now)
    assert held.held

    await _spend(user, held, 5_000_000, now)

    redis = await get_redis_async()
    _, week_key = counter_keys(user, None, held)
    assert await redis.expiretime(week_key) == now + EXPIRY_GRACE_SECONDS
    assert await read_counters(user, None, held) == (5_000_000, 3_000_000)


@pytest.mark.asyncio
@pytest.mark.parametrize("weekly", [False, True])
async def test_clearing_zeroes_the_day_and_the_week_only_when_asked(
    user: str, windows: UsageWindows, now: int, weekly: bool
):
    await _spend(user, windows, 5_000_000, now)

    await clear_counters(user, None, windows, weekly=weekly)

    assert await read_counters(user, None, windows) == (0, 0 if weekly else 3_000_000)


@pytest.mark.asyncio
async def test_a_corrupt_counter_is_an_error_not_a_zero(
    user: str, windows: UsageWindows
):
    redis = await get_redis_async()
    day_key, _ = counter_keys(user, None, windows)
    await redis.set(day_key, "not-a-number", ex=60)

    with pytest.raises(ValueError):
        await read_counters(user, None, windows)
