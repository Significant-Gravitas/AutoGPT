"""Redis counters of plan-allowance spend, one per window of the billing clock.

Each key is named after the window it counts, so a new window is a new key: a
counter can neither clear before its window ends nor linger into the next
one. Expiry is only garbage collection, two days after the window ends.

Two counters per user, in microdollars:

- the **day** counter: spend in the current 24-hour day of the window;
- the **week** counter: spend against the window's main pool, which is each
  day's spend above that day's reserve. With no reserve it is simply the
  week's spend.

Both carry the user id as a Redis Cluster hash tag, so they share a slot and
one script updates them together. Callers decide what a Redis failure means:
nothing here swallows one.
"""

from pydantic import BaseModel

from backend.copilot.usage_windows import UsageWindows
from backend.data.redis_client import get_redis_async

KEY_PREFIX = "usage:v2"
# How long a counter outlives its window, for late costs and inspection.
EXPIRY_GRACE_SECONDS = 2 * 86_400

# KEYS: day, week. ARGV: cost, today's reserve, main pool (negative: no cap),
# "1" when the cost counts against the day, day expiry, week expiry.
# Returns the day counter, the week counter, and how much of this cost fell
# beyond the main pool. The excess of every cost adds up to exactly what the
# week counter is above the pool, however costs interleave.
_RECORD_SCRIPT = """
local cost = tonumber(ARGV[1])
local reserve = tonumber(ARGV[2])
local pool = tonumber(ARGV[3])
local day = tonumber(redis.call("GET", KEYS[1]) or "0")
local to_pool = cost
if ARGV[4] == "1" then
    local before = day
    day = redis.call("INCRBY", KEYS[1], string.format("%d", cost))
    redis.call("EXPIREAT", KEYS[1], ARGV[5])
    to_pool = math.max(0, day - reserve) - math.max(0, before - reserve)
end
local week = tonumber(redis.call("GET", KEYS[2]) or "0")
local before_week = week
if to_pool > 0 then
    week = redis.call("INCRBY", KEYS[2], string.format("%d", to_pool))
    redis.call("EXPIREAT", KEYS[2], ARGV[6])
end
local excess = 0
if pool >= 0 then
    excess = math.max(0, week - pool) - math.max(0, before_week - pool)
end
return {day, week, excess}
"""


class Recorded(BaseModel):
    day_used: int
    week_used: int
    # The part of the recorded cost beyond the window's main pool.
    weekly_excess: int


def counter_keys(
    user_id: str, generation: str | None, windows: UsageWindows
) -> tuple[str, str]:
    """The day and week counter keys for the windows."""
    owner = f"{KEY_PREFIX}:{{{user_id}}}:{generation or '-'}"
    return f"{owner}:d:{windows.day_start}", f"{owner}:w:{windows.week_start}"


async def read_counters(
    user_id: str, generation: str | None, windows: UsageWindows
) -> tuple[int, int]:
    """``(day_used, week_used)``; a missing counter is a fresh window."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows)
    day_raw = await redis.get(day_key)
    week_raw = await redis.get(week_key)
    return int(day_raw or 0), int(week_raw or 0)


async def record_spend(
    user_id: str,
    generation: str | None,
    windows: UsageWindows,
    cost: int,
    *,
    reserve: int,
    pool: int,
    counts_daily: bool,
    now: int,
) -> Recorded:
    """Add a cost to the windows' counters in one atomic step."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows)
    day_used, week_used, excess = await redis.eval(
        _RECORD_SCRIPT,
        2,
        day_key,
        week_key,
        cost,
        max(0, reserve),
        pool,
        "1" if counts_daily else "0",
        max(windows.day_end, now) + EXPIRY_GRACE_SECONDS,
        max(windows.pool_end, now) + EXPIRY_GRACE_SECONDS,
    )
    return Recorded(
        day_used=int(day_used), week_used=int(week_used), weekly_excess=int(excess)
    )


async def clear_counters(
    user_id: str, generation: str | None, windows: UsageWindows, *, weekly: bool
) -> None:
    """Zero the day counter, and the week counter too when ``weekly``."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows)
    await redis.delete(day_key)
    if weekly:
        await redis.delete(week_key)
