"""Redis counters of plan-allowance spend, one per window of the billing clock.

Each key is named after the window it counts, so a new window is a new key: a
counter can neither clear before its window ends nor linger into the next
one. Expiry is only garbage collection: two days after a day ends, forty
after a week's window ends (a held window must keep its counter).

Two counters per user, in microdollars:

- the **day** counter: spend in the current 24-hour day of the window;
- the **week** counter: spend against the window's main pool, which is each
  day's spend above that day's reserve. With no reserve it is simply the
  week's spend.

A paid usage reset starts both over for the rest of its window: the keys also
carry how many resets the window has had, so each reset reads fresh counters
and the spend before it stays where it was.

Both carry the user id as a Redis Cluster hash tag, so they share a slot and
one script updates them together. Callers decide what a Redis failure means:
nothing here swallows one.
"""

from pydantic import BaseModel

from backend.copilot.usage_windows import UsageWindows
from backend.data.redis_client import get_redis_async

KEY_PREFIX = "usage:v2"
# How long a day counter outlives its day, for late costs and inspection.
EXPIRY_GRACE_SECONDS = 2 * 86_400
# A week counter outlives its window far longer: while Stripe hasn't
# confirmed the next billing period the last window is held open, and a
# counter that expired under it would hand out a fresh pool on a guess.
WEEK_EXPIRY_GRACE_SECONDS = 40 * 86_400

# KEYS: day, week. ARGV: cost, today's reserve, "1" when the cost counts
# against the day, day expiry, week expiry. Returns the day and week counters.
_RECORD_SCRIPT = """
local cost = tonumber(ARGV[1])
local reserve = tonumber(ARGV[2])
local day = tonumber(redis.call("GET", KEYS[1]) or "0")
local to_pool = cost
if ARGV[3] == "1" then
    local before = day
    day = redis.call("INCRBY", KEYS[1], string.format("%d", cost))
    redis.call("EXPIREAT", KEYS[1], ARGV[4])
    to_pool = math.max(0, day - reserve) - math.max(0, before - reserve)
end
local week = tonumber(redis.call("GET", KEYS[2]) or "0")
if to_pool > 0 then
    week = redis.call("INCRBY", KEYS[2], string.format("%d", to_pool))
end
-- Every cost keeps the week counter alive, also one within the day's reserve.
redis.call("EXPIREAT", KEYS[2], ARGV[5])
return {day, week}
"""


# How long a grant is announced for: longer than a process remembers a
# user's state (``usage_clock.RECENT_STATE_SECONDS``).
RESET_MARK_SECONDS = 180


class Recorded(BaseModel):
    day_used: int
    week_used: int


def counter_keys(
    user_id: str, generation: str | None, windows: UsageWindows, resets: int = 0
) -> tuple[str, str]:
    """The day and week counter keys for the windows, after ``resets`` paid
    resets in this weekly window."""
    owner = f"{KEY_PREFIX}:{{{user_id}}}:{generation or '-'}"
    reset = f":r{resets}" if resets > 0 else ""
    return (
        f"{owner}:d:{windows.day_start}{reset}",
        f"{owner}:w:{windows.week_start}{reset}",
    )


async def read_counters(
    user_id: str, generation: str | None, windows: UsageWindows, resets: int = 0
) -> tuple[int, int]:
    """``(day_used, week_used)``; a missing counter is a fresh window."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows, resets)
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
    counts_daily: bool,
    now: int,
    resets: int = 0,
) -> Recorded:
    """Add a cost to the windows' counters in one atomic step."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows, resets)
    day_used, week_used = await redis.eval(
        _RECORD_SCRIPT,
        2,
        day_key,
        week_key,
        cost,
        max(0, reserve),
        "1" if counts_daily else "0",
        max(windows.day_end, now) + EXPIRY_GRACE_SECONDS,
        max(windows.pool_end, now) + WEEK_EXPIRY_GRACE_SECONDS,
    )
    return Recorded(day_used=int(day_used), week_used=int(week_used))


async def mark_reset(user_id: str) -> None:
    """Announce that the user was just granted a paid reset, so every process
    recording a cost re-reads their state instead of trusting a remembered
    one."""
    redis = await get_redis_async()
    await redis.set(_reset_mark_key(user_id), "1", ex=RESET_MARK_SECONDS)


async def reset_marked(user_id: str) -> bool:
    redis = await get_redis_async()
    return bool(await redis.exists(_reset_mark_key(user_id)))


def _reset_mark_key(user_id: str) -> str:
    return f"{KEY_PREFIX}:{{{user_id}}}:reset-granted"


async def clear_counters(
    user_id: str,
    generation: str | None,
    windows: UsageWindows,
    *,
    weekly: bool,
    resets: int = 0,
) -> None:
    """Zero the day counter, and the week counter too when ``weekly``."""
    redis = await get_redis_async()
    day_key, week_key = counter_keys(user_id, generation, windows, resets)
    await redis.delete(day_key)
    if weekly:
        await redis.delete(week_key)
