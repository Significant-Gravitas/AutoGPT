"""Where an account stands against its limits on the billing clock.

One snapshot answers every question the limiter is asked: may a turn start,
which limit stops it and when that refills, how much is left, and what the
usage meters show. The gate and the meters read the same snapshot, so they
can't disagree.

The rules, for the current weekly window (amounts in microdollars):

- ``day_used`` is today's spend; ``week_used`` is spend against the window's
  main pool, which is each day's spend above that day's reserve.
- A turn may start when today is under the daily limit and either today's
  reserve or the week's pool has room. The reserve is use-it-or-lose-it and
  part of the weekly limit, not on top of it.
- The **weekly limit has run out** when the pool is used up and today's
  reserve is too.
- A background pass (``skip_daily``) runs on the pool alone: never on the
  reserve, which is kept for the customer's own work.
- A **paid reset** starts the window over: until the window's own reset, which
  does not move, the customer has one full weekly limit from zero and no
  daily limit. With no daily limit there is no reserve either, so the pool is
  the whole weekly limit: exactly one weekly limit can be spent, as fast as
  the customer likes. Another reset in the same window starts over again.
"""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from backend.copilot.usage_clock import ClockSource, ResolvedClock
from backend.copilot.usage_counters import Recorded, read_counters, record_spend
from backend.copilot.usage_windows import (
    UsageWindows,
    main_pool,
    reserve_today,
    weekly_limit_for,
)
from backend.data.pro_activation import UsageActivationState

logger = logging.getLogger(__name__)

BlockingWindow = Literal["daily", "weekly"]


class UsageSnapshot(BaseModel):
    """An account's limits and counters at one moment."""

    model_config = ConfigDict(frozen=True)

    windows: UsageWindows
    source: ClockSource
    # Negative: no cap (also while a paid reset has lifted it). 0: nothing allowed.
    daily_limit: int
    # The window's weekly limit and its part above the days' reserves.
    weekly_limit: int
    pool: int
    # Today's reserve.
    reserve: int
    day_used: int
    week_used: int
    # Paid resets granted in this weekly window; the counters are the ones
    # since the last of them.
    resets: int = 0

    @property
    def reset_active(self) -> bool:
        return self.resets > 0

    @property
    def reserve_left(self) -> int:
        return max(0, self.reserve - self.day_used)

    @property
    def daily_out(self) -> bool:
        return 0 <= self.daily_limit <= self.day_used

    @property
    def pool_out(self) -> bool:
        return 0 <= self.pool <= self.week_used

    @property
    def weekly_out(self) -> bool:
        """The week's pool and today's reserve are both used up."""
        return self.pool_out and self.reserve_left == 0

    def blocked_by(self, *, skip_daily: bool = False) -> BlockingWindow | None:
        """The limit that stops a turn starting now, or None when one may.

        With both used up it is the weekly one: it refills last.
        """
        if skip_daily:
            return "weekly" if self.pool_out else None
        if self.weekly_out:
            return "weekly"
        return "daily" if self.daily_out else None

    def left(self) -> int | None:
        """Microdollars a turn starting now could spend; None for no cap."""
        daily_left = (
            max(0, self.daily_limit - self.day_used) if self.daily_limit >= 0 else None
        )
        week_left = (
            max(0, self.pool - self.week_used) + self.reserve_left
            if self.pool >= 0
            else None
        )
        if week_left is None:
            return daily_left
        if daily_left is None:
            return week_left
        return min(daily_left, week_left)


def resets_in_window(state: UsageActivationState, windows: UsageWindows) -> int:
    """How many paid resets apply now: those granted in this weekly window.
    Never for a trial or an account without a plan."""
    if state.tier.value in ("TRIAL", "NO_TIER"):
        return 0
    if state.reset_week_start != windows.week_start:
        return 0
    return max(0, state.reset_count)


async def take_snapshot(
    state: UsageActivationState,
    clock: ResolvedClock,
    *,
    daily_limit: int,
    weekly_limit: int,
    reserve: int,
) -> UsageSnapshot:
    """Read the account's counters for its current windows.

    Raises whatever Redis raises (and ``ValueError`` for a corrupt counter):
    the caller decides whether that fails the check closed.
    """
    resets = _usable_resets(state, clock.windows, weekly_limit)
    day_used, week_used = await read_counters(
        state.user_id, state.generation, clock.windows, resets
    )
    return build_snapshot(
        state,
        clock,
        daily_limit=daily_limit,
        weekly_limit=weekly_limit,
        reserve=reserve,
        day_used=day_used,
        week_used=week_used,
    )


def build_snapshot(
    state: UsageActivationState,
    clock: ResolvedClock,
    *,
    daily_limit: int,
    weekly_limit: int,
    reserve: int,
    day_used: int,
    week_used: int,
) -> UsageSnapshot:
    windows = clock.windows
    resets = _usable_resets(state, windows, weekly_limit)
    daily_limit, reserve = _after_resets(resets, daily_limit, reserve)
    reserve = usable_reserve(reserve, daily_limit, weekly_limit)
    return UsageSnapshot(
        windows=windows,
        source=clock.source,
        daily_limit=daily_limit,
        weekly_limit=weekly_limit_for(weekly_limit, windows),
        pool=main_pool(weekly_limit, reserve, windows),
        reserve=reserve_today(reserve, windows),
        day_used=day_used,
        week_used=week_used,
        resets=resets,
    )


def usable_reserve(reserve: int, daily_limit: int, weekly_limit: int) -> int:
    """A plan's reserve, only where both of its windows are capped: an
    uncapped install has no pool to keep a reserve out of."""
    if daily_limit <= 0 or weekly_limit <= 0:
        return 0
    return max(0, min(reserve, daily_limit))


async def record_live(
    state: UsageActivationState,
    generation: str | None,
    windows: UsageWindows,
    cost: int,
    *,
    weekly_limit: int,
    daily_limit: int,
    reserve: int,
    skip_daily: bool,
    now: int,
) -> Recorded:
    """Count a cost in the current windows' counters: the ones since the last
    paid reset, when the window has had one."""
    resets = _usable_resets(state, windows, weekly_limit)
    daily_limit, reserve = _after_resets(resets, daily_limit, reserve)
    reserve = usable_reserve(reserve, daily_limit, weekly_limit)
    return await record_spend(
        state.user_id,
        generation,
        windows,
        cost,
        reserve=reserve_today(reserve, windows),
        counts_daily=not skip_daily,
        now=now,
        resets=resets,
    )


def _after_resets(resets: int, daily_limit: int, reserve: int) -> tuple[int, int]:
    """The daily limit and reserve once the window has had a paid reset: no
    daily limit and so no reserve, which makes the pool the whole weekly
    limit. A daily limit of 0 is a switch-off, which a reset doesn't lift."""
    if resets > 0 and daily_limit != 0:
        return -1, 0
    return daily_limit, reserve


def _usable_resets(
    state: UsageActivationState, windows: UsageWindows, weekly_limit: int
) -> int:
    # A reset is a fresh weekly limit: without one there is nothing to reset.
    return resets_in_window(state, windows) if weekly_limit > 0 else 0
