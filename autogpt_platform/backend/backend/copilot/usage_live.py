"""Where an account stands against its limits on the billing clock.

One snapshot answers every question the limiter is asked: may a turn start,
which limit stops it and when that refills, how much is left, and what the
usage meters show. The gate and the meters read the same snapshot, so they
can't disagree.

The rules, for the current weekly window (amounts in microdollars):

- ``day_used`` is today's spend; ``week_used`` is spend against the window's
  main pool, which is each day's spend above that day's reserve.
- A turn may start on the plan's own allowance when today is under the daily
  limit and either today's reserve or the week's pool has room. The reserve
  is use-it-or-lose-it and part of the weekly limit, not on top of it.
- The **weekly limit has run out** when the pool is used up and today's
  reserve is too. Only then does a paid extra week apply; the daily limit
  doesn't apply to it.
- A background pass (``skip_daily``) runs on the pool alone: never on the
  reserve, which is kept for the customer's own work, nor on an extra week.
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
from backend.data.db_accessors import usage_clock_db
from backend.data.pro_activation import UsageActivationState

logger = logging.getLogger(__name__)

BlockingWindow = Literal["daily", "weekly"]


class UsageSnapshot(BaseModel):
    """An account's limits, counters and extra weeks at one moment."""

    model_config = ConfigDict(frozen=True)

    windows: UsageWindows
    source: ClockSource
    # Negative: no cap. 0: nothing allowed.
    daily_limit: int
    # The window's weekly limit and its part above the days' reserves.
    weekly_limit: int
    pool: int
    # Today's reserve, and the plan's figure for a full day.
    reserve: int
    day_used: int
    week_used: int
    extra_remaining: int = 0
    extra_total: int = 0
    extra_expires_at: int | None = None
    extra_count: int = 0

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

    @property
    def extra_applies(self) -> bool:
        """A paid extra week covers the next turn: it has something left and
        the weekly limit has run out."""
        return self.extra_remaining > 0 and self.weekly_out

    def blocked_by(self, *, skip_daily: bool = False) -> BlockingWindow | None:
        """The limit that stops a turn starting now, or None when one may.

        With both used up it is the weekly one: it refills last.
        """
        if skip_daily:
            return "weekly" if self.pool_out else None
        if self.weekly_out:
            return None if self.extra_applies else "weekly"
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
        if daily_left is not None and daily_left < week_left:
            return daily_left
        # The week is what runs out first, and an extra week takes over there.
        return week_left + self.extra_remaining


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
    day_used, week_used = await read_counters(
        state.user_id, state.generation, clock.windows
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
    reserve = usable_reserve(reserve, daily_limit, weekly_limit)
    extra = extra_weeks(state, weekly_limit)
    return UsageSnapshot(
        windows=windows,
        source=clock.source,
        daily_limit=daily_limit,
        weekly_limit=weekly_limit_for(weekly_limit, windows),
        pool=main_pool(weekly_limit, reserve, windows),
        reserve=reserve_today(reserve, windows),
        day_used=day_used,
        week_used=week_used,
        extra_remaining=extra.remaining,
        extra_total=extra.total,
        extra_expires_at=extra.expires_at,
        extra_count=extra.count,
    )


def usable_reserve(reserve: int, daily_limit: int, weekly_limit: int) -> int:
    """A plan's reserve, only where both of its windows are capped: an
    uncapped install has no pool to keep a reserve out of."""
    if daily_limit <= 0 or weekly_limit <= 0:
        return 0
    return max(0, min(reserve, daily_limit))


class ExtraWeeks(BaseModel):
    """The account's usable extra weeks, taken together."""

    remaining: int = 0
    total: int = 0
    expires_at: int | None = None
    count: int = 0


def extra_weeks(state: UsageActivationState, weekly_limit: int) -> ExtraWeeks:
    """The account's extra weeks. They only count on a plan with a weekly
    limit: never for a trial, and never as a plan."""
    if state.tier.value in ("TRIAL", "NO_TIER") or weekly_limit <= 0:
        return ExtraWeeks()
    return ExtraWeeks(
        remaining=max(0, state.extra_remaining),
        total=max(0, state.extra_total),
        expires_at=state.extra_expires_at,
        count=state.extra_count,
    )


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
    sells_extra_weeks: bool = True,
) -> Recorded:
    """Count a cost in the current windows, then take the part beyond the
    week's pool from the account's extra weeks.

    The counters first: if the extra week's debit then fails, the cost is
    still counted against the plan and the customer is never charged twice.
    A background pass never draws on an extra week.
    """
    reserve = usable_reserve(reserve, daily_limit, weekly_limit)
    recorded = await record_spend(
        state.user_id,
        generation,
        windows,
        cost,
        reserve=reserve_today(reserve, windows),
        pool=main_pool(weekly_limit, reserve, windows),
        counts_daily=not skip_daily,
        now=now,
    )
    # A plan that sells extra weeks is asked of the database whenever a cost
    # goes beyond the pool, not of the state in hand: an extra week bought
    # seconds ago must still be drawn on. Any other plan only when the state
    # shows one it kept from an earlier plan.
    may_hold_extra = (
        state.tier.value not in ("TRIAL", "NO_TIER")
        and weekly_limit > 0
        and (sells_extra_weeks or state.extra_remaining > 0)
    )
    if recorded.weekly_excess > 0 and may_hold_extra and not skip_daily:
        await _debit_extra_weeks(state.user_id, recorded.weekly_excess, now)
    return recorded


async def _debit_extra_weeks(user_id: str, amount: int, now: int) -> None:
    # One attempt, on a client that doesn't retry: the debit isn't
    # idempotent, and a reply lost after it committed would otherwise take it
    # twice.
    try:
        await usage_clock_db().consume_extra_allowance(user_id, amount, now)
    except Exception:
        logger.error(
            f"Could not take {amount} microdollars from the extra weeks of "
            f"user {user_id[:8]}; it stays counted against the plan's week",
            exc_info=True,
        )
