"""How much a turn may spend, and which limit that figure comes from.

A turn's budget is the smaller of a per-task cap (a runaway guard set well
above a normal task) and what the account has left in its tightest window.
Which one binds decides what the user is told when the turn pauses -- out of
today's budget, out of this week's, trial allowance used, or this task
reached its size limit -- and when the first two can pick back up.

Unlike the old per-query budget there is no floor: a user with a cent left
gets a cent, not a $0.50 top-up. The turn lands with a checkpoint instead of
overspending, and the CLI's own budget flag is sized separately as a loose
backstop (see ``governor.cli_budget_usd``).
"""

import math
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
    get_global_rate_limits,
    get_remaining_usd_budget,
    get_usage_status,
)
from backend.copilot.sdk.trial_budget import resolve_trial_sdk_budget
from backend.data.db_accessors import delegation_db

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession

BudgetWindow = Literal["daily", "weekly", "trial", "task"]

config = ChatConfig()

# Remaining amounts are microdollar integers converted to USD; anything this
# close counts as the same figure.
_SAME_USD = 1e-6


class TurnBudget(BaseModel):
    """The money one turn may spend and the limit that sets it."""

    usd: float
    limiting: BudgetWindow
    # When the limiting window refills: daily and weekly only. A trial's
    # allowance never refills, and the task cap applies afresh per turn.
    resets_at: datetime | None = None
    # Spend past ``usd`` is never allowed, whichever limit binds: a trial's
    # allowance is real money with no plan behind it, so the CLI backstop
    # sits exactly on the budget instead of above it.
    hard_cap: bool = False
    # Set when a delegated thread's per-hand-off cap is what binds: the turn
    # that lands on it is held at the cap, not just paused.
    cap_usd: float | None = None


async def resolve_turn_budget(user_id: str | None) -> TurnBudget:
    """Budget for a platform-billed turn starting now."""
    task_cap = config.claude_agent_max_budget_usd
    whole_task = TurnBudget(usd=task_cap, limiting="task")
    if not user_id:
        return whole_task
    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    # A negative floor turns a Redis brown-out into -1 rather than a
    # made-up figure; the pre-turn gate has already failed closed on it.
    remaining = await get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        floor_usd=-1.0,
        expected_tier=tier,
    )
    trial = tier == SubscriptionTier.TRIAL
    if trial:
        # Raises when the trial has nothing left: no trial turn may start.
        remaining = resolve_trial_sdk_budget(task_cap, remaining)
    if remaining < 0 or not math.isfinite(remaining) or task_cap <= remaining:
        return whole_task.model_copy(update={"hard_cap": trial})
    window, resets_at = await _limiting_window(
        user_id, daily_limit, weekly_limit, tier, remaining
    )
    return TurnBudget(
        usd=remaining, limiting=window, resets_at=resets_at, hard_cap=trial
    )


async def within_delegation_cap(
    budget: TurnBudget | None, user_id: str | None, session: "ChatSession"
) -> TurnBudget | None:
    """*budget*, cut to what a delegated thread's per-hand-off cap has left.

    The thread's earlier turns are on the cost log; this turn's spend is the
    governor's to meter, so it lands the turn with a checkpoint before the
    cap rather than a later status poll catching the overspend. The cap binds
    as the task's size limit.
    """
    cap = session.metadata.delegation_cap_usd
    if budget is None or cap is None or not user_id:
        return budget
    left = await delegation_cap_left_usd(user_id, session, cap)
    if left >= budget.usd:
        return budget
    # Still the account's money: a trial's hard cap stays a hard cap.
    return budget.model_copy(
        update={"usd": left, "limiting": "task", "resets_at": None, "cap_usd": cap}
    )


async def delegation_cap_left_usd(
    user_id: str, session: "ChatSession", cap_usd: float
) -> float:
    """What a delegated thread may still spend under its cap: the cap less its
    earlier turns, which are on the cost log by the time a new turn starts."""
    costs = await delegation_db().get_session_costs(user_id, [session.session_id])
    return max(0.0, cap_usd - costs.get(session.session_id, 0) / 1_000_000)


async def _limiting_window(
    user_id: str,
    daily_limit: int,
    weekly_limit: int,
    tier: SubscriptionTier,
    remaining_usd: float,
) -> tuple[BudgetWindow, datetime | None]:
    """Name the window whose remainder is the turn's budget.

    When several are equally tight -- both empty, typically -- the one that
    refills last is the one that decides when work can resume.
    """
    unknown: BudgetWindow = "trial" if tier == SubscriptionTier.TRIAL else "daily"
    try:
        status = await get_usage_status(user_id, daily_limit, weekly_limit, tier=tier)
    except RateLimitUnavailable:
        return unknown, None
    binding = [
        (name, window)
        for name, window in (("daily", status.daily), ("weekly", status.weekly))
        if window.limit >= 0 and _left_usd(window) <= remaining_usd + _SAME_USD
    ]
    if not binding:
        return unknown, None
    name, window = max(binding, key=lambda pair: pair[1].resets_at)
    if tier == SubscriptionTier.TRIAL and name != "daily":
        return "trial", None
    return ("daily" if name == "daily" else "weekly"), window.resets_at


def _left_usd(window: UsageWindow) -> float:
    return max(0, window.limit - window.used) / 1_000_000
