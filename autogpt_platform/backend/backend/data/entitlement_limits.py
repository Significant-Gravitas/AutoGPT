"""Otto's usage limits and workspace storage under the entitled tier.

The tier comes from ``backend.data.entitlement``: the user's own plan in a
personal organization, the team organization's plan there. Limits are per
person, so each seat gets the tier's whole allowance; nothing is pooled. The
usage counters are the person's own in every organization. In a team
organization the person's own tier and trial don't apply: the helpers in
``backend.copilot.rate_limit`` run with ``org_plan=True``.
"""

import logging

from backend.copilot import rate_limit
from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    DEFAULT_TIER,
    CoPilotUsageStatus,
    RateLimitExceeded,
    RateLimitUnavailable,
    SubscriptionTier,
)
from backend.data.entitlement import get_user_tier, team_org_snapshot

logger = logging.getLogger(__name__)

_chat_config = ChatConfig()


async def get_global_rate_limits(
    user_id: str,
    config_daily: int,
    config_weekly: int,
    org_id: str | None = None,
) -> tuple[int, int, SubscriptionTier]:
    """``(daily, weekly, tier)`` for work by *user_id* in *org_id*, in
    microdollars, from the tier the entitlement check gives."""
    if not await org_plan_decides(org_id):
        return await rate_limit.get_global_rate_limits(
            user_id, config_daily, config_weekly
        )
    tier = await get_user_tier(user_id, org_id)
    return await rate_limit.rate_limits_for_tier(
        user_id, tier, config_daily, config_weekly
    )


async def check_rate_limit(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
    *,
    org_id: str | None = None,
    skip_daily: bool = False,
    expected_tier: SubscriptionTier | str | None = None,
) -> None:
    """``rate_limit.check_rate_limit`` for work in *org_id*."""
    await rate_limit.check_rate_limit(
        user_id=user_id,
        daily_cost_limit=daily_cost_limit,
        weekly_cost_limit=weekly_cost_limit,
        skip_daily=skip_daily,
        expected_tier=expected_tier,
        org_plan=await org_plan_decides(org_id),
    )


async def get_usage_status(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
    rate_limit_reset_cost: int = 0,
    tier: SubscriptionTier = DEFAULT_TIER,
    fail_closed: bool = False,
    *,
    org_id: str | None = None,
) -> CoPilotUsageStatus:
    """``rate_limit.get_usage_status`` for work in *org_id*."""
    return await rate_limit.get_usage_status(
        user_id=user_id,
        daily_cost_limit=daily_cost_limit,
        weekly_cost_limit=weekly_cost_limit,
        rate_limit_reset_cost=rate_limit_reset_cost,
        tier=tier,
        fail_closed=fail_closed,
        org_plan=await org_plan_decides(org_id),
    )


async def get_remaining_usd_budget(
    user_id: str,
    daily_cost_limit: int,
    weekly_cost_limit: int,
    floor_usd: float = 0.5,
    *,
    expected_tier: SubscriptionTier | str | None = None,
    org_id: str | None = None,
) -> float:
    """``rate_limit.get_remaining_usd_budget`` for work in *org_id*."""
    return await rate_limit.get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily_cost_limit,
        weekly_cost_limit=weekly_cost_limit,
        floor_usd=floor_usd,
        expected_tier=expected_tier,
        org_plan=await org_plan_decides(org_id),
    )


async def get_workspace_storage_limit_bytes(
    user_id: str, org_id: str | None = None
) -> int:
    """The workspace storage cap in bytes for *user_id* working in *org_id*."""
    if not await org_plan_decides(org_id):
        return await rate_limit.get_workspace_storage_limit_bytes(user_id)
    return await rate_limit.storage_limit_bytes_for_tier(
        await get_user_tier(user_id, org_id)
    )


async def build_budget_ctx(
    user_id: str | None,
    default_daily_cost_limit: int,
    default_weekly_cost_limit: int,
    org_id: str | None = None,
) -> str:
    """The inner text of a ``<budget_context>`` block, or ``""``.

    Empty with no user, with no budget left (the paywall refuses first; this
    defends against drift), or when Redis can't say (rather than a misleading
    ``$0.00`` hint; the pre-turn gate already fails closed then).
    """
    if not user_id:
        return ""
    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id, default_daily_cost_limit, default_weekly_cost_limit, org_id
    )
    remaining = await get_remaining_usd_budget(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        # 0.0 is the "no value" sentinel, mapped to "" below.
        floor_usd=0.0,
        expected_tier=tier,
        org_id=org_id,
    )
    if remaining == float("inf") or remaining <= 0.0:
        return ""
    return (
        f"Approximate remaining USD budget for this user: ${remaining:.2f}.\n"
        "Pace your tool use and reasoning depth so the response stays "
        "within this envelope."
    )


async def platform_spend_refusal(
    user_id: str | None, org_id: str | None = None
) -> str | None:
    """Why a call billed to the platform's own key may not run now, or None.

    Turn admission checks the allowance once per turn; a tool call or a dry
    run that pays a provider itself asks again right before each call. Fails
    closed when the allowance can't be read.
    """
    if not user_id:
        return None
    daily, weekly, tier = await get_global_rate_limits(
        user_id,
        _chat_config.daily_cost_limit_microdollars,
        _chat_config.weekly_cost_limit_microdollars,
        org_id,
    )
    if daily == 0 and weekly == 0:
        return "Your plan has no usage allowance for this right now."
    try:
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily,
            weekly_cost_limit=weekly,
            expected_tier=tier,
            org_id=org_id,
        )
    except RateLimitExceeded as e:
        return str(e)
    except RateLimitUnavailable:
        logger.warning(
            f"Usage limits unreadable for user {user_id}; refusing platform spend",
            exc_info=True,
        )
        return "Usage limits can't be checked right now. Retry shortly."
    return None


async def org_plan_decides(org_id: str | None) -> bool:
    """Whether a team organization's plan, not the user's own, sets the limits."""
    return await team_org_snapshot(org_id) is not None
