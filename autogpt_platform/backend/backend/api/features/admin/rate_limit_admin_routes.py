"""Admin endpoints for checking and resetting user CoPilot rate limit usage."""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    SubscriptionTier,
    get_global_rate_limits,
    get_stripe_sweep_revert_warning,
    get_tier_multipliers,
    get_usage_status,
    get_user_tier,
    reset_user_usage,
    set_user_tier,
)
from backend.data.usage_reset import ResetPurchase, list_reset_purchases
from backend.data.user import get_user_by_email, get_user_email_by_id, search_users

logger = logging.getLogger(__name__)

config = ChatConfig()

router = APIRouter(
    prefix="/admin",
    tags=["copilot", "admin"],
    dependencies=[Security(requires_admin_user)],
)


class BillingClockUsage(BaseModel):
    """What a user's limits look like on their billing clock (SECRT-3441)."""

    # stripe: their Stripe billing period. held: that period ended and Stripe
    # hasn't confirmed the next. account: monthly from the account's creation
    # (no Stripe period). trial: the trial's own dates.
    source: str
    daily_resets_at: datetime
    weekly_resets_at: datetime
    # The whole weekly limit; ``weekly_cost_limit_microdollars`` is its pool,
    # the part above the days' reserves.
    weekly_limit_microdollars: int
    reserve_today_microdollars: int
    reserve_used_microdollars: int
    # The limit that stops a turn starting now, if one does.
    blocked_by: Optional[str] = None
    # A paid reset is in force: no daily limit until the weekly reset, and
    # the usage shown is the usage since it.
    reset_active: bool = False
    # The latest paid-reset purchases, newest first, with how each was paid
    # (and its Stripe ids, for a refund). A pending one was never paid.
    reset_purchases: list[ResetPurchase] = []


class UserRateLimitResponse(BaseModel):
    user_id: str
    user_email: Optional[str] = None
    daily_cost_limit_microdollars: int
    weekly_cost_limit_microdollars: int
    daily_cost_used_microdollars: int
    weekly_cost_used_microdollars: int
    tier: SubscriptionTier
    # Full per-tier multiplier map incl. admin-managed tiers like ENTERPRISE,
    # unlike the price-filtered /credits/subscription map.
    tier_multipliers: dict[str, float] = {}
    # Set once limits run on the billing clock; the daily and weekly figures
    # above are then this clock's windows.
    billing_clock: Optional[BillingClockUsage] = None


class UserTierResponse(BaseModel):
    user_id: str
    tier: SubscriptionTier
    # Set by POST /rate_limit/tier when the grant will not stick: the user has
    # a Stripe customer without an active subscription, so the Stripe
    # reconciliation sweep will revert the tier to NO_TIER on its next run.
    warning: Optional[str] = None


class SetUserTierRequest(BaseModel):
    user_id: str
    tier: SubscriptionTier


async def _resolve_user_id(
    user_id: Optional[str], email: Optional[str]
) -> tuple[str, Optional[str]]:
    """Resolve a user_id and email from the provided parameters.

    Returns (user_id, email). Accepts either user_id or email; at least one
    must be provided.  When both are provided, ``email`` takes precedence.
    """
    if email:
        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=404, detail="No user found with the provided email."
            )
        return user.id, email

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or email query parameter is required.",
        )

    # We have a user_id; try to look up their email for display purposes.
    # This is non-critical -- a failure should not block the response.
    try:
        resolved_email = await get_user_email_by_id(user_id)
    except Exception:
        logger.warning("Failed to resolve email for user %s", user_id, exc_info=True)
        resolved_email = None
    return user_id, resolved_email


@router.get(
    "/rate_limit",
    response_model=UserRateLimitResponse,
    summary="Get User Rate Limit",
)
async def get_user_rate_limit(
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    admin_user_id: str = Security(get_user_id),
) -> UserRateLimitResponse:
    """Get a user's current usage and effective rate limits. Admin-only.

    Accepts either ``user_id`` or ``email`` as a query parameter.
    When ``email`` is provided the user is looked up by email first.
    """
    resolved_id, resolved_email = await _resolve_user_id(user_id, email)

    logger.info("Admin %s checking rate limit for user %s", admin_user_id, resolved_id)

    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        resolved_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    usage = await get_usage_status(resolved_id, daily_limit, weekly_limit, tier=tier)
    multipliers = await get_tier_multipliers()

    return UserRateLimitResponse(
        user_id=resolved_id,
        user_email=resolved_email,
        daily_cost_limit_microdollars=daily_limit,
        weekly_cost_limit_microdollars=weekly_limit,
        daily_cost_used_microdollars=usage.daily.used,
        weekly_cost_used_microdollars=usage.weekly.used,
        tier=tier,
        tier_multipliers=multipliers,
        billing_clock=await _billing_clock_usage(resolved_id, usage),
    )


async def _billing_clock_usage(
    user_id: str, usage: CoPilotUsageStatus
) -> BillingClockUsage | None:
    if not usage.on_billing_clock:
        return None
    return BillingClockUsage(
        source=usage.clock_source or "",
        daily_resets_at=usage.daily.resets_at,
        weekly_resets_at=usage.weekly.resets_at,
        weekly_limit_microdollars=(
            usage.weekly.limit if usage.weekly_limit is None else usage.weekly_limit
        ),
        reserve_today_microdollars=usage.reserve.limit if usage.reserve else 0,
        reserve_used_microdollars=usage.reserve.used if usage.reserve else 0,
        blocked_by=usage.blocked_by,
        reset_active=usage.reset_active,
        reset_purchases=await list_reset_purchases(user_id),
    )


@router.post(
    "/rate_limit/reset",
    response_model=UserRateLimitResponse,
    summary="Reset User Rate Limit Usage",
)
async def reset_user_rate_limit(
    user_id: str = Body(embed=True),
    reset_weekly: bool = Body(False, embed=True),
    admin_user_id: str = Security(get_user_id),
) -> UserRateLimitResponse:
    """Reset a user's daily usage counter (and optionally weekly). Admin-only."""
    logger.info(
        "Admin %s resetting rate limit for user %s (reset_weekly=%s)",
        admin_user_id,
        user_id,
        reset_weekly,
    )

    try:
        await reset_user_usage(user_id, reset_weekly=reset_weekly)
    except Exception as e:
        logger.exception("Failed to reset user usage")
        raise HTTPException(status_code=500, detail="Failed to reset usage") from e

    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    usage = await get_usage_status(user_id, daily_limit, weekly_limit, tier=tier)
    multipliers = await get_tier_multipliers()

    try:
        resolved_email = await get_user_email_by_id(user_id)
    except Exception:
        logger.warning("Failed to resolve email for user %s", user_id, exc_info=True)
        resolved_email = None

    return UserRateLimitResponse(
        user_id=user_id,
        user_email=resolved_email,
        daily_cost_limit_microdollars=daily_limit,
        weekly_cost_limit_microdollars=weekly_limit,
        daily_cost_used_microdollars=usage.daily.used,
        weekly_cost_used_microdollars=usage.weekly.used,
        tier=tier,
        tier_multipliers=multipliers,
        billing_clock=await _billing_clock_usage(user_id, usage),
    )


@router.get(
    "/rate_limit/tier",
    response_model=UserTierResponse,
    summary="Get User Rate Limit Tier",
)
async def get_user_rate_limit_tier(
    user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> UserTierResponse:
    """Get a user's current rate-limit tier. Admin-only.

    Returns 404 if the user does not exist in the database.
    """
    logger.info("Admin %s checking tier for user %s", admin_user_id, user_id)

    resolved_email = await get_user_email_by_id(user_id)
    if resolved_email is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    tier = await get_user_tier(user_id)
    return UserTierResponse(user_id=user_id, tier=tier)


@router.post(
    "/rate_limit/tier",
    response_model=UserTierResponse,
    summary="Set User Rate Limit Tier",
)
async def set_user_rate_limit_tier(
    request: SetUserTierRequest,
    admin_user_id: str = Security(get_user_id),
) -> UserTierResponse:
    """Set a user's rate-limit tier. Admin-only.

    Returns 404 if the user does not exist in the database.
    """
    try:
        resolved_email = await get_user_email_by_id(request.user_id)
    except Exception:
        logger.warning(
            "Failed to resolve email for user %s",
            request.user_id,
            exc_info=True,
        )
        resolved_email = None

    if resolved_email is None:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")

    old_tier = await get_user_tier(request.user_id)
    logger.info(
        "Admin %s changing tier for user %s (%s): %s -> %s",
        admin_user_id,
        request.user_id,
        resolved_email,
        old_tier.value,
        request.tier.value,
    )
    try:
        await set_user_tier(request.user_id, request.tier)
    except Exception as e:
        logger.exception("Failed to set user tier")
        raise HTTPException(status_code=500, detail="Failed to set tier") from e

    warning = await _sweep_revert_warning(request.user_id, request.tier)
    return UserTierResponse(user_id=request.user_id, tier=request.tier, warning=warning)


async def _sweep_revert_warning(user_id: str, tier: SubscriptionTier) -> Optional[str]:
    """Best-effort: the tier is already written, so a Stripe or DB hiccup here
    must not turn the response into a 500 or hold the admin for long."""
    try:
        return await asyncio.wait_for(
            get_stripe_sweep_revert_warning(user_id, tier), timeout=5.0
        )
    except Exception:
        logger.warning(
            "Could not check whether the Stripe sweep will revert the tier grant"
            " for user %s",
            user_id,
            exc_info=True,
        )
        return None


class UserSearchResult(BaseModel):
    user_id: str
    user_email: Optional[str] = None


@router.get(
    "/rate_limit/search_users",
    response_model=list[UserSearchResult],
    summary="Search Users by Name or Email",
)
async def admin_search_users(
    query: str,
    limit: int = 20,
    admin_user_id: str = Security(get_user_id),
) -> list[UserSearchResult]:
    """Search users by partial email or name. Admin-only.

    Queries the User table directly — returns results even for users
    without credit transaction history.
    """
    if len(query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Search query must be at least 3 characters.",
        )
    logger.info("Admin %s searching users with query=%r", admin_user_id, query)
    results = await search_users(query, limit=max(1, min(limit, 50)))
    return [UserSearchResult(user_id=uid, user_email=email) for uid, email in results]
