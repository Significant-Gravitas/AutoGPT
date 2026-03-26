"""Admin endpoints for checking and resetting user CoPilot rate limit usage."""

import logging

import prisma.errors
from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    RateLimitTier,
    get_global_rate_limits,
    get_usage_status,
    get_user_tier,
    reset_user_usage,
    set_user_tier,
)

logger = logging.getLogger(__name__)

config = ChatConfig()

router = APIRouter(
    prefix="/admin",
    tags=["copilot", "admin"],
    dependencies=[Security(requires_admin_user)],
)


class UserRateLimitResponse(BaseModel):
    user_id: str
    daily_token_limit: int
    weekly_token_limit: int
    daily_tokens_used: int
    weekly_tokens_used: int
    tier: RateLimitTier


class UserTierResponse(BaseModel):
    user_id: str
    tier: RateLimitTier


class SetUserTierRequest(BaseModel):
    user_id: str
    tier: RateLimitTier


@router.get(
    "/rate_limit",
    response_model=UserRateLimitResponse,
    summary="Get User Rate Limit",
)
async def get_user_rate_limit(
    user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> UserRateLimitResponse:
    """Get a user's current usage and effective rate limits. Admin-only."""
    logger.info(f"Admin {admin_user_id} checking rate limit for user {user_id}")

    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(user_id, daily_limit, weekly_limit)

    return UserRateLimitResponse(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        daily_tokens_used=usage.daily.used,
        weekly_tokens_used=usage.weekly.used,
        tier=tier,
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
        f"Admin {admin_user_id} resetting rate limit for user {user_id} "
        f"(reset_weekly={reset_weekly})"
    )

    try:
        await reset_user_usage(user_id, reset_weekly=reset_weekly)
    except Exception as e:
        logger.exception("Failed to reset user usage")
        raise HTTPException(status_code=500, detail="Failed to reset usage") from e

    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(user_id, daily_limit, weekly_limit)

    return UserRateLimitResponse(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        daily_tokens_used=usage.daily.used,
        weekly_tokens_used=usage.weekly.used,
        tier=tier,
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
    """Get a user's current rate-limit tier. Admin-only."""
    logger.info(f"Admin {admin_user_id} checking tier for user {user_id}")
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
    """Set a user's rate-limit tier. Admin-only."""
    logger.info(
        f"Admin {admin_user_id} setting tier for user {request.user_id} "
        f"to {request.tier.value}"
    )
    try:
        await set_user_tier(request.user_id, request.tier)
    except prisma.errors.RecordNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=f"User {request.user_id} not found"
        ) from e
    except Exception as e:
        logger.exception("Failed to set user tier")
        raise HTTPException(status_code=500, detail="Failed to set tier") from e

    return UserTierResponse(user_id=request.user_id, tier=request.tier)
