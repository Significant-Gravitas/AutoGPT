"""Admin endpoints for checking and resetting per-user CoPilot rate limits."""

import logging

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import get_usage_status, reset_user_usage
from backend.util.feature_flag import Flag, get_feature_flag_value

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


async def _get_global_limits(user_id: str) -> tuple[int, int]:
    """Resolve global rate limits from LaunchDarkly, falling back to config."""
    daily = await get_feature_flag_value(
        Flag.COPILOT_DAILY_TOKEN_LIMIT.value,
        user_id,
        config.daily_token_limit,
    )
    weekly = await get_feature_flag_value(
        Flag.COPILOT_WEEKLY_TOKEN_LIMIT.value,
        user_id,
        config.weekly_token_limit,
    )
    return int(daily), int(weekly)


@router.get(
    "/rate_limit",
    response_model=UserRateLimitResponse,
    summary="Get User Rate Limit",
)
async def get_user_rate_limit(
    user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> UserRateLimitResponse:
    """Get a user's current usage and effective rate limits."""
    logger.info(f"Admin {admin_user_id} checking rate limit for user {user_id}")

    daily_limit, weekly_limit = await _get_global_limits(user_id)
    usage = await get_usage_status(user_id, daily_limit, weekly_limit)

    return UserRateLimitResponse(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        daily_tokens_used=usage.daily.used,
        weekly_tokens_used=usage.weekly.used,
    )


@router.post(
    "/rate_limit/reset",
    response_model=UserRateLimitResponse,
    summary="Reset User Rate Limit Usage",
)
async def reset_user_rate_limit(
    user_id: str = Body(embed=True),
    admin_user_id: str = Security(get_user_id),
) -> UserRateLimitResponse:
    """Reset a user's daily and weekly usage counters to zero."""
    logger.info(f"Admin {admin_user_id} resetting rate limit for user {user_id}")

    try:
        await reset_user_usage(user_id)
    except Exception as e:
        logger.exception(f"Failed to reset user usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset usage") from e

    daily_limit, weekly_limit = await _get_global_limits(user_id)
    usage = await get_usage_status(user_id, daily_limit, weekly_limit)

    return UserRateLimitResponse(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        daily_tokens_used=usage.daily.used,
        weekly_tokens_used=usage.weekly.used,
    )
