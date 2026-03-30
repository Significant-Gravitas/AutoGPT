"""Admin endpoints for checking and resetting user CoPilot rate limit usage."""

import logging
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    get_global_rate_limits,
    get_usage_status,
    reset_user_usage,
)
from backend.data.user import get_user_by_email, get_user_email_by_id

logger = logging.getLogger(__name__)

config = ChatConfig()

router = APIRouter(
    prefix="/admin",
    tags=["copilot", "admin"],
    dependencies=[Security(requires_admin_user)],
)


class UserRateLimitResponse(BaseModel):
    user_id: str
    user_email: Optional[str] = None
    daily_token_limit: int
    weekly_token_limit: int
    daily_tokens_used: int
    weekly_tokens_used: int


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

    daily_limit, weekly_limit = await get_global_rate_limits(
        resolved_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(resolved_id, daily_limit, weekly_limit)

    return UserRateLimitResponse(
        user_id=resolved_id,
        user_email=resolved_email,
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

    daily_limit, weekly_limit = await get_global_rate_limits(
        user_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(user_id, daily_limit, weekly_limit)

    try:
        resolved_email = await get_user_email_by_id(user_id)
    except Exception:
        logger.warning("Failed to resolve email for user %s", user_id, exc_info=True)
        resolved_email = None

    return UserRateLimitResponse(
        user_id=user_id,
        user_email=resolved_email,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        daily_tokens_used=usage.daily.used,
        weekly_tokens_used=usage.weekly.used,
    )
