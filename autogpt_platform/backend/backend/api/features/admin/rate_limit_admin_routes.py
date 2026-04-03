"""Admin endpoints for checking and resetting user CoPilot rate limit usage."""

import logging
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Security
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    SubscriptionTier,
    get_global_rate_limits,
    get_usage_status,
    get_user_tier,
    reset_user_usage,
    set_user_tier,
)
from backend.data.user import get_user_by_email, get_user_email_by_id, search_users

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
    tier: SubscriptionTier


class UserTierResponse(BaseModel):
    user_id: str
    tier: SubscriptionTier


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
        resolved_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(resolved_id, daily_limit, weekly_limit, tier=tier)

    return UserRateLimitResponse(
        user_id=resolved_id,
        user_email=resolved_email,
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
        user_id, config.daily_token_limit, config.weekly_token_limit
    )
    usage = await get_usage_status(user_id, daily_limit, weekly_limit, tier=tier)

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

    return UserTierResponse(user_id=request.user_id, tier=request.tier)


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
