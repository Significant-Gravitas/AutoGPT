"""Buy an extra week of usage with credits.

Served under ``/api/chat`` next to the usage read-out. An extra week is a
separate allowance the size of the plan's weekly limit: it is used once the
weekly limit has run out, the daily limit doesn't apply to it, and it lasts
30 days from the purchase (``backend.data.usage_extra``). Sold to the plans
that price one (``usage-plan-limits``), from the launch of the billing-clock
limits, and only where credits are on: with credits off a spend is free.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal, NoReturn

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security
from prisma.enums import SubscriptionTier
from pydantic import BaseModel, Field

from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    CoPilotUsagePublic,
    RateLimitUnavailable,
    get_global_rate_limits,
    get_usage_status,
)
from backend.copilot.trial_reads import forget_usage_state
from backend.copilot.usage_limits import plan_limits_for
from backend.copilot.usage_mode import billing_clock_live, now_seconds
from backend.data.usage_extra import buy_extra_week
from backend.util import posthog_client
from backend.util.exceptions import InsufficientBalanceError
from backend.util.posthog_events import PostHogEvent
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])
config = ChatConfig()
settings = Settings()

RefusalReason = Literal["not_live", "not_eligible", "insufficient_credits"]


class ExtraWeekPurchaseRequest(BaseModel):
    purchase_key: str = Field(
        min_length=8,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="A key the client makes up for this purchase (a UUID). "
        "Sending it again returns the same purchase and charges nothing more.",
    )


class ExtraWeekPurchaseResponse(BaseModel):
    credits_charged: int = Field(
        description="Credits taken by this request; 0 when the key had "
        "already bought it."
    )
    remaining_balance: int = Field(description="Credit balance now (in cents).")
    expires_at: datetime = Field(description="When this extra week expires.")
    usage: CoPilotUsagePublic


@router.post(
    "/usage/extra-week",
    responses={
        402: {"description": "Not enough credits; nothing was charged"},
        409: {"description": "This account can't buy an extra week"},
        503: {"description": "Usage limits can't be read right now"},
    },
)
async def buy_extra_week_of_usage(
    request: ExtraWeekPurchaseRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ExtraWeekPurchaseResponse:
    """Buy one extra week of usage from the credit balance."""
    now = now_seconds()
    if not settings.config.enable_credit or not await billing_clock_live(now):
        _refuse(user_id, "not_live")
    try:
        daily, weekly, tier = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        plan = await plan_limits_for(tier.value)
        if plan is None or plan.extra_week_credits <= 0 or weekly <= 0:
            _refuse(user_id, "not_eligible", tier=tier.value)
        try:
            purchase = await buy_extra_week(
                user_id,
                request.purchase_key,
                tier=SubscriptionTier(tier.value),
                amount=weekly,
                price_credits=plan.extra_week_credits,
                now=now,
            )
        except InsufficientBalanceError as error:
            _track_refusal(user_id, "insufficient_credits", tier.value)
            raise HTTPException(
                status_code=402,
                detail={
                    "message": "Not enough credits for an extra week.",
                    "needed": plan.extra_week_credits,
                    "balance": int(error.balance),
                },
            ) from error
        if purchase.charged:
            posthog_client.capture(
                user_id,
                PostHogEvent.USAGE_EXTRA_WEEK_PURCHASED,
                {"tier": tier.value, "price_credits": plan.extra_week_credits},
            )
        # The purchase changed the usage state this request may have shared.
        forget_usage_state(user_id)
        status = await get_usage_status(user_id, daily, weekly, tier=tier)
    except RateLimitUnavailable as error:
        raise HTTPException(
            status_code=503,
            detail="Usage limits can't be read right now. Please try again shortly.",
            headers={"Retry-After": "30"},
        ) from error
    return ExtraWeekPurchaseResponse(
        credits_charged=plan.extra_week_credits if purchase.charged else 0,
        remaining_balance=purchase.balance,
        expires_at=datetime.fromtimestamp(purchase.extra_week.expires_at, UTC),
        usage=CoPilotUsagePublic.from_status(status),
    )


def _refuse(user_id: str, reason: RefusalReason, tier: str | None = None) -> NoReturn:
    _track_refusal(user_id, reason, tier)
    raise HTTPException(
        status_code=409, detail="An extra week isn't available for this account."
    )


def _track_refusal(user_id: str, reason: RefusalReason, tier: str | None) -> None:
    posthog_client.capture(
        user_id,
        PostHogEvent.USAGE_EXTRA_WEEK_REFUSED,
        {"reason": reason, "tier": tier},
    )
