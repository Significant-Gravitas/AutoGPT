"""Buy a usage reset, with credits or by card.

Served under ``/api/chat`` next to the usage read-out. A reset starts the
customer's weekly limit over and lifts the daily limit until the weekly
reset, whose time does not move (``backend.data.usage_reset``). It is sold:

- from the launch of the billing-clock limits, where credits are on;
- to the plans that price one (``usage-plan-limits``);
- only while a limit stops the customer (it replaces what is left of the
  week), and not in the last minutes before the weekly reset.

It is paid from the credit balance when that covers the whole price, and
otherwise by card for the whole price: never part of each. A card payment
grants nothing until Stripe confirms it.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal, NoReturn

import stripe
from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Security
from prisma.enums import SubscriptionTier as PrismaTier
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, Field

from backend.api.features.chat.usage_reset_models import (
    UsageResetConfirmRequest,
    UsageResetConfirmResponse,
    UsageResetRequest,
    UsageResetResponse,
)
from backend.copilot.config import ChatConfig
from backend.copilot.rate_limit import (
    CoPilotUsagePublic,
    CoPilotUsageStatus,
    RateLimitUnavailable,
    SubscriptionTier,
    enforce_payment_paywall,
    get_global_rate_limits,
    get_usage_status,
)
from backend.copilot.trial_reads import forget_usage_state
from backend.copilot.usage_mode import billing_clock_live, now_seconds
from backend.data.credit import get_user_credit_model
from backend.data.usage_reset import (
    CreditReset,
    ResetPurchase,
    ResetWindowChanged,
    buy_reset_with_credits,
    find_reset_purchase,
)
from backend.data.usage_reset_card import (
    CheckoutUnsettled,
    fulfill_card_reset,
    start_card_reset,
)
from backend.util import posthog_client
from backend.util.exceptions import InsufficientBalanceError
from backend.util.posthog_events import PostHogEvent
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])
config = ChatConfig()
settings = Settings()

RefusalReason = Literal[
    "not_live",
    "not_eligible",
    "not_limited",
    "resets_soon",
    "price_changed",
    "payment_changed",
    "window_changed",
]
_REFUSALS: dict[RefusalReason, str] = {
    "not_live": "A usage reset isn't available for this account.",
    "not_eligible": "A usage reset isn't available for this account.",
    "not_limited": "You haven't reached a usage limit, so there is nothing to reset.",
    "resets_soon": "Your limits reset shortly anyway, so a reset can't be bought now.",
    "price_changed": "price_changed",
    "payment_changed": "payment_changed",
    "window_changed": "Your usage window has just changed. Check your usage "
    "and try again.",
}


_RESPONSES: dict[int | str, dict[str, str]] = {
    409: {
        "description": "A reset can't be bought now (no limit is reached, the "
        "weekly reset is minutes away, the plan has none), or its price is no "
        "longer the one shown; nothing was charged"
    },
    503: {
        "description": "Usage limits can't be read, or the card checkout "
        "can't be opened, right now; nothing was charged"
    },
}


@router.post(
    "/usage/paid-reset",
    # A reset restarts a plan's limits: without a plan there is nothing to.
    dependencies=[Depends(enforce_payment_paywall)],
    responses=_RESPONSES,
)
async def buy_usage_reset(
    request: UsageResetRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> UsageResetResponse:
    """Buy a usage reset: with credits when the balance covers it, otherwise
    by card through the checkout this returns."""
    now = now_seconds()
    if not settings.config.enable_credit or not await billing_clock_live(now):
        _refuse(user_id, "not_live")
    earlier = await find_reset_purchase(user_id, request.purchase_key)
    try:
        daily, weekly, tier = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        if earlier is not None and earlier.status == "granted":
            usage = await _usage_now(user_id, daily, weekly, tier)
            return _granted(earlier, 0, usage)
        status = await get_usage_status(
            user_id, daily, weekly, tier=tier, fail_closed=True
        )
    except RateLimitUnavailable as error:
        raise _unavailable() from error
    if earlier is None and status.reset_refusal is not None:
        # This key may have bought its reset in another request a moment
        # ago, which is why nothing is limited now: answer with it.
        bought_meanwhile = await find_reset_purchase(user_id, request.purchase_key)
        if bought_meanwhile is not None and bought_meanwhile.status == "granted":
            usage = await _usage_now(user_id, daily, weekly, tier)
            return _granted(bought_meanwhile, 0, usage)
    _refuse_unless_sold(user_id, request, status)
    if earlier is None and await _balance_covers(user_id, status.reset_price):
        if request.pay_with == "card":
            # They confirmed a card payment; the balance covers it by now.
            # Credits aren't taken on a confirmation that named the card.
            _refuse(user_id, "payment_changed", status.tier.value)
        bought = await _pay_with_credits(user_id, request, status, now)
        if bought is not None:
            return _granted(
                bought.purchase,
                status.reset_price if bought.charged else 0,
                await _usage_now(user_id, daily, weekly, tier),
            )
        # The credits went elsewhere meanwhile, perhaps on another reset of
        # this window: check again that there is still one to sell.
        status = await _status_again(user_id, daily, weekly, tier)
        _refuse_unless_sold(user_id, request, status)
    return await _pay_by_card(user_id, request, status, now, (daily, weekly))


async def _status_again(
    user_id: str, daily: int, weekly: int, tier: SubscriptionTier
) -> CoPilotUsageStatus:
    forget_usage_state(user_id)
    try:
        return await get_usage_status(
            user_id, daily, weekly, tier=tier, fail_closed=True
        )
    except RateLimitUnavailable as error:
        raise _unavailable() from error


@router.post(
    "/usage/paid-reset/confirm",
    dependencies=[Depends(enforce_payment_paywall)],
    responses={503: _RESPONSES[503]},
)
async def confirm_usage_reset(
    request: UsageResetConfirmRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> UsageResetConfirmResponse:
    """Called when the customer returns from paying by card: grants the reset
    if Stripe says the payment went through and its webhook hasn't yet."""
    purchase = await settle_card_reset(request.session_id, user_id=user_id)
    if purchase is None:
        return UsageResetConfirmResponse(status="pending")
    try:
        daily, weekly, tier = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
    except RateLimitUnavailable:
        return UsageResetConfirmResponse(status="granted", resets_at=_ends_at(purchase))
    return UsageResetConfirmResponse(
        status="granted",
        resets_at=_ends_at(purchase),
        usage=await _usage_now(user_id, daily, weekly, tier),
    )


async def settle_card_reset(
    session_id: str, *, user_id: str | None = None
) -> ResetPurchase | None:
    """Grant the reset a Checkout session paid for, once, whoever asks first:
    Stripe's webhook (for a completed session that says it is a reset's) or
    the customer returning (``user_id``: only their own).
    Returns the granted purchase, or None when there is none to grant."""
    grant = await fulfill_card_reset(session_id, now_seconds(), user_id=user_id)
    if grant is None:
        return None
    if grant.granted_now:
        purchase = grant.purchase
        posthog_client.capture(
            purchase.user_id,
            PostHogEvent.USAGE_RESET_PURCHASED,
            {
                "tier": purchase.tier,
                "price_credits": purchase.price_credits,
                "paid_with": "card",
            },
        )
    return grant.purchase


def _refuse_unless_sold(
    user_id: str, request: UsageResetRequest, status: CoPilotUsageStatus
) -> None:
    """Refuse a purchase the customer couldn't use or didn't agree to: the
    account's limits must run on its billing clock, its plan must sell a
    reset, a limit must be stopping it now with time left to use a reset,
    and the price charged must be the one shown."""
    tier = status.tier.value
    if not status.on_billing_clock:
        _refuse(user_id, "not_live", tier)
    if status.reset_price <= 0 or status.week_start is None:
        _refuse(user_id, "not_eligible", tier)
    if status.reset_refusal is not None:
        _refuse(user_id, status.reset_refusal, tier)
    if request.price_credits != status.reset_price:
        _refuse(user_id, "price_changed", tier)


async def _balance_covers(user_id: str, price_credits: int) -> bool:
    credits = await get_user_credit_model(user_id)
    return await credits.get_credits(user_id) >= price_credits


async def _pay_with_credits(
    user_id: str, request: UsageResetRequest, status: CoPilotUsageStatus, now: int
) -> CreditReset | None:
    """The granted purchase; None when the balance turned out not to cover
    it after all (it was spent meanwhile), or this key started a card
    payment at the same moment: both leave the card."""
    if status.week_start is None:
        _refuse(user_id, "not_eligible", status.tier.value)
    try:
        bought = await buy_reset_with_credits(
            user_id,
            request.purchase_key,
            tier=PrismaTier(status.tier.value),
            price_credits=status.reset_price,
            week_start=status.week_start,
            week_end=int(status.weekly.resets_at.timestamp()),
            sequence=status.reset_count + 1,
            now=now,
        )
    except InsufficientBalanceError:
        return None
    except ResetWindowChanged:
        _refuse(user_id, "window_changed", status.tier.value)
    except UniqueViolationError:
        if await find_reset_purchase(user_id, request.purchase_key) is not None:
            return None
        # Another purchase reset this window a moment ago: nothing to reset.
        _refuse(user_id, "not_limited", status.tier.value)
    if bought.charged:
        _track_purchase(user_id, status, "credits")
    return bought


async def _pay_by_card(
    user_id: str,
    request: UsageResetRequest,
    status: CoPilotUsageStatus,
    now: int,
    limits: tuple[int, int],
) -> UsageResetResponse:
    try:
        checkout = await start_card_reset(
            user_id,
            request.purchase_key,
            tier=PrismaTier(status.tier.value),
            price_credits=status.reset_price,
            now=now,
        )
    except (stripe.StripeError, CheckoutUnsettled) as error:
        # Nothing was charged by this request. The same key can be sent
        # again: it resumes this purchase rather than starting another.
        logger.warning("Could not open a usage reset checkout: %s", type(error))
        raise HTTPException(
            status_code=503,
            detail="Checkout can't be opened right now. Please try again shortly.",
            headers={"Retry-After": "30"},
        ) from error
    if checkout.purchase.status == "granted":
        # Its earlier checkout was paid meanwhile.
        usage = await _usage_now(user_id, limits[0], limits[1], status.tier)
        return _granted(checkout.purchase, 0, usage)
    if not checkout.url:
        raise HTTPException(
            status_code=503, detail="Checkout can't be opened right now."
        )
    posthog_client.capture(
        user_id,
        PostHogEvent.USAGE_RESET_CHECKOUT_STARTED,
        {"tier": status.tier.value, "price_credits": status.reset_price},
    )
    return UsageResetResponse(
        status="payment_required", paid_with="card", checkout_url=checkout.url
    )


def _granted(
    purchase: ResetPurchase, credits_charged: int, usage: CoPilotUsagePublic | None
) -> UsageResetResponse:
    return UsageResetResponse(
        status="granted",
        paid_with=purchase.paid_with,
        credits_charged=credits_charged,
        resets_at=_ends_at(purchase),
        usage=usage,
    )


def _ends_at(purchase: ResetPurchase) -> datetime | None:
    if purchase.week_end is None:
        return None
    return datetime.fromtimestamp(purchase.week_end, UTC)


async def _usage_now(
    user_id: str, daily: int, weekly: int, tier: SubscriptionTier
) -> CoPilotUsagePublic | None:
    """The usage read-out with the reset in it, when it can be read. The
    purchase is already made: failing to read it back must not turn a paid
    purchase into an error."""
    # The purchase changed the usage state this request may have shared.
    forget_usage_state(user_id)
    try:
        status = await get_usage_status(user_id, daily, weekly, tier=tier)
    except RateLimitUnavailable:
        logger.warning("Could not read usage back after a usage reset")
        return None
    return CoPilotUsagePublic.from_status(status)


def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Usage limits can't be read right now. Please try again shortly.",
        headers={"Retry-After": "30"},
    )


def _track_purchase(
    user_id: str, status: CoPilotUsageStatus, paid_with: Literal["credits", "card"]
) -> None:
    posthog_client.capture(
        user_id,
        PostHogEvent.USAGE_RESET_PURCHASED,
        {
            "tier": status.tier.value,
            "price_credits": status.reset_price,
            "paid_with": paid_with,
        },
    )


def _refuse(user_id: str, reason: RefusalReason, tier: str | None = None) -> NoReturn:
    posthog_client.capture(
        user_id, PostHogEvent.USAGE_RESET_REFUSED, {"reason": reason, "tier": tier}
    )
    raise HTTPException(status_code=409, detail=_REFUSALS[reason])
