from datetime import datetime
from typing import Annotated, Literal

import stripe
from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Depends, Header, HTTPException, Security
from pydantic import BaseModel, Field

from backend.api.features.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.data.credit import _datafast_metadata, sync_subscription_from_stripe
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import (
    get_subscription_trial,
    has_received_onboarding_credit,
)
from backend.data.subscription_trial_checkout import (
    TrialUnavailable,
    confirm_trial_checkout,
    create_trial_checkout,
    resolve_trial_price,
)
from backend.data.subscription_trial_config import AcceptedTrialOffer, get_trial_offer
from backend.data.user import get_user_by_id
from backend.util.settings import Settings

router = APIRouter(
    prefix="/credits/trial",
    tags=["trials"],
    dependencies=[Depends(enforce_subscription_status_rate_limit)],
)
CurrentUser = Annotated[str, Security(get_user_id)]


class TrialOfferResponse(BaseModel):
    token: str
    version: str
    duration_days: int
    tier: Literal["BASIC", "PRO", "MAX", "BUSINESS"]
    billing_cycle: Literal["monthly", "yearly"]
    unit_amount: int
    currency: str
    onboarding_credit_amount: int

    @classmethod
    def from_offer(cls, offer: AcceptedTrialOffer) -> "TrialOfferResponse":
        return cls(**offer.model_dump(), token=offer.token)


class TrialStatusResponse(BaseModel):
    eligible: bool = False
    offer: TrialOfferResponse | None = None
    status: str | None = None
    ends_at: datetime | None = None
    cancel_at_period_end: bool = False
    allowance_used_percent: float | None = None
    active: bool = False
    converted: bool = False
    onboarding_credits_previously_received: bool = False


class TrialCheckoutRequest(BaseModel):
    offer_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    return_to: Literal["onboarding", "billing"] = "billing"


class TrialCheckoutResponse(BaseModel):
    url: str


@router.get("")
async def get_trial_status(user_id: CurrentUser) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    if trial:
        return TrialStatusResponse(
            eligible=(
                trial.status == "checkout_pending"
                and trial.consumed_at is None
                and await get_trial_offer(user_id) is not None
            ),
            offer=TrialOfferResponse.from_offer(trial.offer),
            status=trial.status,
            ends_at=trial.ends_at,
            cancel_at_period_end=trial.cancel_at_period_end,
            active=trial.active,
            converted=trial.converted_at is not None,
            onboarding_credits_previously_received=await has_received_onboarding_credit(
                user_id
            ),
            allowance_used_percent=min(
                100, 100 * trial.cost_microdollars / trial.offer.total_cost_limit
            ),
        )
    offer = await get_trial_offer(user_id)
    if offer is None:
        return TrialStatusResponse()
    user = await get_user_by_id(user_id)
    has_history = False
    if user.stripe_customer_id:
        subscriptions = await stripe_call(
            stripe.Subscription.list_async,
            customer=user.stripe_customer_id,
            status="all",
            limit=1,
        )
        has_history = bool(subscriptions.data)
    if not offer.is_eligible(
        created_at=user.created_at,
        current_tier=user.subscription_tier.value,
        has_subscription_history=has_history,
    ):
        return TrialStatusResponse()
    try:
        accepted = await resolve_trial_price(offer)
    except TrialUnavailable:
        return TrialStatusResponse()
    return TrialStatusResponse(
        eligible=True,
        offer=TrialOfferResponse.from_offer(accepted),
        onboarding_credits_previously_received=await has_received_onboarding_credit(
            user_id
        ),
    )


@router.post("")
async def start_trial_checkout(
    body: TrialCheckoutRequest,
    user_id: CurrentUser,
    x_datafast_visitor_id: Annotated[str | None, Header()] = None,
    x_datafast_session_id: Annotated[str | None, Header()] = None,
) -> TrialCheckoutResponse:
    config = Settings().config
    base = config.frontend_base_url or config.platform_base_url
    if not base:
        raise HTTPException(503, "The billing return URL is not configured")
    billing = f"{base.rstrip('/')}/settings/billing"
    destination = (
        f"{base.rstrip('/')}/onboarding" if body.return_to == "onboarding" else billing
    )
    try:
        url = await create_trial_checkout(
            user_id=user_id,
            offer_token=body.offer_token,
            success_url=f"{destination}?trial=success",
            cancel_url=f"{destination}?trial=cancelled",
            metadata=_datafast_metadata(x_datafast_visitor_id, x_datafast_session_id),
        )
    except TrialUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(502, "Unable to start checkout. Please try again.") from exc
    return TrialCheckoutResponse(url=url)


@router.post("/cancel")
async def cancel_trial(user_id: CurrentUser) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    if trial is None or trial.subscription_id is None or trial.consumed_at is None:
        raise HTTPException(409, "No trial subscription is available to cancel")
    subscription = await stripe_call(
        stripe.Subscription.retrieve_async, trial.subscription_id
    )
    if subscription.customer != trial.customer_id or subscription.status != "trialing":
        raise HTTPException(409, "This trial has ended. Manage the plan in billing.")
    subscription = await stripe_call(
        stripe.Subscription.modify_async,
        trial.subscription_id,
        cancel_at_period_end=True,
    )
    await sync_subscription_from_stripe(dict(subscription))
    return await get_trial_status(user_id)


@router.post("/confirm")
async def confirm_trial(user_id: CurrentUser) -> TrialStatusResponse:
    try:
        await confirm_trial_checkout(user_id)
    except TrialUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(
            502, "Unable to confirm the trial yet. Please retry."
        ) from exc
    return await get_trial_status(user_id)
