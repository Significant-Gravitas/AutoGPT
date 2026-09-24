import logging
from datetime import datetime
from typing import Annotated, Literal

import stripe
from autogpt_libs.auth import get_user_id
from autogpt_libs.auth.service import frontend_service_claims
from fastapi import APIRouter, Depends, Header, HTTPException, Security
from pydantic import BaseModel, Field

from backend.api.features.billing.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.data.credit import _datafast_metadata, sync_subscription_from_stripe
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import (
    get_subscription_trial,
    has_received_onboarding_credit,
)
from backend.data.subscription_trial_capacity import trial_seat_available
from backend.data.subscription_trial_checkout import (
    TrialUnavailable,
    confirm_trial_checkout,
    create_trial_checkout,
    resolve_trial_price,
)
from backend.data.subscription_trial_config import AcceptedTrialOffer, get_trial_offer
from backend.data.subscription_trial_rejection import TrialRejectionReason
from backend.data.user import get_user_by_id
from backend.util.product_analytics import track_checkout_started
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

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
    rejection_reason: TrialRejectionReason | None = None
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


CLIENT_COUNTRY_SCOPE = "client-country"


async def attested_country(
    token: Annotated[
        str | None, Header(alias="X-Client-Country-Token", include_in_schema=False)
    ] = None,
) -> str | None:
    """The visitor's country, as the frontend proxy vouches for it, or None.

    The backend is reachable directly -- the browser already calls it with
    its own bearer token -- so a plain country header would be whatever the
    caller typed. The proxy instead sends what Vercel's edge geolocated inside
    a short-lived frontend service token, signed with the JWKS key only the
    frontend holds. Anything else -- no token, a forged or expired one, a
    user token -- is no country at all, which the offer's country rule treats
    as unknown and withholds. Hidden from the schema: it is proxy-to-backend
    plumbing, not API surface.
    """
    if not token:
        return None
    claims = await frontend_service_claims(token, CLIENT_COUNTRY_SCOPE)
    country = claims.get("country") if claims else None
    return country if isinstance(country, str) else None


ClientCountry = Annotated[str | None, Depends(attested_country)]


@router.get("")
async def get_trial_status(
    user_id: CurrentUser, country: ClientCountry = None
) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    if trial:
        return TrialStatusResponse(
            eligible=(
                trial.status == "checkout_pending"
                and trial.consumed_at is None
                and (offer := await get_trial_offer(user_id, country=country))
                is not None
                and await trial_seat_available(offer, trial_id=trial.id)
            ),
            offer=TrialOfferResponse.from_offer(trial.offer),
            status=trial.status,
            rejection_reason=trial.rejection_reason,
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
    offer = await get_trial_offer(user_id, country=country)
    if offer is None or not await trial_seat_available(offer):
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


@router.post(
    "",
    responses={
        409: {"description": "Trial offer unavailable"},
        502: {"description": "Stripe checkout unavailable"},
        503: {"description": "Billing return URL not configured"},
    },
)
async def start_trial_checkout(
    body: TrialCheckoutRequest,
    user_id: CurrentUser,
    country: ClientCountry = None,
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
            country=country,
        )
    except TrialUnavailable as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(502, "Unable to start checkout. Please try again.") from exc
    await _track_trial_checkout_started(user_id, surface=body.return_to)
    return TrialCheckoutResponse(url=url)


async def _track_trial_checkout_started(user_id: str, *, surface: str) -> None:
    """Best-effort: the reserved trial names the plan the card is set up for."""
    try:
        trial = await get_subscription_trial(user_id)
    except Exception:
        logger.warning("Could not read the trial for checkout_started", exc_info=True)
        trial = None
    await track_checkout_started(
        user_id=user_id,
        checkout_kind="trial",
        surface=surface,
        subscription_tier=trial.offer.tier if trial else None,
        billing_cycle=trial.offer.billing_cycle if trial else None,
    )


@router.post(
    "/cancel",
    responses={
        409: {"description": "No cancelable trial subscription"},
        502: {"description": "Stripe cancellation temporarily unavailable"},
    },
)
async def cancel_trial(user_id: CurrentUser) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    if (
        trial is None
        or trial.subscription_id is None
        or trial.consumed_at is None
        or trial.converted_at is not None
    ):
        raise HTTPException(409, "No trial subscription is available to cancel")
    try:
        subscription = await stripe_call(
            stripe.Subscription.retrieve_async, trial.subscription_id
        )
        if subscription.customer != trial.customer_id or subscription.status not in (
            "trialing",
            "canceled",
        ):
            raise HTTPException(
                409, "This trial has ended. Manage the plan in billing."
            )
        if subscription.status == "trialing":
            subscription = await stripe_call(
                stripe.Subscription.cancel_async,
                trial.subscription_id,
                invoice_now=False,
                prorate=False,
            )
    except stripe.StripeError as exc:
        raise HTTPException(502, "Unable to cancel your trial. Please retry.") from exc
    await sync_subscription_from_stripe(dict(subscription))
    return await get_trial_status(user_id)


@router.post(
    "/confirm",
    responses={
        409: {"description": "Trial checkout is unavailable or no longer current"},
        502: {"description": "Stripe confirmation temporarily unavailable"},
    },
)
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
