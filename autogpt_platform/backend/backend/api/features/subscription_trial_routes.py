import logging
from typing import Annotated, Awaitable

import stripe
from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, Depends, Header, HTTPException, Security

from backend.api.features.billing.client_country import (  # noqa: F401 -- re-exported
    CLIENT_COUNTRY_SCOPE,
    ClientCountry,
    attested_country,
)
from backend.api.features.billing.credits_rate_limit import (
    enforce_subscription_status_rate_limit,
)
from backend.api.features.subscription_trial_models import (
    TrialCancelRequest,
    TrialCheckoutRequest,
    TrialCheckoutResponse,
    TrialOfferResponse,
    TrialStatusResponse,
)
from backend.data.checkout_audience import schedule_checkout_opened
from backend.data.credit import _datafast_metadata, sync_subscription_from_stripe
from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial import (
    get_subscription_trial,
    has_received_onboarding_credit,
)
from backend.data.subscription_trial_cancel import (
    TrialChangeRefused,
    resume_trial_subscription,
    schedule_trial_cancellation,
)
from backend.data.subscription_trial_capacity import trial_seat_available
from backend.data.subscription_trial_checkout import (
    TrialUnavailable,
    confirm_trial_checkout,
    create_trial_checkout,
    resolve_trial_price,
)
from backend.data.subscription_trial_config import get_trial_offer
from backend.data.user import get_user_by_id
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.product_analytics import track_checkout_started
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/credits/trial",
    tags=["trials"],
    dependencies=[Depends(enforce_subscription_status_rate_limit)],
)
CurrentUser = Annotated[str, Security(get_user_id)]
CANCEL_RETRY = "Unable to cancel your trial. Please retry."


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
            cancel_keeps_access=not await _cancel_flag_is_off(user_id),
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
        cancel_keeps_access=not await _cancel_flag_is_off(user_id),
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
    schedule_checkout_opened(user_id, ip_country=country)
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
async def cancel_trial(
    user_id: CurrentUser, body: TrialCancelRequest | None = None
) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    if (
        trial is None
        or trial.subscription_id is None
        or trial.consumed_at is None
        or trial.converted_at is not None
    ):
        raise HTTPException(409, "No trial subscription is available to cancel")
    promised = body is not None and body.keeps_access
    if promised or not await _cancel_flag_is_off(user_id):
        await _apply_trial_change(
            schedule_trial_cancellation(trial, keeps_access=promised), CANCEL_RETRY
        )
        return await get_trial_status(user_id)
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
    "/resume",
    responses={
        409: {
            "description": (
                "No live cancel-pending trial to resume, another plan is active,"
                " or the trial is already being updated"
            )
        },
        502: {"description": "Stripe update temporarily unavailable"},
    },
)
async def resume_trial(user_id: CurrentUser) -> TrialStatusResponse:
    trial = await get_subscription_trial(user_id)
    await _apply_trial_change(
        resume_trial_subscription(trial), "Unable to resume your trial. Please retry."
    )
    return await get_trial_status(user_id)


async def _cancel_flag_is_off(user_id: str) -> bool:
    """Only an authoritative "off" ends a trial at once, which cannot be undone.
    An unreadable flag schedules the end instead: that never charges, and the
    sync after it still ends the trial at once if the flag then reads "off".
    The status copy (cancel_keeps_access) follows the same rule."""
    return await evaluate_feature_flag(
        Flag.TRIAL_CANCEL_AT_PERIOD_END, user_id, default=False
    ) == (False, True)


async def _apply_trial_change(change: Awaitable[None], retry: str) -> None:
    try:
        await change
    except TrialChangeRefused as exc:
        raise HTTPException(409, str(exc)) from exc
    except stripe.StripeError as exc:
        raise HTTPException(502, retry) from exc


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
