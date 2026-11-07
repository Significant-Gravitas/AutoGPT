"""Card-only Checkout using the durable, server-selected trial offer."""

import logging
from typing import Any

import stripe
from prisma.enums import SubscriptionTier
from prisma.errors import PrismaError
from prisma.models import SubscriptionTrial

from backend.data import experiments
from backend.data.credit import (
    get_stripe_customer_id,
    get_subscription_price_id,
    sync_tier_from_checkout_session,
)
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_checkout import (
    SubscriptionCheckoutUnavailable as TrialUnavailable,
)
from backend.data.subscription_checkout import (
    expire_other_subscription_checkouts,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import (
    TrialState,
    get_subscription_trial,
    reserve_subscription_trial,
)
from backend.data.subscription_trial_capacity import (
    TRIAL_FULL,
    TrialCapacityReached,
    renew_trial_seat,
)
from backend.data.subscription_trial_config import (
    AcceptedTrialOffer,
    TrialOffer,
    get_trial_offer,
)
from backend.data.subscription_trial_eligibility import (
    verify_trial_eligibility as _verify_eligibility,
)

logger = logging.getLogger(__name__)

# The arm of Experiment.TRIAL_CHECKOUT_NO_CHARGE_COPY that shows the line; any
# other arm, or no enrolment, keeps today's page.
NO_CHARGE_COPY_ARM = "no-charge-copy"


async def confirm_trial_checkout(user_id: str) -> None:
    trial = await get_subscription_trial(user_id)
    if trial is None:
        raise TrialUnavailable("There is no trial checkout for this account")
    session = await _find_checkout(trial)
    if session is None or session.status != "complete":
        raise TrialUnavailable("Card setup is not complete yet")
    if session.customer != trial.customer_id or session.mode != "subscription":
        raise TrialUnavailable("Checkout does not match the trial enrollment")
    await sync_tier_from_checkout_session(dict(session))


async def create_trial_checkout(
    user_id: str,
    offer_token: str,
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
    *,
    country: str | None = None,
) -> str:
    async with subscription_checkout_lock(user_id):
        return await _create_trial_checkout(
            user_id, offer_token, success_url, cancel_url, metadata, country
        )


async def _create_trial_checkout(
    user_id: str,
    offer_token: str,
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
    country: str | None,
) -> str:
    # The same question the status endpoint answers before showing the offer,
    # asked of the *current* flag: someone who would not be shown the trial
    # now cannot start or resume one, whatever they were shown earlier.
    offer = await get_trial_offer(user_id, country=country)
    if offer is None:
        raise TrialUnavailable("Trials are not available right now")
    existing = await get_subscription_trial(user_id)
    if existing:
        if existing.consumed_at is not None or existing.status != "checkout_pending":
            raise TrialUnavailable("A trial has already been used for this account")
        if existing.offer.token != offer_token:
            raise TrialUnavailable("Refresh to accept the reserved trial terms")
        if not await renew_trial_seat(offer, existing.id):
            raise TrialUnavailable(TRIAL_FULL)
        return await _resume_checkout(existing)
    customer_id = await get_stripe_customer_id(user_id)
    await _verify_eligibility(user_id, customer_id, offer)
    accepted = await resolve_trial_price(offer)
    if accepted.token != offer_token:
        raise TrialUnavailable(
            "The trial offer changed. Refresh to see the current terms"
        )
    try:
        trial = await reserve_subscription_trial(
            user_id, accepted, customer_id, success_url, cancel_url, metadata
        )
    except TrialCapacityReached as exc:
        raise TrialUnavailable(str(exc)) from exc
    if trial.offer.token != offer_token:
        raise TrialUnavailable(
            "Another checkout reserved different trial terms. Refresh"
        )
    return await _resume_checkout(trial)


async def resolve_trial_price(offer: TrialOffer) -> AcceptedTrialOffer:
    price_id = await get_subscription_price_id(
        SubscriptionTier(offer.tier), offer.billing_cycle
    )
    if not price_id:
        raise TrialUnavailable("The trial's paid plan is not configured")
    price = await stripe_call(stripe.Price.retrieve_async, price_id)
    interval = "month" if offer.billing_cycle == "monthly" else "year"
    if (
        not price.active
        or price.unit_amount is None
        or price.unit_amount <= 0
        or price.recurring is None
        or price.recurring.interval != interval
        or price.recurring.interval_count != 1
        or price.recurring.usage_type != "licensed"
    ):
        raise TrialUnavailable(
            "The trial's paid price must be an active recurring price"
        )
    return AcceptedTrialOffer(
        **offer.model_dump(),
        price_id=price.id,
        unit_amount=price.unit_amount,
        currency=price.currency,
    )


async def _resume_checkout(trial: TrialState) -> str:
    session = await _find_checkout(trial)
    if session is None or session.status == "open":
        await expire_other_subscription_checkouts(
            trial.customer_id, session.id if session else None
        )
        await _verify_eligibility(
            trial.user_id, trial.customer_id, trial.offer, trial=trial, session=session
        )
    if session is None:
        no_charge_copy = await _shows_no_charge_copy(trial.user_id)
        session = await stripe_call(
            stripe.checkout.Session.create_async,
            **trial_checkout_params(trial, no_charge_copy=no_charge_copy),
            idempotency_key=f"trial-checkout-{trial.id}-{trial.checkout_attempt}",
        )
        await SubscriptionTrial.prisma().update_many(
            where={"id": trial.id, "checkoutAttempt": trial.checkout_attempt},
            data={"stripeCheckoutSessionId": session.id},
        )
    if session.status == "expired":
        return await _replace_expired_checkout(trial, session)
    if session.status != "open" or not session.url:
        raise TrialUnavailable(
            "Trial checkout is complete. Refresh your billing status"
        )
    return session.url


async def _find_checkout(trial: TrialState) -> stripe.checkout.Session | None:
    if trial.checkout_session_id:
        return await stripe_call(
            stripe.checkout.Session.retrieve_async, trial.checkout_session_id
        )
    sessions = await stripe_call(
        stripe.checkout.Session.list_async, customer=trial.customer_id, limit=100
    )
    async for session in stripe_list_items(sessions):
        metadata = session.metadata or {}
        if metadata.get("trial_enrollment_id") == trial.id and metadata.get(
            "trial_checkout_attempt"
        ) == str(trial.checkout_attempt):
            await SubscriptionTrial.prisma().update_many(
                where={"id": trial.id, "checkoutAttempt": trial.checkout_attempt},
                data={"stripeCheckoutSessionId": session.id},
            )
            return session
    return None


async def _replace_expired_checkout(
    trial: TrialState, session: stripe.checkout.Session
) -> str:
    await _verify_eligibility(
        trial.user_id, trial.customer_id, trial.offer, trial=trial, session=session
    )
    await SubscriptionTrial.prisma().update_many(
        where={
            "id": trial.id,
            "checkoutAttempt": trial.checkout_attempt,
            "status": "checkout_pending",
            "consumedAt": None,
        },
        data={
            "checkoutAttempt": {"increment": 1},
            "stripeCheckoutSessionId": None,
            "stripeSubscriptionId": None,
            "startedAt": None,
            "endsAt": None,
            "cardVerifiedAt": None,
        },
    )
    current = await get_subscription_trial(trial.user_id)
    if current is None or current.consumed_at is not None:
        raise TrialUnavailable("Trial checkout is no longer available")
    if current.checkout_attempt == trial.checkout_attempt:
        raise TrialUnavailable("Trial checkout changed. Refresh your billing status")
    return await _resume_checkout(current)


def trial_checkout_params(trial: TrialState, *, no_charge_copy: bool) -> dict:
    metadata = {
        **trial.checkout_metadata,
        "user_id": trial.user_id,
        "tier": trial.offer.tier,
        "billing_cycle": trial.offer.billing_cycle,
        "trial_enrollment_id": trial.id,
        "trial_offer_version": trial.offer.version,
        "trial_checkout_attempt": str(trial.checkout_attempt),
    }
    params: dict[str, Any] = {
        "customer": trial.customer_id,
        "mode": "subscription",
        "line_items": [{"price": trial.offer.price_id, "quantity": 1}],
        "success_url": trial.success_url,
        "cancel_url": trial.cancel_url,
        "payment_method_types": ["card"],
        "payment_method_collection": "always",
        "subscription_data": {
            "trial_period_days": trial.offer.duration_days,
            "trial_settings": {"end_behavior": {"missing_payment_method": "cancel"}},
            "metadata": metadata,
        },
        "allow_promotion_codes": False,
        "automatic_tax": {"enabled": True},
        "billing_address_collection": "auto",
        "customer_update": {"address": "auto"},
        "metadata": metadata,
    }
    if no_charge_copy:
        params["custom_text"] = {
            "after_submit": {"message": _no_charge_today(trial.offer.duration_days)}
        }
    return params


async def _shows_no_charge_copy(user_id: str) -> bool:
    # Assigned before the route sends checkout_started, so that event carries the arm.
    try:
        assignment = await experiments.assign(
            user_id, experiments.Experiment.TRIAL_CHECKOUT_NO_CHARGE_COPY
        )
    except PrismaError:
        # An experiment never blocks a trial: today's page, and nothing stored.
        logger.warning(
            f"Trial checkout experiment not assigned for user {user_id}", exc_info=True
        )
        return False
    return assignment is not None and assignment.variant == NO_CHARGE_COPY_ARM


def _no_charge_today(duration_days: int) -> str:
    # No date: an open session can be completed up to a day after it is
    # created, and Stripe already prints the live first-charge date above.
    return (
        f"No charge today. You're charged when your {duration_days}-day trial "
        "ends. Cancel before then in Settings > Billing and you won't pay "
        "anything."
    )
