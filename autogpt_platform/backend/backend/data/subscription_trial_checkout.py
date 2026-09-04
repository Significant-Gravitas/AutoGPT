"""Card-only Checkout using the durable, server-selected trial offer."""

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial

from backend.data.credit import (
    get_stripe_customer_id,
    get_subscription_price_id,
    sync_tier_from_checkout_session,
)
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
from backend.data.subscription_trial_config import (
    AcceptedTrialOffer,
    TrialOffer,
    get_trial_offer,
)
from backend.data.user import get_user_by_id


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
) -> str:
    async with subscription_checkout_lock(user_id):
        return await _create_trial_checkout(
            user_id, offer_token, success_url, cancel_url, metadata
        )


async def _create_trial_checkout(
    user_id: str,
    offer_token: str,
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
) -> str:
    offer = await get_trial_offer(user_id)
    if offer is None:
        raise TrialUnavailable("Trials are not available right now")
    existing = await get_subscription_trial(user_id)
    if existing:
        if existing.consumed_at is not None or existing.status != "checkout_pending":
            raise TrialUnavailable("A trial has already been used for this account")
        if existing.offer.token != offer_token:
            raise TrialUnavailable("Refresh to accept the reserved trial terms")
        return await _resume_checkout(existing)
    customer_id = await get_stripe_customer_id(user_id)
    await _verify_eligibility(user_id, customer_id, offer)
    accepted = await resolve_trial_price(offer)
    if accepted.token != offer_token:
        raise TrialUnavailable(
            "The trial offer changed. Refresh to see the current terms"
        )
    trial = await reserve_subscription_trial(
        user_id, accepted, customer_id, success_url, cancel_url, metadata
    )
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
    price = await stripe.Price.retrieve_async(price_id)
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


async def _verify_eligibility(
    user_id: str, customer_id: str, offer: TrialOffer
) -> None:
    user = await get_user_by_id(user_id)
    subscriptions = await stripe.Subscription.list_async(
        customer=customer_id, status="all", limit=1
    )
    if not offer.is_eligible(
        created_at=user.created_at,
        current_tier=user.subscription_tier.value,
        has_subscription_history=bool(subscriptions.data),
    ):
        raise TrialUnavailable("This account is not eligible for a trial")


async def _resume_checkout(trial: TrialState) -> str:
    session = await _find_checkout(trial)
    if session is None or session.status == "open":
        await expire_other_subscription_checkouts(
            trial.customer_id, session.id if session else None
        )
        await _verify_eligibility(trial.user_id, trial.customer_id, trial.offer)
    if session is None:
        session = await stripe.checkout.Session.create_async(
            **trial_checkout_params(trial),
            idempotency_key=f"trial-checkout-{trial.id}-{trial.checkout_attempt}",
        )
        await SubscriptionTrial.prisma().update_many(
            where={"id": trial.id, "checkoutAttempt": trial.checkout_attempt},
            data={"stripeCheckoutSessionId": session.id},
        )
    if session.status == "expired":
        return await _replace_expired_checkout(trial)
    if session.status != "open" or not session.url:
        raise TrialUnavailable(
            "Trial checkout is complete. Refresh your billing status"
        )
    return session.url


async def _find_checkout(trial: TrialState) -> stripe.checkout.Session | None:
    if trial.checkout_session_id:
        return await stripe.checkout.Session.retrieve_async(trial.checkout_session_id)
    sessions = await stripe.checkout.Session.list_async(
        customer=trial.customer_id, limit=100
    )
    async for session in sessions.auto_paging_iter():
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


async def _replace_expired_checkout(trial: TrialState) -> str:
    await _verify_eligibility(trial.user_id, trial.customer_id, trial.offer)
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
        },
    )
    current = await get_subscription_trial(trial.user_id)
    if current is None or current.consumed_at is not None:
        raise TrialUnavailable("Trial checkout is no longer available")
    if current.checkout_attempt == trial.checkout_attempt:
        raise TrialUnavailable("Trial checkout changed. Refresh your billing status")
    return await _resume_checkout(current)


def trial_checkout_params(trial: TrialState) -> dict:
    metadata = {
        **trial.checkout_metadata,
        "user_id": trial.user_id,
        "tier": trial.offer.tier,
        "billing_cycle": trial.offer.billing_cycle,
        "trial_enrollment_id": trial.id,
        "trial_offer_version": trial.offer.version,
        "trial_checkout_attempt": str(trial.checkout_attempt),
    }
    return {
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
