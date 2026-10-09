"""Schedule a trial's end, or take that back, while the trial keeps its access.

Stripe ends a cancel-pending trial at trial_end with no invoice. Neither change
sends a notice here: the customer.subscription.updated webhook does, once per
flip, keyed by the trial's notification revision.
"""

from datetime import UTC, datetime

import stripe

from backend.data.credit import sync_subscription_from_stripe
from backend.data.stripe_client import stripe_call
from backend.data.subscription_checkout import (
    SubscriptionCheckoutUnavailable,
    expire_other_subscription_checkouts,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import TrialState

TRIAL_ENDED = "This trial has ended. Manage the plan in billing."
NOTHING_TO_RESUME = "Nothing to resume."


class TrialChangeRefused(ValueError):
    """The live trial cannot take this change; the message is safe to show."""


async def schedule_trial_cancellation(trial: TrialState) -> None:
    subscription = await _live_trial_subscription(trial)
    if not subscription.get("cancel_at_period_end"):
        subscription = await _set_cancel_at_period_end(subscription, True)
    await sync_subscription_from_stripe(dict(subscription))


async def resume_trial_subscription(trial: TrialState | None) -> None:
    if (
        trial is None
        or trial.subscription_id is None
        or trial.consumed_at is None
        or trial.converted_at is not None
    ):
        raise TrialChangeRefused(NOTHING_TO_RESUME)
    try:
        async with subscription_checkout_lock(trial.user_id):
            await _resume_locked(trial)
    except SubscriptionCheckoutUnavailable as exc:
        raise TrialChangeRefused(str(exc)) from exc


async def _resume_locked(trial: TrialState) -> None:
    subscription = await _live_trial_subscription(trial)
    if (subscription.get("trial_end") or 0) <= datetime.now(UTC).timestamp():
        raise TrialChangeRefused(TRIAL_ENDED)
    if not subscription.get("cancel_at_period_end"):
        raise TrialChangeRefused(NOTHING_TO_RESUME)
    # A plan checkout opened while cancel-pending must not complete beside the
    # resumed trial: that would leave two live subscriptions. The checkout lock
    # keeps a new one from opening between this expiry and the resume.
    await expire_other_subscription_checkouts(trial.customer_id)
    subscription = await _set_cancel_at_period_end(subscription, False)
    await sync_subscription_from_stripe(dict(subscription))


async def _live_trial_subscription(trial: TrialState) -> stripe.Subscription:
    if trial.subscription_id is None:
        raise TrialChangeRefused(TRIAL_ENDED)
    subscription = await stripe_call(
        stripe.Subscription.retrieve_async, trial.subscription_id
    )
    metadata = subscription.get("metadata") or {}
    if (
        subscription.get("customer") != trial.customer_id
        or metadata.get("trial_enrollment_id") != trial.id
        or metadata.get("user_id") != trial.user_id
    ):
        raise TrialChangeRefused(TRIAL_ENDED)
    if subscription.get("status") == "canceled":
        await sync_subscription_from_stripe(dict(subscription))
        raise TrialChangeRefused(TRIAL_ENDED)
    if subscription.get("status") != "trialing":
        raise TrialChangeRefused(TRIAL_ENDED)
    return subscription


async def _set_cancel_at_period_end(
    subscription: stripe.Subscription, cancel_at_period_end: bool
) -> stripe.Subscription:
    try:
        return await stripe_call(
            stripe.Subscription.modify_async,
            subscription.id,
            cancel_at_period_end=cancel_at_period_end,
        )
    except stripe.InvalidRequestError as exc:
        # Stripe rejects the update once the trial has ended between our read
        # and the write; anything else stays a retryable failure.
        latest = await stripe_call(stripe.Subscription.retrieve_async, subscription.id)
        if latest.get("status") != "canceled":
            raise
        await sync_subscription_from_stripe(dict(latest))
        raise TrialChangeRefused(TRIAL_ENDED) from exc
