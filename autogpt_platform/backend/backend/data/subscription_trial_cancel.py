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
    ANOTHER_PLAN_LIVE,
    SubscriptionCheckoutUnavailable,
    expire_other_subscription_checkouts,
    other_plan_is_live,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import TrialState
from backend.data.subscription_trial_payment import subscription_card_can_be_charged

TRIAL_ENDED = "This trial has ended. Manage the plan in billing."
NOTHING_TO_RESUME = "Nothing to resume."
TRIAL_BUSY = "Your trial is already being updated. Please retry."
CARD_NEEDED = "Update your card under Payment method, then resume your trial."


class TrialChangeRefused(ValueError):
    """The live trial cannot take this change; the message is safe to show."""


async def schedule_trial_cancellation(trial: TrialState) -> None:
    """Under the checkout lock, so it never lands on the paid plan that
    subscribing now converts the trial into."""
    try:
        async with subscription_checkout_lock(trial.user_id):
            await _schedule_locked(trial)
    except SubscriptionCheckoutUnavailable as exc:
        raise TrialChangeRefused(TRIAL_BUSY) from exc


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
        raise TrialChangeRefused(TRIAL_BUSY) from exc


async def _schedule_locked(trial: TrialState) -> None:
    subscription = await _live_trial_subscription(trial)
    if not subscription.get("cancel_at_period_end"):
        subscription = await _change_live_trial(
            subscription, {"cancel_at_period_end": True}
        )
    await sync_subscription_from_stripe(dict(subscription))


async def _resume_locked(trial: TrialState) -> None:
    subscription = await _live_trial_subscription(trial)
    if (subscription.get("trial_end") or 0) <= datetime.now(UTC).timestamp():
        raise await _synced_refusal(subscription, TRIAL_ENDED)
    if not subscription.get("cancel_at_period_end"):
        raise await _synced_refusal(subscription, NOTHING_TO_RESUME)
    # Only a cancel-pending trial keeps access without a card Stripe can charge;
    # resuming without one would end the access it still has.
    if not await subscription_card_can_be_charged(subscription.id, datetime.now(UTC)):
        raise TrialChangeRefused(CARD_NEEDED)
    # A plan checkout opened while cancel-pending must not complete beside the
    # resumed trial: that would leave two live subscriptions. The checkout lock
    # keeps a new one from opening between this expiry and the resume.
    await expire_other_subscription_checkouts(trial.customer_id)
    # A plan already bought that way ends the trial only once its webhook (or
    # the stale-subscription cleanup) runs; resuming before then would let the
    # trial convert next to it and bill twice.
    if await other_plan_is_live(trial.customer_id, subscription.id):
        raise TrialChangeRefused(ANOTHER_PLAN_LIVE)
    subscription = await _change_live_trial(
        subscription, {"cancel_at_period_end": False}
    )
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
    if subscription.get("status") != "trialing":
        raise await _synced_refusal(subscription, TRIAL_ENDED)
    return subscription


async def _synced_refusal(
    subscription: stripe.Subscription, message: str
) -> TrialChangeRefused:
    """Stripe moved on before its webhook landed: save its state first, so the
    status the person reloads shows why the change was refused."""
    await sync_subscription_from_stripe(dict(subscription))
    return TrialChangeRefused(message)


async def _change_live_trial(
    subscription: stripe.Subscription, change: stripe.Subscription.ModifyParams
) -> stripe.Subscription:
    try:
        return await stripe_call(
            stripe.Subscription.modify_async, subscription.id, **change
        )
    except stripe.InvalidRequestError as exc:
        # Stripe rejects the update once the trial has ended between our read
        # and the write; anything else stays a retryable failure.
        latest = await stripe_call(stripe.Subscription.retrieve_async, subscription.id)
        if latest.get("status") != "canceled":
            raise
        raise await _synced_refusal(latest, TRIAL_ENDED) from exc
