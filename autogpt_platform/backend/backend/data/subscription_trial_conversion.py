"""Start the paid plan a cancel-pending trial was accepted for, today."""

from datetime import UTC, datetime

import stripe
from prisma.enums import SubscriptionTier

from backend.data.credit import (
    invalidate_active_subscription_cache,
    sync_subscription_from_stripe,
)
from backend.data.stripe_client import stripe_call
from backend.data.subscription_checkout import (
    ANOTHER_PLAN_LIVE,
    expire_other_subscription_checkouts,
    other_plan_is_live,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import TrialState, get_subscription_trial

TRIAL_ENDED = "This trial has ended. Manage the plan in billing."
TRIAL_RUNNING = (
    "Your accepted plan starts after your trial. Manage the trial in billing."
)


class TrialConversionRefused(Exception):
    """The live trial is no longer one this request may convert."""


async def get_cancel_pending_trial(user_id: str) -> TrialState | None:
    """The user's trial when it is running but scheduled to end unconverted."""
    trial = await get_subscription_trial(user_id)
    if trial is None or not _is_cancel_pending(trial):
        return None
    return trial


def is_trial_plan(
    trial: TrialState, tier: SubscriptionTier, billing_cycle: str
) -> bool:
    """Whether this is the plan the trial accepted. The conversion bills the
    accepted price, the one the plan card shows, even after a re-price."""
    offer = trial.offer
    return tier.value == offer.tier and billing_cycle == offer.billing_cycle


async def convert_cancel_pending_trial(trial: TrialState) -> None:
    """End the trial now and bill the first period of its accepted plan.

    The items stay as accepted: the reconcile refuses an unconverted trial
    whose price changed. ``error_if_incomplete`` leaves the trial
    cancel-pending when the charge fails.
    """
    async with subscription_checkout_lock(trial.user_id):
        subscription = await _live_cancel_pending_subscription(trial)
        await expire_other_subscription_checkouts(trial.customer_id)
        # A plan bought through Checkout ends the trial only once its webhook
        # is handled; converting before then would bill for both plans.
        if await other_plan_is_live(trial.customer_id, subscription.id):
            raise TrialConversionRefused(ANOTHER_PLAN_LIVE)
        converted = await stripe_call(
            stripe.Subscription.modify_async,
            subscription.id,
            cancel_at_period_end=False,
            trial_end="now",
            proration_behavior="none",
            payment_behavior="error_if_incomplete",
        )
        invalidate_active_subscription_cache(trial.customer_id)
        await sync_subscription_from_stripe(dict(converted))


def _is_cancel_pending(trial: TrialState) -> bool:
    return bool(
        trial.subscription_id
        and trial.consumed_at is not None
        and trial.converted_at is None
        and trial.status == "trialing"
        and trial.cancel_at_period_end
        and trial.ends_at is not None
        and trial.ends_at > datetime.now(UTC)
    )


async def _live_cancel_pending_subscription(trial: TrialState) -> stripe.Subscription:
    if not trial.subscription_id:
        raise TrialConversionRefused(TRIAL_ENDED)
    subscription = await stripe_call(
        stripe.Subscription.retrieve_async, trial.subscription_id
    )
    metadata = subscription.get("metadata") or {}
    if (
        subscription.get("customer") != trial.customer_id
        or metadata.get("trial_enrollment_id") != trial.id
        or metadata.get("user_id") != trial.user_id
    ):
        raise TrialConversionRefused(TRIAL_ENDED)
    status = subscription.get("status")
    if status == "canceled":
        await sync_subscription_from_stripe(dict(subscription))
    trial_end = subscription.get("trial_end") or 0
    if status != "trialing" or trial_end <= datetime.now(UTC).timestamp():
        raise TrialConversionRefused(TRIAL_ENDED)
    if not subscription.get("cancel_at_period_end"):
        raise TrialConversionRefused(TRIAL_RUNNING)
    return subscription
