"""Start the paid plan a cancel-pending trial was accepted for, today."""

import logging
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
    SubscriptionCheckoutUnavailable,
    expire_other_subscription_checkouts,
    other_plan_is_live,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import TrialState, get_subscription_trial
from backend.data.subscription_trial_payment import subscription_card_can_be_charged

logger = logging.getLogger(__name__)

TRIAL_ENDED = "This trial has ended. Manage the plan in billing."
TRIAL_RUNNING = (
    "Your accepted plan starts after your trial. Manage the trial in billing."
)
TRIAL_BUSY = "Your trial is already being updated. Please retry."


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


async def convert_cancel_pending_trial(trial: TrialState) -> bool:
    """End the trial now and bill the first period of its accepted plan.

    Returns False, changing nothing, when the saved card can't be charged:
    only Checkout can take a new card. Stripe would otherwise end a card-less
    trial under its missing-payment-method setting instead of billing it.

    The items stay as accepted: the reconcile refuses an unconverted trial
    whose price changed. ``error_if_incomplete`` leaves the trial
    cancel-pending when the charge fails.
    """
    try:
        async with subscription_checkout_lock(trial.user_id):
            return await _convert_locked(trial)
    except SubscriptionCheckoutUnavailable as exc:
        raise TrialConversionRefused(TRIAL_BUSY) from exc


async def _convert_locked(trial: TrialState) -> bool:
    subscription = await _live_cancel_pending_subscription(trial)
    if subscription is None:
        return True
    if not await subscription_card_can_be_charged(subscription.id, datetime.now(UTC)):
        return False
    await expire_other_subscription_checkouts(trial.customer_id)
    # A plan bought through Checkout ends the trial only once its webhook
    # is handled; converting before then would bill for both plans.
    if await other_plan_is_live(trial.customer_id, subscription.id):
        raise TrialConversionRefused(ANOTHER_PLAN_LIVE)
    # No proration_behavior: ending a trial has nothing to prorate, and
    # "none" would skip the first invoice entirely under Stripe's flexible
    # billing mode.
    converted = await stripe_call(
        stripe.Subscription.modify_async,
        subscription.id,
        cancel_at_period_end=False,
        trial_end="now",
        payment_behavior="error_if_incomplete",
    )
    invalidate_active_subscription_cache(trial.customer_id)
    if converted.get("status") != "active":
        await sync_subscription_from_stripe(dict(converted))
        raise TrialConversionRefused(TRIAL_ENDED)
    try:
        await sync_subscription_from_stripe(dict(converted))
    except Exception:
        # The card is charged and the plan is live in Stripe; its webhooks
        # save it. Failing now would tell a paying customer it failed.
        logger.exception(
            f"Trial {trial.id} converted in Stripe but the follow-up sync failed"
        )
    return True


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


async def _live_cancel_pending_subscription(
    trial: TrialState,
) -> stripe.Subscription | None:
    """The trial's subscription while it is still cancel-pending, or None
    once Stripe shows it converted (an earlier request whose answer was lost)."""
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
    running = (subscription.get("trial_end") or 0) > datetime.now(UTC).timestamp()
    if status == "trialing" and running and subscription.get("cancel_at_period_end"):
        return subscription
    # Stripe moved on before its webhook landed: save its state first, so the
    # status the person reloads shows the plan, or why this was refused.
    await sync_subscription_from_stripe(dict(subscription))
    if status == "active":
        return None
    if status == "trialing" and running:
        raise TrialConversionRefused(TRIAL_RUNNING)
    raise TrialConversionRefused(TRIAL_ENDED)
