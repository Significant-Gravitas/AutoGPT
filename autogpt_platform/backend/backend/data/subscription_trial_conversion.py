"""Start a paid plan on a cancel-pending trial's own subscription, today.

Stripe limits a customer to one subscription, so the plan can't be bought
beside the trial: the trial's subscription is the one that changes.
"""

import logging
from datetime import UTC, datetime

import stripe
from prisma.enums import SubscriptionTier
from pydantic import BaseModel

from backend.data.credit import (
    invalidate_billing_lookups,
    sync_subscription_from_stripe,
)
from backend.data.stripe_authentication import requires_authentication
from backend.data.stripe_client import stripe_call
from backend.data.subscription_checkout import (
    ANOTHER_PLAN_LIVE,
    SubscriptionCheckoutUnavailable,
    expire_other_subscription_checkouts,
    other_plan_is_live,
    subscription_checkout_lock,
)
from backend.data.subscription_trial import TrialState, get_subscription_trial
from backend.data.subscription_trial_payment import (
    CONVERSION_PRICE_KEY,
    CONVERSION_TIER_KEY,
    Invoice,
    subscription_card_can_be_charged,
)

logger = logging.getLogger(__name__)

TRIAL_ENDED = "This trial has ended. Manage the plan in billing."
TRIAL_RUNNING = (
    "Your accepted plan starts after your trial. Manage the trial in billing."
)
TRIAL_BUSY = "Your trial is already being updated. Please retry."
CARD_NEEDED = (
    "Your saved card can't be charged. Update it under Payment method,"
    " then try again."
)
# A pending update takes metadata from 2026-05-27.dahlia and
# cancel_at_period_end from 2026-09-30.endive. The SDK's own 2025-02-24.acacia
# rejects both with pending_if_incomplete.
PENDING_UPDATE_API_VERSION = "2026-09-30.endive"


class TrialConversionRefused(Exception):
    """The live trial is no longer one this request may convert."""


class _HeldConversion(BaseModel):
    """Stripe's answer to a conversion held for the bank to confirm."""

    pending_update: dict[str, object] | None = None
    latest_invoice: Invoice | None = None


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


async def convert_cancel_pending_trial(
    trial: TrialState, tier: SubscriptionTier, price_id: str | None = None
) -> str | None:
    """End the trial now and bill the first period of a paid plan.

    ``price_id`` names a plan other than the accepted one; None bills the
    accepted price. Returns the Stripe page where the customer confirms the
    payment with their bank, or None once the plan has started.

    ``error_if_incomplete`` leaves the trial cancel-pending when the charge
    fails. A bank that wants to confirm the charge gets a pending update
    instead, which changes nothing until it is paid.
    """
    try:
        async with subscription_checkout_lock(trial.user_id):
            return await _convert_locked(trial, tier, price_id)
    except SubscriptionCheckoutUnavailable as exc:
        raise TrialConversionRefused(TRIAL_BUSY) from exc


async def _convert_locked(
    trial: TrialState, tier: SubscriptionTier, price_id: str | None
) -> str | None:
    subscription = await _live_cancel_pending_subscription(trial)
    if subscription is None:
        return None
    # Stripe ends a trial it has no card to charge for under its
    # missing-payment-method setting instead of billing it.
    if not await subscription_card_can_be_charged(subscription.id, datetime.now(UTC)):
        raise TrialConversionRefused(CARD_NEEDED)
    await expire_other_subscription_checkouts(trial.customer_id)
    # A plan bought through Checkout ends the trial only once its webhook
    # is handled; converting before then would bill for both plans.
    if await other_plan_is_live(trial.customer_id, subscription.id):
        raise TrialConversionRefused(ANOTHER_PLAN_LIVE)
    changes = _plan_changes(subscription, tier, price_id)
    charge: stripe.Subscription.ModifyParams = {
        **changes,
        "payment_behavior": "error_if_incomplete",
    }
    try:
        # No proration_behavior: ending a trial has nothing to prorate, and
        # "none" would skip the first invoice entirely under Stripe's flexible
        # billing mode.
        converted = await stripe_call(
            stripe.Subscription.modify_async, subscription.id, **charge
        )
    except stripe.CardError as exc:
        if not requires_authentication(exc):
            raise
        return await _confirm_with_bank(trial, subscription.id, changes)
    await _save_conversion(trial, converted)
    return None


def _plan_changes(
    subscription: stripe.Subscription, tier: SubscriptionTier, price_id: str | None
) -> stripe.Subscription.ModifyParams:
    changes: stripe.Subscription.ModifyParams = {
        "cancel_at_period_end": False,
        "trial_end": "now",
    }
    if price_id is None:
        return changes
    items = subscription["items"]
    if items.get("has_more") or len(items.get("data") or []) != 1:
        raise ValueError("A trial subscription carries exactly one plan")
    # The reconcile grants a price other than the accepted one only beside the
    # tier chosen for it here.
    changes["items"] = [{"id": items["data"][0]["id"], "price": price_id}]
    changes["metadata"] = {
        CONVERSION_PRICE_KEY: price_id,
        CONVERSION_TIER_KEY: tier.value,
    }
    return changes


async def _confirm_with_bank(
    trial: TrialState,
    subscription_id: str,
    changes: stripe.Subscription.ModifyParams,
) -> str | None:
    """Hold the same change for the customer to pay on Stripe's invoice page.

    Stripe applies a pending update only once its invoice is paid, so the trial
    stays cancel-pending meanwhile; an unpaid one lapses by trial_end at the
    latest, with its invoice voided. Only this call uses the newer API version,
    and only its pending update and invoice page are read from the answer.
    """
    hold: stripe.Subscription.ModifyParams = {
        **changes,
        "payment_behavior": "pending_if_incomplete",
        "expand": ["latest_invoice"],
        "stripe_version": PENDING_UPDATE_API_VERSION,
    }
    held = await stripe_call(stripe.Subscription.modify_async, subscription_id, **hold)
    answer = _HeldConversion.model_validate(held)
    if answer.pending_update is None:
        # Paid at once: read the plan back in the version the sync parses.
        converted = await stripe_call(
            stripe.Subscription.retrieve_async, subscription_id
        )
        await _save_conversion(trial, converted)
        return None
    page = answer.latest_invoice.hosted_invoice_url if answer.latest_invoice else None
    if not page:
        raise ValueError("Stripe returned no page to confirm the payment on")
    logger.info(f"Trial {trial.id} conversion waits for the bank to confirm it")
    return page


async def _save_conversion(trial: TrialState, converted: stripe.Subscription) -> None:
    # The tier is saved first; dropping the billing page's cached lookup is
    # bounded, never raises, and comes after so it can't delay the plan.
    try:
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
    finally:
        await invalidate_billing_lookups(trial.customer_id)


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
