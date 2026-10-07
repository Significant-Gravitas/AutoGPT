"""Bind paid trial conversions to accepted terms and consumed checkout proof."""

from datetime import datetime

from prisma import Prisma
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data.db import query_raw_with_schema
from backend.data.subscription_activation_models import ActivationAttempt
from backend.data.subscription_trial import TrialState

PAID_CONVERSION_TIERS = (SubscriptionTier.PRO, SubscriptionTier.MAX)


def subscription_price_id(subscription: dict) -> str | None:
    items = subscription.get("items") or {}
    data = items.get("data", [])
    if items.get("has_more") or len(data) != 1 or data[0].get("quantity") != 1:
        return None
    return data[0].get("price", {}).get("id")


async def accepted_conversion_target(
    trial: TrialState, subscription: dict, tx: Prisma
) -> tuple[SubscriptionTier, str] | None:
    """Resolve original accepted terms or an immutable confirmed plan change."""
    price_id = subscription_price_id(subscription)
    metadata = subscription.get("metadata") or {}
    if (
        not price_id
        or subscription.get("customer") != trial.customer_id
        or metadata.get("user_id") != trial.user_id
        or metadata.get("trial_enrollment_id") != trial.id
        or metadata.get("trial_checkout_attempt") != str(trial.checkout_attempt)
        or (trial.subscription_id and subscription.get("id") != trial.subscription_id)
    ):
        return None
    if price_id == trial.offer.price_id and trial.offer.tier in PAID_CONVERSION_TIERS:
        return SubscriptionTier(trial.offer.tier), price_id
    attempt_id = metadata.get("pro_activation_attempt_id")
    if not attempt_id:
        return None
    rows = await query_raw_with_schema(
        'SELECT "id", "userId" AS user_id, '
        '"stripeSubscriptionId" AS subscription_id, '
        '"stripeCustomerId" AS customer_id, "terms", "returnTo" AS return_to, '
        '"confirmedAt" AS confirmed_at, "invoiceId" AS invoice_id '
        'FROM {schema_prefix}"ProActivationAttempt" '
        'WHERE "id" = $1 AND "userId" = $2 AND "confirmedAt" IS NOT NULL',
        attempt_id,
        trial.user_id,
        client=tx,
        model=ActivationAttempt,
    )
    if not rows:
        return None
    attempt = rows[0]
    if (
        attempt.id != attempt_id
        or attempt.confirmed_at is None
        or attempt.user_id != trial.user_id
        or attempt.customer_id != trial.customer_id
        or attempt.subscription_id != subscription.get("id")
        or attempt.terms.accepted_offer_token != trial.offer.token
        or attempt.terms.price_id != price_id
        or attempt.terms.plan not in PAID_CONVERSION_TIERS
    ):
        return None
    return SubscriptionTier(attempt.terms.plan), price_id


def owns_consumed_trial(user: User, subscription: dict, trial: TrialState) -> bool:
    metadata = subscription.get("metadata") or {}
    return bool(
        trial.consumed_at is not None
        and trial.user_id == user.id
        and trial.customer_id == user.stripeCustomerId == subscription.get("customer")
        and trial.subscription_id == subscription.get("id")
        and metadata.get("user_id") == trial.user_id
        and metadata.get("trial_enrollment_id") == trial.id
        and metadata.get("trial_checkout_attempt") == str(trial.checkout_attempt)
        and subscription.get("trial_end") is not None
    )


async def record_checkout_consumption(
    trial: TrialState, subscription_id: str, now: datetime, tx: Prisma
) -> None:
    """Persist verified Checkout proof before publishing a delayed conversion."""
    if trial.consumed_at is not None and trial.subscription_id is not None:
        return
    trial.consumed_at = trial.consumed_at or now
    trial.subscription_id = subscription_id
    await tx.subscriptiontrial.update(
        where={"userId": trial.user_id},
        data={"consumedAt": trial.consumed_at, "stripeSubscriptionId": subscription_id},
    )
