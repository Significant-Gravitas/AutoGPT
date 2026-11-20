"""Refresh owned trials when Stripe billing-card details change."""

import logging

import stripe
from pydantic import BaseModel, Field

from backend.data.credit import sync_subscription_from_stripe
from backend.data.db import query_raw_with_schema
from backend.data.stripe_client import stripe_call

logger = logging.getLogger(__name__)

TRIAL_BILLING_EVENTS = frozenset(
    {
        "customer.updated",
        "customer.deleted",
        "payment_method.attached",
        "payment_method.updated",
        "payment_method.automatically_updated",
        "payment_method.detached",
        "setup_intent.succeeded",
        "setup_intent.requires_action",
        "setup_intent.setup_failed",
        "setup_intent.canceled",
    }
)


class BillingObject(BaseModel):
    id: str = Field(min_length=1)
    customer: str | None = None


class PreviousBillingAttributes(BaseModel):
    customer: str | None = None


class BillingEventData(BaseModel):
    object: BillingObject
    previous_attributes: PreviousBillingAttributes | None = None


class TrialBillingTarget(BaseModel):
    id: str
    user_id: str
    customer_id: str
    subscription_id: str


class SubscriptionOwner(BaseModel):
    id: str
    customer: str
    metadata: dict[str, str] = Field(default_factory=dict)


async def sync_trials_for_billing_event(event_type: str, event_data: object) -> None:
    customer_ids = billing_event_customer_ids(event_type, event_data)
    if not customer_ids:
        return
    after_id = ""
    failed = False
    while targets := await get_trial_billing_targets(customer_ids, after_id):
        for target in targets:
            try:
                await _refresh_trial(target)
            except Exception:
                failed = True
                logger.exception(
                    f"Could not refresh billing card for trial {target.id}"
                )
        after_id = targets[-1].id
    if failed:
        raise RuntimeError("Trial billing refresh failed; retry the event")


def billing_event_customer_ids(event_type: str, event_data: object) -> list[str]:
    if event_type not in TRIAL_BILLING_EVENTS:
        return []
    event = BillingEventData.model_validate(event_data)
    if event_type in {"customer.updated", "customer.deleted"}:
        return [event.object.id]
    previous = event.previous_attributes
    return sorted(
        {
            customer
            for customer in (
                event.object.customer,
                previous.customer if previous else None,
            )
            if customer
        }
    )


async def get_trial_billing_targets(
    customer_ids: list[str], after_id: str = ""
) -> list[TrialBillingTarget]:
    return await query_raw_with_schema(
        """
        SELECT t."id", t."userId" AS user_id,
               t."stripeCustomerId" AS customer_id,
               t."stripeSubscriptionId" AS subscription_id
        FROM {schema_prefix}"SubscriptionTrial" t
        JOIN {schema_prefix}"User" u ON u."id" = t."userId"
        WHERE t."stripeCustomerId" = ANY($1::text[]) AND t."id" > $2
          AND t."consumedAt" IS NOT NULL AND t."convertedAt" IS NULL
          AND t."stripeSubscriptionId" IS NOT NULL
          AND u."stripeCustomerId" = t."stripeCustomerId"
          AND u."subscriptionTier" != 'ENTERPRISE'
        ORDER BY t."id" LIMIT 100
        """,
        customer_ids,
        after_id,
        model=TrialBillingTarget,
    )


async def _refresh_trial(target: TrialBillingTarget) -> None:
    subscription = await stripe_call(
        stripe.Subscription.retrieve_async, target.subscription_id
    )
    owner = SubscriptionOwner.model_validate(subscription)
    if (
        owner.id != target.subscription_id
        or owner.customer != target.customer_id
        or owner.metadata.get("trial_enrollment_id") != target.id
        or owner.metadata.get("user_id") != target.user_id
    ):
        raise ValueError("Stripe billing change does not match trial ownership")
    await sync_subscription_from_stripe(dict(subscription))
