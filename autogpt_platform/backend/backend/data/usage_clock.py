"""Our copy of each customer's Stripe billing period, which usage limits reset on.

Stripe is the source of truth. The copy is written wherever a subscription is
read fresh from Stripe (every subscription webhook through the tier sync, an
upgrade's own response, the periodic sweep, the backfill command) and re-read
on demand once it has ended (:func:`refresh_billing_clock`).

It only moves forwards. For the subscription already stored, a period that
starts earlier never replaces one that starts later: a renewal starts later
and an upgrade that resets the billing date starts now, so a late, duplicate
or out-of-order delivery cannot put the clock back. Another subscription
takes the clock over only when it is the newer plan (created later), or when
the stored period has ended and that plan is the one still live. So an old
plan left running beside its replacement neither pulls the clock back nor
trades it back and forth at each renewal, and when the newer plan is the one
that ends, the clock passes to the plan that remains.

Writing the clock never changes a tier.
"""

import logging
from collections.abc import Mapping
from typing import Any, Literal

import stripe
from prisma import Prisma
from prisma.enums import SubscriptionTier
from prisma.models import User
from pydantic import BaseModel

from backend.data.db import execute_raw_with_schema
from backend.data.stripe_client import stripe_call

logger = logging.getLogger(__name__)

# Unix seconds outside this range are a unit mistake (milliseconds) or junk.
EARLIEST_SECOND = 1_000_000_000
LATEST_SECOND = 100_000_000_000
SHORTEST_PERIOD = 3600

RefreshOutcome = Literal["saved", "unchanged", "no_subscription", "not_stripe_billed"]


class BillingPeriod(BaseModel):
    """The period Stripe reports for a subscription, in Unix seconds."""

    subscription_id: str
    start: int
    end: int
    anchor: int | None = None
    # When Stripe created the subscription: which of two plans is the newer.
    created: int


def billing_period(subscription: Mapping[str, Any]) -> BillingPeriod | None:
    """The subscription's current period, or None when it carries no sane one.

    Stripe moved the period from the subscription to its items in API version
    2025-03-31; both places are read so an API upgrade doesn't lose the clock.
    """
    items = (subscription.get("items") or {}).get("data") or []
    item = items[0] if items else {}
    start = _second(
        subscription.get("current_period_start") or item.get("current_period_start")
    )
    end = _second(
        subscription.get("current_period_end") or item.get("current_period_end")
    )
    subscription_id = subscription.get("id")
    if not subscription_id or start is None or end is None:
        return None
    if end - start < SHORTEST_PERIOD:
        return None
    return BillingPeriod(
        subscription_id=subscription_id,
        start=start,
        end=end,
        anchor=_second(subscription.get("billing_cycle_anchor")),
        created=(
            _second(subscription.get("created"))
            or _second(subscription.get("start_date"))
            or start
        ),
    )


async def save_billing_clock(
    user_id: str, subscription: Mapping[str, Any], *, client: Prisma | None = None
) -> bool:
    """Copy the subscription's period for the user, unless a later one is
    already stored. True when the row was written."""
    period = billing_period(subscription)
    if period is None:
        logger.error(
            "A subscription has no usable billing period; keeping the stored clock"
        )
        return False
    written = await execute_raw_with_schema(
        """INSERT INTO {schema_prefix}"UsageBillingClock" AS c
            ("userId", "stripeSubscriptionId", "periodStart", "periodEnd",
             "billingCycleAnchor", "subscriptionCreatedAt", "updatedAt")
        VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
        ON CONFLICT ("userId") DO UPDATE SET
            "stripeSubscriptionId" = EXCLUDED."stripeSubscriptionId",
            "periodStart" = EXCLUDED."periodStart",
            "periodEnd" = EXCLUDED."periodEnd",
            "billingCycleAnchor" = EXCLUDED."billingCycleAnchor",
            "subscriptionCreatedAt" = EXCLUDED."subscriptionCreatedAt",
            "updatedAt" = CURRENT_TIMESTAMP
        WHERE (
            c."stripeSubscriptionId" = EXCLUDED."stripeSubscriptionId"
            AND c."periodStart" <= EXCLUDED."periodStart"
        ) OR (
            c."stripeSubscriptionId" <> EXCLUDED."stripeSubscriptionId"
            AND (
                c."subscriptionCreatedAt" < EXCLUDED."subscriptionCreatedAt"
                OR c."periodEnd" <= EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::bigint
            )
        )
        """,
        user_id,
        period.subscription_id,
        period.start,
        period.end,
        period.anchor,
        period.created,
        client=client,
    )
    # No user id and nothing read off the Stripe object: the code scanner
    # counts both as private here. Callers that know who it is for say so.
    if written:
        logger.info("Billing clock copied from Stripe")
    else:
        logger.warning(
            "Billing clock kept: Stripe's period is older than the stored "
            "one, or belongs to an older plan"
        )
    return bool(written)


async def refresh_billing_clock(user_id: str) -> RefreshOutcome:
    """Re-read the user's live subscription from Stripe and copy its period.

    For a stored period that has ended, or a Stripe-billed plan with none
    stored. Only the clock is written, never the tier. Raises
    ``stripe.StripeError`` when Stripe can't be read, so the caller can tell
    "no subscription" from "no answer".
    """
    user = await User.prisma().find_unique(where={"id": user_id})
    if (
        user is None
        or not user.stripeCustomerId
        or user.subscriptionTier
        in (SubscriptionTier.ENTERPRISE, SubscriptionTier.TRIAL)
    ):
        return "not_stripe_billed"
    subscription = await _live_plan(user.stripeCustomerId, user_id)
    if subscription is None:
        return "no_subscription"
    return "saved" if await save_billing_clock(user_id, subscription) else "unchanged"


async def _live_plan(customer_id: str, user_id: str) -> dict[str, Any] | None:
    """The customer's active (else trialing) plan that belongs to this user.
    A trial enrolment isn't one: a trial's windows run on the trial's own dates."""
    for status in ("active", "trialing"):
        page = await stripe_call(
            stripe.Subscription.list_async,
            customer=customer_id,
            status=status,
            limit=10,
        )
        for found in page.data:
            subscription = dict(found)
            metadata = subscription.get("metadata") or {}
            if subscription.get("customer") != customer_id:
                continue
            if metadata.get("user_id", user_id) != user_id:
                continue
            if metadata.get("trial_enrollment_id"):
                continue
            return subscription
    return None


def _second(value: Any) -> int | None:
    """The value when it is a Unix second in a sane range, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if EARLIEST_SECOND < value < LATEST_SECOND else None
