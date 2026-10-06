"""Publish a fresh usage generation only with settled Stripe billing evidence.

The generation, trial conversion identity and entitlement commit together. Redis
is only read, so a DB rollback or any replay cannot delete paid consumption.
"""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import stripe
from prisma import Prisma
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.copilot.usage_activation import verify_activation_usage
from backend.data import credit
from backend.data.db import query_raw_with_schema, transaction
from backend.data.pro_activation import get_usage_activation_state
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_activation_evidence import (
    INVOICE_REFERENCE,
    first_settled_invoice,
    invoice_subscription,
    line_price,
    qualifying_invoice,
    settled_recurring_invoice,
)
from backend.data.subscription_trial import TrialState


async def reconcile_paid_activation(user_id: str, subscription_id: str) -> bool:
    user = await User.prisma().find_unique_or_raise(where={"id": user_id})
    raw = await stripe_call(stripe.Subscription.retrieve_async, subscription_id)
    verify_subscription_owner(user, dict(raw))
    await credit.sync_subscription_from_stripe(dict(raw))
    state = await get_usage_activation_state(user_id)
    if not state.ready or state.tier != SubscriptionTier.PRO:
        return False
    if raw.status != "active" or not await current_invoice_is_settled(dict(raw)):
        return False
    async with transaction() as tx:
        locked = await lock_activation_user(user_id, tx)
        if locked.subscriptionTier != SubscriptionTier.PRO:
            return False
        await query_raw_with_schema(
            'SELECT "id" FROM {schema_prefix}"SubscriptionTrial" WHERE "userId" = $1 FOR UPDATE',
            user_id,
            client=tx,
        )
        row = await tx.subscriptiontrial.find_unique(where={"userId": user_id})
        trial = TrialState.from_db(row) if row else None
        current = dict(
            await stripe_call(stripe.Subscription.retrieve_async, subscription_id)
        )
        price_id = subscription_price_id(current)
        if not price_id:
            return False
        if trial and trial.subscription_id == subscription_id:
            if price_id != trial.offer.price_id:
                return False
        elif (await credit.build_price_to_tier_map()).get(
            price_id
        ) != SubscriptionTier.PRO:
            return False
        # Even a pre-existing/admin Pro label cannot hide incomplete first-payment
        # reconciliation. This returns false if reset eligibility is unknown.
        return await publish_initial_pro_activation(
            locked, current, price_id, tx, trial
        )


async def lock_activation_user(user_id: str, tx: Prisma) -> User:
    await query_raw_with_schema(
        'SELECT "id" FROM {schema_prefix}"User" WHERE "id" = $1 FOR UPDATE',
        user_id,
        client=tx,
    )
    return await tx.user.find_unique_or_raise(where={"id": user_id})


def verify_subscription_owner(user: User, subscription: dict) -> None:
    if (
        subscription.get("customer") != user.stripeCustomerId
        or (subscription.get("metadata") or {}).get("user_id", user.id) != user.id
    ):
        raise ValueError("Subscription does not belong to this user")


def subscription_price_id(subscription: dict) -> str | None:
    items = subscription.get("items", {})
    data = items.get("data", [])
    if items.get("has_more") or len(data) != 1 or data[0].get("quantity") != 1:
        return None
    return data[0].get("price", {}).get("id")


async def current_invoice_is_settled(subscription: dict) -> bool:
    latest = subscription.get("latest_invoice")
    if not latest or subscription.get("status") != "active":
        return False
    invoice_id = INVOICE_REFERENCE.validate_python(latest).invoice_id
    invoice = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
    if invoice.get("lines", {}).get("has_more"):
        lines = await stripe_call(
            stripe.Invoice.list_lines_async, invoice_id, limit=100
        )
        invoice["lines"] = {
            "data": [dict(line) async for line in stripe_list_items(lines)],
            "has_more": False,
        }
    return bool(
        settled_recurring_invoice(invoice)
        and invoice.get("customer") == subscription.get("customer")
        and invoice_subscription(invoice) == subscription.get("id")
        and (
            subscription.get("trial_end") is None
            or invoice.get("created", 0) >= subscription["trial_end"]
        )
    )


async def publish_initial_pro_activation(
    user: User,
    subscription: dict,
    price_id: str,
    tx: Prisma,
    trial: TrialState | None = None,
) -> bool:
    """Under User then Trial row locks; return whether paid access is proven."""
    verify_subscription_owner(user, subscription)
    if subscription_price_id(subscription) != price_id:
        return False
    if not await current_invoice_is_settled(subscription):
        return False
    existing = await tx.paidusageactivation.find_unique(where={"userId": user.id})
    if existing:
        if existing.stripeSubscriptionId == subscription["id"]:
            await _complete_activation(existing.id, user.id, tx)
        return True
    # Enterprise remains manually managed. Paid history comes from Stripe,
    # never from a tier label that an admin or delayed cache may have assigned.
    if user.subscriptionTier == SubscriptionTier.ENTERPRISE:
        return True
    first = await first_settled_invoice(user.stripeCustomerId or "")
    if first is None:
        return False
    if invoice_subscription(first) != subscription["id"]:
        return True  # A returning subscriber keeps their existing consumption.
    policy = await tx.paidusageactivationpolicy.find_unique_or_raise(
        where={"id": "initial-pro-v1"}
    )
    paid_at = datetime.fromtimestamp(first["status_transitions"]["paid_at"], UTC)
    if paid_at < policy.startsAt.replace(tzinfo=UTC):
        return True
    if not qualifying_invoice(
        first,
        customer_id=user.stripeCustomerId or "",
        subscription_id=subscription["id"],
        price_id=price_id,
        trial_end=subscription.get("trial_end"),
    ):
        # A paid-plan change has historical service at another price. Missing
        # first-period evidence is different: leave it recoverable, fail closed.
        prices = {line_price(line) for line in first.get("lines", {}).get("data", [])}
        return (
            trial is None
            and bool(prices)
            and None not in prices
            and price_id not in prices
        )
    generation = str(
        uuid5(
            NAMESPACE_URL,
            f"autogpt:initial-pro:{user.id}:{subscription['id']}:{first['id']}",
        )
    )
    await tx.paidusageactivation.create(
        data={
            "id": generation,
            "userId": user.id,
            "stripeSubscriptionId": subscription["id"],
            "stripeInvoiceId": first["id"],
        }
    )
    await _complete_activation(generation, user.id, tx)
    if trial:
        await tx.subscriptiontrial.update(
            where={"userId": user.id},
            data={"convertedAt": paid_at, "stripeConversionInvoiceId": first["id"]},
        )
    return True


async def _complete_activation(generation: str, user_id: str, tx: Prisma) -> None:
    await verify_activation_usage(user_id, generation)
    await tx.paidusageactivation.update_many(
        where={"id": generation, "userId": user_id, "readyAt": None},
        data={"readyAt": datetime.now(UTC)},
    )
