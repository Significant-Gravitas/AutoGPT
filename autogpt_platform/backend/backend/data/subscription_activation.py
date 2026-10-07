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
    settled_proration_invoice,
    settled_recurring_invoice,
)
from backend.data.subscription_activation_models import PaidActivationResult
from backend.data.subscription_activation_target import (
    PAID_CONVERSION_TIERS,
    accepted_conversion_target,
    owns_consumed_trial,
    subscription_price_id,
)
from backend.data.subscription_trial import TrialState


async def reconcile_paid_activation(
    user_id: str, subscription_id: str, *, expected_price_id: str | None = None
) -> PaidActivationResult | None:
    user = await User.prisma().find_unique_or_raise(where={"id": user_id})
    raw = await stripe_call(stripe.Subscription.retrieve_async, subscription_id)
    verify_subscription_owner(user, dict(raw))
    await credit.sync_subscription_from_stripe(dict(raw))
    state = await get_usage_activation_state(user_id)
    if not state.ready or state.tier not in PAID_CONVERSION_TIERS:
        return None
    if raw.status != "active" or not await current_invoice_is_settled(
        dict(raw), allow_proration=True
    ):
        return None
    async with transaction() as tx:
        locked = await lock_activation_user(user_id, tx)
        if locked.subscriptionTier not in PAID_CONVERSION_TIERS:
            return None
        return await _reconcile_locked_activation(
            locked, subscription_id, tx, expected_price_id
        )


async def _reconcile_locked_activation(
    user: User, subscription_id: str, tx: Prisma, expected_price_id: str | None
) -> PaidActivationResult | None:
    await query_raw_with_schema(
        'SELECT "id" FROM {schema_prefix}"SubscriptionTrial" WHERE "userId" = $1 FOR UPDATE',
        user.id,
        client=tx,
    )
    row = await tx.subscriptiontrial.find_unique(where={"userId": user.id})
    trial = (
        TrialState.from_db(row)
        if row and row.stripeSubscriptionId == subscription_id
        else None
    )
    current = dict(
        await stripe_call(stripe.Subscription.retrieve_async, subscription_id)
    )
    price_id = subscription_price_id(current)
    if not price_id or (
        expected_price_id is not None and price_id != expected_price_id
    ):
        return None
    target = await accepted_conversion_target(trial, current, tx) if trial else None
    if trial and trial.converted_at is None and target is None:
        return None
    tier = (
        target[0] if target else (await credit.build_price_to_tier_map()).get(price_id)
    )
    if tier not in PAID_CONVERSION_TIERS or user.subscriptionTier != tier:
        return None
    # Even an admin paid label cannot hide incomplete payment reconciliation.
    if not await publish_initial_pro_activation(user, current, price_id, tx, trial):
        return None
    return await _paid_activation_result(user.id, current, tx, trial)


async def _paid_activation_result(
    user_id: str, subscription: dict, tx: Prisma, trial: TrialState | None
) -> PaidActivationResult | None:
    activation = await tx.paidusageactivation.find_unique(where={"userId": user_id})
    if activation and activation.readyAt is None:
        return None
    invoice_id = INVOICE_REFERENCE.validate_python(
        subscription["latest_invoice"]
    ).invoice_id
    usage_reset = bool(
        activation
        and trial
        and trial.consumed_at is not None
        and trial.conversion_invoice_id == invoice_id
        and activation.stripeSubscriptionId == subscription["id"]
        and activation.stripeInvoiceId == invoice_id
    )
    if usage_reset:
        invoice = await _retrieve_invoice(invoice_id)
        usage_reset = qualifying_invoice(
            invoice,
            customer_id=subscription["customer"],
            subscription_id=subscription["id"],
            price_id=subscription_price_id(subscription) or "",
            trial_end=subscription.get("trial_end"),
        )
    return PaidActivationResult(
        invoice_id=invoice_id,
        usage_reset=usage_reset,
        activation_id=activation.id if activation and usage_reset else None,
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


async def current_invoice_is_settled(
    subscription: dict, *, allow_proration: bool = False
) -> bool:
    return (
        await _current_settled_invoice(subscription, allow_proration=allow_proration)
        is not None
    )


async def _current_settled_invoice(
    subscription: dict, *, allow_proration: bool = False
) -> dict | None:
    latest = subscription.get("latest_invoice")
    if not latest or subscription.get("status") != "active":
        return None
    invoice_id = INVOICE_REFERENCE.validate_python(latest).invoice_id
    invoice = await _retrieve_invoice(invoice_id)
    settled = (
        (
            settled_recurring_invoice(invoice)
            or (
                allow_proration
                and settled_proration_invoice(
                    invoice, subscription_price_id(subscription)
                )
            )
        )
        and invoice.get("customer") == subscription.get("customer")
        and invoice_subscription(invoice) == subscription.get("id")
        and (
            subscription.get("trial_end") is None
            or invoice.get("created", 0) >= subscription["trial_end"]
        )
    )
    return invoice if settled else None


async def _retrieve_invoice(invoice_id: str) -> dict:
    invoice = dict(await stripe_call(stripe.Invoice.retrieve_async, invoice_id))
    if invoice.get("lines", {}).get("has_more"):
        lines = await stripe_call(
            stripe.Invoice.list_lines_async, invoice_id, limit=100
        )
        invoice["lines"] = {
            "data": [dict(line) async for line in stripe_list_items(lines)],
            "has_more": False,
        }
    return invoice


async def publish_initial_pro_activation(
    user: User,
    subscription: dict,
    price_id: str,
    tx: Prisma,
    trial: TrialState | None = None,
) -> bool:
    """Prove paid access; only an initial consumed trial conversion may reset.

    The historical name and policy ID remain compatible with existing rows.
    Call under User then Trial row locks when a trial enrollment is supplied.
    """
    verify_subscription_owner(user, subscription)
    if subscription_price_id(subscription) != price_id:
        return False
    current_invoice = await _current_settled_invoice(subscription, allow_proration=True)
    if current_invoice is None:
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
    if (trial is None or trial.converted_at is not None) and settled_recurring_invoice(
        current_invoice
    ):
        return True
    first = await first_settled_invoice(user.stripeCustomerId or "")
    if first is None:
        return False
    # Paid signups and already-converted trials retain their usage. In
    # particular, replaying a legacy trial's original invoice cannot mint a
    # generation merely because no activation row was recorded at that time.
    if trial is None or trial.converted_at is not None:
        return True
    if not owns_consumed_trial(user, subscription, trial):
        return False
    if await accepted_conversion_target(trial, subscription, tx) is None:
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
            bool(prices)
            and None not in prices
            and (
                price_id not in prices
                or first.get("created", 0) < subscription["trial_end"]
            )
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
