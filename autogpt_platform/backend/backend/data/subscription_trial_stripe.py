"""Reconcile trial entitlements from current Stripe state, never event order."""

from datetime import UTC, datetime

import stripe
from prisma import Prisma
from prisma.enums import SubscriptionTier
from pydantic import BaseModel, Field

from backend.data.db import query_raw_with_schema, transaction
from backend.data.stripe_client import stripe_call, stripe_list_items
from backend.data.subscription_trial import TrialState, get_subscription_trial
from backend.data.subscription_trial_config import AcceptedTrialOffer


class Card(BaseModel):
    exp_month: int
    exp_year: int


class PaymentMethod(BaseModel):
    id: str
    type: str
    card: Card | None = None


class Invoice(BaseModel):
    id: str
    status: str | None = None
    created: int
    billing_reason: str | None = None


class SubscriptionPrice(BaseModel):
    id: str


class SubscriptionItem(BaseModel):
    price: SubscriptionPrice
    quantity: int | None = None


class SubscriptionItems(BaseModel):
    data: list[SubscriptionItem]
    has_more: bool = False


class SubscriptionSnapshot(BaseModel):
    id: str
    customer: str
    status: str
    metadata: dict[str, str] = Field(default_factory=dict)
    trial_start: int | None = None
    trial_end: int | None = None
    cancel_at_period_end: bool = False
    default_payment_method: PaymentMethod | None = None
    pending_setup_intent: str | dict | None = None
    latest_invoice: Invoice | None = None
    items: SubscriptionItems | None = None

    def has_accepted_price(self, offer: AcceptedTrialOffer) -> bool:
        return bool(
            self.items
            and not self.items.has_more
            and len(self.items.data) == 1
            and self.items.data[0].price.id == offer.price_id
            and self.items.data[0].quantity == 1
        )

    def has_verified_card(self, now: datetime) -> bool:
        method = self.default_payment_method
        return bool(
            method
            and method.type == "card"
            and method.card
            and (method.card.exp_year, method.card.exp_month) >= (now.year, now.month)
            and not self.pending_setup_intent
        )


async def reconcile_trial_subscription(
    user_id: str, subscription_id: str
) -> tuple[dict, SubscriptionTier | None] | None:
    trial = await get_subscription_trial(user_id)
    if trial is None:
        return None
    async with transaction() as tx:
        await query_raw_with_schema(
            'SELECT "id" FROM {schema_prefix}"SubscriptionTrial" '
            'WHERE "userId" = $1 FOR UPDATE',
            user_id,
            client=tx,
        )
        row = await tx.subscriptiontrial.find_unique_or_raise(where={"userId": user_id})
        return await _reconcile_locked(TrialState.from_db(row), subscription_id, tx)


async def _reconcile_locked(
    trial: TrialState, subscription_id: str, tx: Prisma
) -> tuple[dict, SubscriptionTier | None] | None:
    raw = await stripe_call(
        stripe.Subscription.retrieve_async,
        subscription_id,
        expand=["default_payment_method", "latest_invoice"],
    )
    snapshot = SubscriptionSnapshot.model_validate(raw)
    if snapshot.metadata.get("trial_enrollment_id") != trial.id:
        return None
    if (
        snapshot.customer != trial.customer_id
        or snapshot.metadata.get("user_id") != trial.user_id
    ):
        raise ValueError("Stripe trial ownership does not match the enrollment")
    if snapshot.metadata.get("trial_checkout_attempt") != str(trial.checkout_attempt):
        user = await tx.user.find_unique_or_raise(where={"id": trial.user_id})
        return dict(raw), user.subscriptionTier
    if trial.subscription_id and trial.subscription_id != snapshot.id:
        raise ValueError("A different subscription already consumed this trial")
    if trial.converted_at is None and not snapshot.has_accepted_price(trial.offer):
        raise ValueError(
            "Stripe items do not match the accepted trial price and quantity"
        )
    now = datetime.now(UTC)
    checkout_complete = trial.consumed_at is not None or await _completed_card_checkout(
        trial, snapshot.id, tx
    )
    tier = (
        trial_subscription_tier(trial, snapshot, now)
        if checkout_complete
        else SubscriptionTier.NO_TIER
    )
    await _save_snapshot(trial, snapshot, tier, now, tx, checkout_complete)
    if trial.converted_at:
        return dict(raw), None
    if tier == SubscriptionTier.NO_TIER:
        for status in ("active", "trialing"):
            others = await stripe_call(
                stripe.Subscription.list_async,
                customer=trial.customer_id,
                status=status,
                limit=100,
            )
            if any(sub.id != snapshot.id for sub in others.data):
                return dict(raw), None
    await tx.user.update_many(
        where={
            "id": trial.user_id,
            "subscriptionTier": {"not": SubscriptionTier.ENTERPRISE},
        },
        data={"subscriptionTier": tier},
    )
    return dict(raw), tier


async def _completed_card_checkout(
    trial: TrialState, subscription_id: str, tx: Prisma
) -> bool:
    if trial.checkout_session_id:
        session = await stripe_call(
            stripe.checkout.Session.retrieve_async, trial.checkout_session_id
        )
        return _checkout_proves_card_collection(trial, subscription_id, session)
    sessions = await stripe_call(
        stripe.checkout.Session.list_async, customer=trial.customer_id, limit=100
    )
    async for session in stripe_list_items(sessions):
        if (session.metadata or {}).get("trial_enrollment_id") != trial.id:
            continue
        if (session.metadata or {}).get("trial_checkout_attempt") != str(
            trial.checkout_attempt
        ):
            continue
        complete = _checkout_proves_card_collection(trial, subscription_id, session)
        await tx.subscriptiontrial.update(
            where={"id": trial.id}, data={"stripeCheckoutSessionId": session.id}
        )
        return complete
    return False


def _checkout_proves_card_collection(
    trial: TrialState, subscription_id: str, session: stripe.checkout.Session
) -> bool:
    if session.status != "complete":
        return False
    metadata = session.metadata or {}
    if (
        session.mode != "subscription"
        or session.customer != trial.customer_id
        or session.subscription != subscription_id
        or metadata.get("user_id") != trial.user_id
        or metadata.get("trial_enrollment_id") != trial.id
        or metadata.get("trial_checkout_attempt") != str(trial.checkout_attempt)
        or metadata.get("trial_offer_version") != trial.offer.version
        or session.payment_method_collection != "always"
        or session.payment_method_types != ["card"]
    ):
        raise ValueError("Checkout does not prove this enrollment's card collection")
    return True


def trial_subscription_tier(
    trial: TrialState, subscription: SubscriptionSnapshot, now: datetime
) -> SubscriptionTier:
    end = subscription.trial_end
    if end is None:
        return SubscriptionTier.NO_TIER
    if subscription.status == "trialing" and end > now.timestamp():
        if subscription.has_verified_card(now):
            return SubscriptionTier.TRIAL
        return SubscriptionTier.NO_TIER
    invoice = subscription.latest_invoice
    if (
        subscription.status == "active"
        and end <= now.timestamp()
        and invoice is not None
        and invoice.status == "paid"
        and invoice.created >= end
        and invoice.billing_reason != "subscription_create"
    ):
        return SubscriptionTier(trial.offer.tier)
    return SubscriptionTier.NO_TIER


async def _save_snapshot(
    trial: TrialState,
    snapshot: SubscriptionSnapshot,
    tier: SubscriptionTier,
    now: datetime,
    tx: Prisma,
    checkout_complete: bool,
) -> None:
    verified_at = (
        (trial.card_verified_at or now)
        if checkout_complete and snapshot.has_verified_card(now)
        else None
    )
    consumed_at = trial.consumed_at
    if checkout_complete:
        consumed_at = consumed_at or now
    converted_at = trial.converted_at
    conversion_invoice_id = trial.conversion_invoice_id
    if converted_at is None and tier.value == trial.offer.tier:
        if snapshot.latest_invoice is None:
            raise ValueError("Trial conversion requires a paid invoice")
        converted_at = now
        conversion_invoice_id = snapshot.latest_invoice.id
    await tx.subscriptiontrial.update(
        where={"userId": trial.user_id},
        data={
            "stripeSubscriptionId": snapshot.id,
            "status": (
                "checkout_pending"
                if not checkout_complete
                and snapshot.status
                in ("trialing", "incomplete", "canceled", "incomplete_expired")
                else snapshot.status
            ),
            "cardVerifiedAt": verified_at,
            "startedAt": _timestamp(snapshot.trial_start),
            "endsAt": _timestamp(snapshot.trial_end),
            "consumedAt": consumed_at,
            "convertedAt": converted_at,
            "stripeConversionInvoiceId": conversion_invoice_id,
            "cancelAtPeriodEnd": snapshot.cancel_at_period_end,
        },
    )


def _timestamp(value: int | None) -> datetime | None:
    return datetime.fromtimestamp(value, UTC) if value is not None else None
