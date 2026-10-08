"""Live owned subscription and invoice previews for explicit trial conversion."""

from datetime import UTC, datetime, timedelta
from typing import TypedDict

import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data import credit
from backend.data.stripe_client import stripe_call
from backend.data.subscription_activation_models import (
    ActivationItem,
    ActivationPlan,
    ActivationPrice,
    ActivationTerms,
    ActivationUnavailable,
    BillingInvoice,
    BillingSubscription,
    InvoicePreview,
    RenewalDiscount,
    RenewalTax,
)
from backend.data.subscription_trial import TrialState, get_subscription_trial


class SubscriptionItemsUpdate(TypedDict, total=False):
    items: list[stripe.Subscription.ModifyParamsItem]


async def owned_subscription(user_id: str, subscription_id: str) -> BillingSubscription:
    user = await User.prisma().find_unique_or_raise(where={"id": user_id})
    sub = BillingSubscription.model_validate(
        await stripe_call(
            stripe.Subscription.retrieve_async,
            subscription_id,
            expand=["latest_invoice"],
        )
    )
    if (
        sub.id != subscription_id
        or sub.customer != user.stripeCustomerId
        or sub.metadata.get("user_id") != user_id
        or (sub.latest_invoice and sub.latest_invoice.customer != sub.customer)
    ):
        raise ActivationUnavailable("Subscription ownership could not be established")
    return sub


async def is_paid_activation_subscription(
    user_id: str, sub: BillingSubscription, accepted_price_id: str | None = None
) -> bool | None:
    if sub.items.has_more or len(sub.items.data) != 1:
        return None
    item = sub.items.data[0]
    if item.quantity != 1:
        return None
    if accepted_price_id == item.price.id:
        return True
    trial = await get_subscription_trial(user_id)
    if (
        trial
        and trial.subscription_id == sub.id
        and trial.offer.tier in ("PRO", "MAX")
        and trial.offer.price_id == item.price.id
    ):
        return True
    tier = (await credit.build_price_to_tier_map()).get(item.price.id)
    return (
        tier in (SubscriptionTier.PRO, SubscriptionTier.MAX)
        if tier is not None
        else None
    )


async def conversion_trial(user_id: str) -> tuple[TrialState, BillingSubscription]:
    trial = await get_subscription_trial(user_id)
    if (
        trial is None
        or not trial.subscription_id
        or not trial.consumed_at
        or trial.converted_at
    ):
        raise ActivationUnavailable("No unconverted trial is available")
    sub = await owned_subscription(user_id, trial.subscription_id)
    if (
        sub.status != "trialing"
        or sub.cancel_at_period_end
        or sub.metadata.get("trial_enrollment_id") != trial.id
        or sub.customer != trial.customer_id
    ):
        raise ActivationUnavailable("The trial has ended or is no longer available")
    return trial, sub


async def quote_terms(
    trial: TrialState,
    sub: BillingSubscription,
    plan: ActivationPlan | None = None,
    *,
    accepted_price_id: str | None = None,
) -> ActivationTerms:
    item = accepted_trial_item(trial, sub)
    target = plan or trial.offer.tier
    if target not in ("PRO", "MAX"):
        raise ActivationUnavailable("The trial must convert to Pro or Max")
    price = item.price
    if target != trial.offer.tier:
        require_plan_change_window(sub)
        price = await alternate_price(trial, target, accepted_price_id)
    updates = subscription_item_update(sub, price.id)
    raw_preview = await stripe_call(
        stripe.Invoice.create_preview_async,
        customer=sub.customer,
        subscription=sub.id,
        subscription_details={
            "trial_end": "now",
            "proration_behavior": "none",
            **updates,
        },
        expand=["discounts.coupon.applies_to"],
    )
    preview = InvoicePreview.model_validate(raw_preview)
    if preview.customer != sub.customer or preview.currency != price.currency:
        raise ActivationUnavailable(
            "The invoice preview does not match the subscription"
        )
    return _quoted_terms(trial, sub, target, price, preview)


def _quoted_terms(
    trial: TrialState,
    sub: BillingSubscription,
    target: ActivationPlan,
    price: ActivationPrice,
    preview: InvoicePreview,
) -> ActivationTerms:
    return ActivationTerms(
        plan=target,
        price_id=price.id,
        accepted_offer_token=trial.offer.token,
        amount_due=preview.amount_due,
        currency=price.currency,
        billing_interval=price.recurring.interval,
        billing_interval_count=price.recurring.interval_count,
        renewal_unit_amount=price.unit_amount,
        renewal_discounts=_applicable_discounts(preview, price),
        renewal_tax=RenewalTax(
            automatic=sub.automatic_tax.enabled,
            price_tax_behavior=price.tax_behavior,
            rates=sub.default_tax_rates,
        ),
        renewal_terms=(
            f"Your {target.title()} subscription renews automatically every "
            f"{price.recurring.interval} at the "
            "displayed recurring price. The displayed discount terms and applicable "
            "taxes apply. Your paid billing period starts when you confirm and end "
            "the trial. Cancel before the next renewal to avoid the next charge."
        ),
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )


def _applicable_discounts(
    preview: InvoicePreview, price: ActivationPrice
) -> list[RenewalDiscount]:
    if (
        any(discount.coupon.applies_to for discount in preview.discounts)
        and not price.product
    ):
        raise ActivationUnavailable("Discount applicability could not be established")
    return [
        RenewalDiscount(**discount.coupon.model_dump(), ends_at=discount.end)
        for discount in preview.discounts
        if discount.coupon.applies_to is None
        or price.product in discount.coupon.applies_to.products
    ]


def accepted_trial_item(trial: TrialState, sub: BillingSubscription) -> ActivationItem:
    if sub.items.has_more or len(sub.items.data) != 1:
        raise ActivationUnavailable("The subscription must retain its accepted plan")
    item = sub.items.data[0]
    price = item.price
    interval = "month" if trial.offer.billing_cycle == "monthly" else "year"
    if (
        item.quantity != 1
        or price.id != trial.offer.price_id
        or price.unit_amount != trial.offer.unit_amount
        or price.currency != trial.offer.currency
        or price.recurring.interval != interval
        or price.recurring.interval_count != 1
    ):
        raise ActivationUnavailable("The subscription no longer matches accepted terms")
    return item


async def alternate_price(
    trial: TrialState, plan: ActivationPlan, accepted_price_id: str | None
) -> ActivationPrice:
    price_id = accepted_price_id or await credit.get_subscription_price_id(
        SubscriptionTier(plan), trial.offer.billing_cycle
    )
    if not price_id or price_id == trial.offer.price_id:
        raise ActivationUnavailable("The selected plan price is unavailable")
    price = ActivationPrice.model_validate(
        await stripe_call(stripe.Price.retrieve_async, price_id)
    )
    interval = "month" if trial.offer.billing_cycle == "monthly" else "year"
    if (
        price.id != price_id
        or price.currency != trial.offer.currency
        or price.recurring.interval != interval
        or price.recurring.interval_count != 1
    ):
        raise ActivationUnavailable(
            "The selected price does not match the billing cycle"
        )
    return price


def subscription_item_update(
    sub: BillingSubscription, price_id: str
) -> SubscriptionItemsUpdate:
    item = sub.items.data[0]
    if item.price.id == price_id:
        return {}
    if not item.id:
        raise ActivationUnavailable("The subscription item could not be established")
    return {"items": [{"id": item.id, "price": price_id, "quantity": 1}]}


def require_plan_change_window(sub: BillingSubscription) -> None:
    if sub.trial_end is None or sub.trial_end <= datetime.now(UTC).timestamp() + 300:
        raise ActivationUnavailable(
            "The trial is ending. Recover its payment before changing plans."
        )


async def matches_trial_source(
    user_id: str, sub: BillingSubscription, terms: ActivationTerms
) -> bool:
    trial = await get_subscription_trial(user_id)
    if (
        trial is None
        or trial.subscription_id != sub.id
        or trial.customer_id != sub.customer
        or trial.offer.token != terms.accepted_offer_token
        or sub.metadata.get("trial_enrollment_id") != trial.id
        or not trial.consumed_at
        or trial.converted_at
    ):
        return False
    try:
        accepted_trial_item(trial, sub)
    except ActivationUnavailable:
        return False
    return True


async def invoice_payment_intent(invoice: BillingInvoice) -> str | None:
    if invoice.payment_intent:
        return invoice.payment_intent
    if invoice.payments is None:
        expanded = BillingInvoice.model_validate(
            await stripe_call(
                stripe.Invoice.retrieve_async,
                invoice.id,
                expand=["payments"],
            )
        )
        if expanded.id != invoice.id or expanded.customer != invoice.customer:
            raise ActivationUnavailable(
                "Invoice ownership changed during payment recovery"
            )
        invoice = expanded
    if invoice.payments is None or invoice.payments.has_more:
        raise ActivationUnavailable("Invoice payment state could not be established")
    defaults = [
        payment
        for payment in invoice.payments.data
        if payment.is_default
        and payment.invoice == invoice.id
        and payment.payment.type == "payment_intent"
    ]
    if len(defaults) != 1:
        return None
    return defaults[0].payment.payment_intent
