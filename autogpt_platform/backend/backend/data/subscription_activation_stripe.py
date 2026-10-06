"""Live owned subscription and invoice previews for explicit trial conversion."""

from datetime import UTC, datetime, timedelta
from typing import Literal

import stripe
from prisma.models import User
from pydantic import BaseModel, Field

from backend.data.stripe_client import stripe_call
from backend.data.subscription_activation_models import (
    ActivationTerms,
    ActivationUnavailable,
    RenewalDiscount,
    RenewalTax,
    RenewalTaxRate,
)
from backend.data.subscription_trial import TrialState, get_subscription_trial


class InvoicePaymentSource(BaseModel):
    type: str
    payment_intent: str | None = None


class InvoicePayment(BaseModel):
    invoice: str
    is_default: bool = False
    payment: InvoicePaymentSource


class InvoicePayments(BaseModel):
    data: list[InvoicePayment] = Field(default_factory=list)
    has_more: bool = False


class BillingInvoice(BaseModel):
    id: str
    customer: str
    status: str | None = None
    amount_remaining: int | None = None
    hosted_invoice_url: str | None = None
    payment_intent: str | None = None
    payments: InvoicePayments | None = None


class Coupon(BaseModel):
    amount_off: int | None = None
    percent_off: float | None = None
    currency: str | None = None
    duration: str
    duration_in_months: int | None = None


class Discount(BaseModel):
    coupon: Coupon
    end: int | None = None


class InvoicePreview(BaseModel):
    customer: str
    currency: str
    amount_due: int
    discounts: list[Discount] = Field(default_factory=list)


class AutomaticTax(BaseModel):
    enabled: bool = False


class RecurringPrice(BaseModel):
    interval: Literal["month", "year"]
    interval_count: int


class ActivationPrice(BaseModel):
    id: str
    unit_amount: int
    currency: str
    recurring: RecurringPrice
    tax_behavior: str = "unspecified"


class ActivationItem(BaseModel):
    price: ActivationPrice
    quantity: int


class ActivationItems(BaseModel):
    data: list[ActivationItem]
    has_more: bool = False


class BillingSubscription(BaseModel):
    id: str
    customer: str
    status: str
    metadata: dict[str, str] = Field(default_factory=dict)
    cancel_at_period_end: bool = False
    trial_end: int | None = None
    items: ActivationItems
    latest_invoice: BillingInvoice | None = None
    automatic_tax: AutomaticTax = Field(default_factory=AutomaticTax)
    default_tax_rates: list[RenewalTaxRate] = Field(default_factory=list)


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


async def conversion_trial(user_id: str) -> tuple[TrialState, BillingSubscription]:
    trial = await get_subscription_trial(user_id)
    if (
        trial is None
        or not trial.subscription_id
        or not trial.consumed_at
        or trial.converted_at
        or trial.offer.tier != "PRO"
    ):
        raise ActivationUnavailable("No unconverted Pro trial is available")
    sub = await owned_subscription(user_id, trial.subscription_id)
    if (
        sub.status != "trialing"
        or sub.cancel_at_period_end
        or sub.metadata.get("trial_enrollment_id") != trial.id
        or sub.customer != trial.customer_id
    ):
        raise ActivationUnavailable("The trial has ended or is no longer available")
    return trial, sub


async def quote_terms(trial: TrialState, sub: BillingSubscription) -> ActivationTerms:
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
    raw_preview = await stripe_call(
        stripe.Invoice.create_preview_async,
        customer=sub.customer,
        subscription=sub.id,
        subscription_details={"trial_end": "now", "proration_behavior": "none"},
        expand=["discounts"],
    )
    preview = InvoicePreview.model_validate(raw_preview)
    if preview.customer != sub.customer or preview.currency != price.currency:
        raise ActivationUnavailable(
            "The invoice preview does not match the subscription"
        )
    return ActivationTerms(
        price_id=price.id,
        accepted_offer_token=trial.offer.token,
        amount_due=preview.amount_due,
        currency=price.currency,
        billing_interval=price.recurring.interval,
        billing_interval_count=price.recurring.interval_count,
        renewal_unit_amount=price.unit_amount,
        renewal_discounts=[
            RenewalDiscount(**discount.coupon.model_dump(), ends_at=discount.end)
            for discount in preview.discounts
        ],
        renewal_tax=RenewalTax(
            automatic=sub.automatic_tax.enabled,
            price_tax_behavior=price.tax_behavior,
            rates=sub.default_tax_rates,
        ),
        renewal_terms=(
            f"Your Pro subscription renews automatically every {interval} at the "
            "displayed recurring price. The displayed discount terms and applicable "
            "taxes apply. Your paid billing period starts when you confirm and end "
            "the trial. Cancel before the next renewal to avoid the next charge."
        ),
        expires_at=datetime.now(UTC) + timedelta(minutes=15),
    )


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
