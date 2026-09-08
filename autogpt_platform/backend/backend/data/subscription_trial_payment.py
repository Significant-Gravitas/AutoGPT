"""Read the effective customer billing card without changing payment defaults."""

from datetime import datetime

import stripe
from pydantic import BaseModel, Field

from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial_config import AcceptedTrialOffer


class Card(BaseModel):
    exp_month: int
    exp_year: int
    fingerprint: str | None = None


class PaymentMethod(BaseModel):
    id: str
    type: str
    card: Card | None = None


class CustomerInvoiceSettings(BaseModel):
    default_payment_method: PaymentMethod | None = None


class CustomerSnapshot(BaseModel):
    id: str
    deleted: bool = False
    invoice_settings: CustomerInvoiceSettings | None = None


async def get_customer_default_payment_method(customer_id: str) -> PaymentMethod | None:
    customer = CustomerSnapshot.model_validate(
        await stripe_call(
            stripe.Customer.retrieve_async,
            customer_id,
            expand=["invoice_settings.default_payment_method"],
        )
    )
    if customer.id != customer_id:
        raise ValueError("Stripe customer ownership does not match the enrollment")
    if customer.deleted or customer.invoice_settings is None:
        return None
    return customer.invoice_settings.default_payment_method


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


class CancellationDetails(BaseModel):
    comment: str | None = None


class SubscriptionSnapshot(BaseModel):
    id: str
    customer: str
    status: str
    metadata: dict[str, str] = Field(default_factory=dict)
    trial_start: int | None = None
    trial_end: int | None = None
    ended_at: int | None = None
    cancel_at_period_end: bool = False
    cancellation_details: CancellationDetails | None = None
    default_payment_method: PaymentMethod | None = None
    default_source: str | dict | None = None
    customer_default_payment_method: PaymentMethod | None = None
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
        method = self.effective_payment_method()
        return bool(
            method
            and method.type == "card"
            and method.card
            and (method.card.exp_year, method.card.exp_month) >= (now.year, now.month)
            and not self.pending_setup_intent
        )

    def effective_payment_method(self) -> PaymentMethod | None:
        if self.default_payment_method is None and self.default_source is None:
            return self.customer_default_payment_method
        return self.default_payment_method
