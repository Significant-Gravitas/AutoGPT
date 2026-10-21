"""Read the effective customer billing card without changing payment defaults."""

from datetime import datetime

import stripe
from prisma.enums import SubscriptionTier
from pydantic import BaseModel, Field

from backend.data.stripe_client import stripe_call
from backend.data.subscription_trial_config import AcceptedTrialOffer

# A trial that ends on a plan other than the accepted one is moved onto that
# plan's price with these beside it; the reconcile grants no other price.
CONVERSION_PRICE_KEY = "trial_conversion_price_id"
CONVERSION_TIER_KEY = "trial_conversion_tier"
_CHOSEN_TIERS = frozenset(
    {
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    }
)


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


async def subscription_card_can_be_charged(subscription_id: str, now: datetime) -> bool:
    """Whether the trial's saved card is one Stripe can charge, by the same
    rule the reconcile grants trial access with."""
    snapshot = SubscriptionSnapshot.model_validate(
        await stripe_call(
            stripe.Subscription.retrieve_async,
            subscription_id,
            expand=["default_payment_method", "latest_invoice"],
        )
    )
    if snapshot.default_payment_method is None and snapshot.default_source is None:
        snapshot.customer_default_payment_method = (
            await get_customer_default_payment_method(snapshot.customer)
        )
    return snapshot.has_verified_card(now)


class Invoice(BaseModel):
    id: str
    status: str | None = None
    created: int
    billing_reason: str | None = None
    hosted_invoice_url: str | None = None


class Recurring(BaseModel):
    interval: str


class SubscriptionPrice(BaseModel):
    id: str
    unit_amount: int | None = None
    currency: str | None = None
    recurring: Recurring | None = None


class SubscriptionItem(BaseModel):
    price: SubscriptionPrice
    quantity: int | None = None


class SubscriptionItems(BaseModel):
    data: list[SubscriptionItem]
    has_more: bool = False


class CancellationDetails(BaseModel):
    comment: str | None = None


class ChosenPlan(BaseModel):
    tier: SubscriptionTier
    price: SubscriptionPrice


class SubscriptionPlanTerms(BaseModel):
    """The plan a trial subscription carries."""

    metadata: dict[str, str] = Field(default_factory=dict)
    items: SubscriptionItems | None = None

    def has_accepted_price(self, offer: AcceptedTrialOffer) -> bool:
        price = self._sole_price()
        return price is not None and price.id == offer.price_id

    def chosen_plan(self) -> ChosenPlan | None:
        """The plan the customer ended the trial on in place of the accepted
        one, while the subscription carries exactly that plan's price."""
        price = self._sole_price()
        if price is None or price.id != self.metadata.get(CONVERSION_PRICE_KEY):
            return None
        tier = self.metadata.get(CONVERSION_TIER_KEY)
        chosen = next((t for t in _CHOSEN_TIERS if t.value == tier), None)
        return ChosenPlan(tier=chosen, price=price) if chosen else None

    def _sole_price(self) -> SubscriptionPrice | None:
        if not self.items or self.items.has_more or len(self.items.data) != 1:
            return None
        item = self.items.data[0]
        return item.price if item.quantity == 1 else None


class SubscriptionSnapshot(SubscriptionPlanTerms):
    id: str
    customer: str
    status: str
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
