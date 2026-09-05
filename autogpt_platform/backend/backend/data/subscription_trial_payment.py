"""Read the effective customer billing card without changing payment defaults."""

import stripe
from pydantic import BaseModel

from backend.data.stripe_client import stripe_call


class Card(BaseModel):
    exp_month: int
    exp_year: int


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
