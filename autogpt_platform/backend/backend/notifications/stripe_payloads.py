"""Typed views over the Stripe payloads the billing emails read.

Webhook payloads are not the same shape as an API response, and that
difference has already cost us one silently-undelivered email:

* **Nothing is expanded.** `invoice.payment_intent` and
  `invoice.default_payment_method` arrive as ID *strings*. Walking into them
  as if they were objects raises `AttributeError`, and in `on_payment_failed`
  that happens *after* the idempotency key is claimed — so Stripe's retry is
  deduped away and the customer never hears that their payment failed.
* **The shape drifts with the API version.** `PaymentIntent.charges` was
  replaced by `latest_charge` in 2022-11-15, and the account renders webhooks
  at whatever version its endpoint is pinned to, not at the library's version.

Both problems have the same answer: read Stripe payloads through models that
treat every nested field as optional-and-possibly-an-id, in one place, instead
of chained `.get()` calls spread across the formatting helpers.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _object_or_none(value: Any) -> dict | None:
    """A nested Stripe field, or None when absent or an unexpanded ID string."""
    return value if isinstance(value, dict) else None


class StripeCard(BaseModel):
    brand: str | None = None
    last4: str | None = None


class StripePaymentMethodDetails(BaseModel):
    card: StripeCard | None = None

    @field_validator("card", mode="before")
    @classmethod
    def _unwrap(cls, v: Any) -> Any:
        return _object_or_none(v)


class StripeCharge(BaseModel):
    payment_method_details: StripePaymentMethodDetails | None = None

    @field_validator("payment_method_details", mode="before")
    @classmethod
    def _unwrap(cls, v: Any) -> Any:
        return _object_or_none(v)


class StripePaymentIntent(BaseModel):
    """Both charge shapes, because which one is populated depends on the API
    version the webhook endpoint is pinned to."""

    latest_charge: StripeCharge | None = None
    charges: list[StripeCharge] = Field(default_factory=list)

    @field_validator("latest_charge", mode="before")
    @classmethod
    def _unwrap_charge(cls, v: Any) -> Any:
        return _object_or_none(v)

    @field_validator("charges", mode="before")
    @classmethod
    def _unwrap_charges(cls, v: Any) -> Any:
        data = (v or {}).get("data") if isinstance(v, dict) else v
        if not isinstance(data, list):
            return []
        return [c for c in data if isinstance(c, dict)]

    @property
    def card(self) -> StripeCard | None:
        for charge in (
            [self.latest_charge] if self.latest_charge else []
        ) + self.charges:
            details = charge.payment_method_details
            if details and details.card:
                return details.card
        return None


class StripePaymentMethod(BaseModel):
    card: StripeCard | None = None

    @field_validator("card", mode="before")
    @classmethod
    def _unwrap(cls, v: Any) -> Any:
        return _object_or_none(v)


class StripeInvoice(BaseModel):
    """The invoice fields the payment emails read.

    `model_config` is deliberately permissive: Stripe adds fields constantly
    and an unknown one must never fail a billing email.
    """

    id: str | None = None
    amount_due: int | None = None
    currency: str = "usd"
    attempt_count: int = 0
    next_payment_attempt: int | None = None
    period_end: int | None = None
    payment_intent: StripePaymentIntent | None = None
    default_payment_method: StripePaymentMethod | None = None

    @field_validator("payment_intent", "default_payment_method", mode="before")
    @classmethod
    def _unwrap(cls, v: Any) -> Any:
        return _object_or_none(v)

    @field_validator("currency", mode="before")
    @classmethod
    def _default_currency(cls, v: Any) -> Any:
        return v or "usd"

    @property
    def card(self) -> StripeCard | None:
        """Card details from wherever this payload happens to carry them."""
        if self.payment_intent and (card := self.payment_intent.card):
            return card
        if self.default_payment_method:
            return self.default_payment_method.card
        return None

    @classmethod
    def parse(cls, payload: dict) -> "StripeInvoice":
        return cls.model_validate(payload)
