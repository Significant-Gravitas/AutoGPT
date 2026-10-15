"""The fields of a Stripe subscription the lifecycle mapping reads.

Read from a subscription list call with ``data.discounts`` and
``data.customer`` expanded (``LIST_EXPAND``), so a coupon on the subscription
or on the customer is visible without a call per subscription. Without the
expansion a discount is only an ID, and is not read as a coupon.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Pass as ``expand=`` to ``stripe.Subscription.list``.
LIST_EXPAND = ["data.discounts", "data.customer"]

_EPOCH = datetime.min.replace(tzinfo=UTC)


class DiscountFacts(BaseModel):
    """A Stripe discount: its coupon's reduction, and when it stops."""

    percent_off: float | None = None
    amount_off: int | None = None
    end: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_stripe(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "coupon" not in data:
            return data
        coupon = data.get("coupon") or {}
        return {
            "percent_off": coupon.get("percent_off"),
            "amount_off": coupon.get("amount_off"),
            "end": data.get("end"),
        }

    def in_force_at(self, moment: datetime) -> bool:
        return self.end is None or utc(self.end) > moment


class StripeSubscriptionFacts(BaseModel):
    """Validates straight from a Stripe object or dict; Stripe's unix
    timestamps become UTC datetimes."""

    id: str
    customer: str | None = None
    status: str
    cancel_at_period_end: bool = False
    start_date: datetime | None = None
    created: datetime | None = None
    canceled_at: datetime | None = None
    ended_at: datetime | None = None
    trial_start: datetime | None = None
    trial_end: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    # Stripe's ``cancellation_details``: ``cancellation_requested`` for any
    # cancel made through the API, ``payment_failed`` when Stripe gave up on a
    # payment. Our own cancels carry a comment.
    cancellation_reason: str | None = None
    cancellation_comment: str | None = None
    # The subscription's own discounts, and the customer's, which applies
    # only to a subscription without any of its own.
    discounts: list[DiscountFacts] = Field(default_factory=list)
    customer_discount: DiscountFacts | None = None
    # The price of one billing period before any discount, when every item
    # has a flat price.
    gross_amount: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_stripe(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        customer = values.get("customer")
        if isinstance(customer, Mapping):
            values["customer"] = customer.get("id")
            values.setdefault("customer_discount", customer.get("discount"))
        details = values.get("cancellation_details") or {}
        values.setdefault("cancellation_reason", details.get("reason"))
        values.setdefault("cancellation_comment", details.get("comment"))
        # Unexpanded, a discount is only its ID.
        values["discounts"] = [
            d for d in values.get("discounts") or [] if isinstance(d, Mapping)
        ]
        items = (values.get("items") or {}).get("data")
        if items is not None:
            values.setdefault("gross_amount", _gross_amount(items))
        return values

    @property
    def started_at(self) -> datetime | None:
        return self.start_date or self.created

    @property
    def trial_enrollment_id(self) -> str | None:
        return self.metadata.get("trial_enrollment_id") or None

    @property
    def cancel_requested(self) -> bool:
        """The subscriber asked for it to end: set to cancel at the period
        end, or canceled through the API with no comment of ours. Our own
        cancels (after a failed payment, of a rejected trial, of a replaced
        plan) carry a comment, and Stripe ending it after a failed payment
        says ``payment_failed``."""
        return self.cancel_at_period_end or (
            self.cancellation_reason == "cancellation_requested"
            and not self.cancellation_comment
        )

    def free_at(self, moment: datetime) -> bool:
        """Its discounts in force at ``moment`` take the whole price, or the
        price itself is zero: nothing is charged. A partial discount, or an
        amount off a price that isn't flat, still leaves a charge."""
        if self.gross_amount == 0:
            return True
        discounts = self.discounts or (
            [self.customer_discount] if self.customer_discount else []
        )
        live = [d for d in discounts if d.in_force_at(moment)]
        if any((d.percent_off or 0) >= 100 for d in live):
            return True
        if self.gross_amount is None or not any(d.amount_off for d in live):
            return False
        remaining = float(self.gross_amount)
        for discount in live:
            remaining *= 1 - (discount.percent_off or 0) / 100
        return remaining - sum(d.amount_off or 0 for d in live) <= 0


def _gross_amount(items: Sequence[Mapping[str, Any]]) -> int | None:
    units = [(item.get("price") or {}).get("unit_amount") for item in items]
    flat = [unit for unit in units if unit is not None]
    if not units or len(flat) < len(units):
        return None
    return sum(unit * (item.get("quantity") or 1) for unit, item in zip(flat, items))


def utc(value: datetime | None) -> datetime:
    if value is None:
        return _EPOCH
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value
