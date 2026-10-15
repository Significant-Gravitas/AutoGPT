"""The fields of a Stripe subscription the lifecycle mapping reads.

Read from a subscription list call with ``data.discounts``,
``data.customer`` and ``data.latest_invoice`` expanded (``LIST_EXPAND``), so a
coupon on the subscription or on the customer, and a first invoice that
charged nothing, are visible without a call per subscription. Without the
expansion a discount is only an ID, and is not read as a coupon.
"""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Pass as ``expand=`` to ``stripe.Subscription.list``.
LIST_EXPAND = ["data.discounts", "data.customer", "data.latest_invoice"]

_EPOCH = datetime.min.replace(tzinfo=UTC)

# The comments on our own cancels: after a failed payment
# (``autogpt:payment_failed``), of a rejected trial (``autogpt_trial:...``),
# of a replaced plan. A comment the customer wrote, as the billing portal
# asks for one, doesn't start with these.
OUR_CANCEL_COMMENT_PREFIXES = ("autogpt:", "autogpt_trial:")


class DiscountFacts(BaseModel):
    """A Stripe discount: its coupon's reduction, and when it starts and
    stops."""

    percent_off: float | None = None
    amount_off: int | None = None
    start: datetime | None = None
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
            "start": data.get("start"),
            "end": data.get("end"),
        }

    def in_force_at(self, moment: datetime) -> bool:
        # A customer's discount is read as it is now, so one that started
        # after ``moment`` (say, after an old subscription ended) didn't apply.
        # Its end counts as in force: a subscription that ends when its
        # repeating coupon does was free to the last moment.
        started = self.start is None or utc(self.start) <= moment
        return started and (self.end is None or utc(self.end) >= moment)


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
    # cancel made through the API or the billing portal, ``payment_failed``
    # when Stripe gave up on a payment. Our own cancels carry one of our
    # comments (``OUR_CANCEL_COMMENT_PREFIXES``).
    cancellation_reason: str | None = None
    cancellation_comment: str | None = None
    # The subscription's own discounts, and the customer's, which applies
    # only to a subscription without any of its own.
    discounts: list[DiscountFacts] = Field(default_factory=list)
    customer_discount: DiscountFacts | None = None
    # The price of one billing period before any discount, when every item
    # has a flat price.
    gross_amount: int | None = None
    # Its only invoice so far is the first one, and a discount took all of
    # it: a 100% coupon for one period, which Stripe takes off the
    # subscription once that invoice is made, so ``discounts`` no longer
    # shows it. A first invoice that is $0 without a discount (a trial, a
    # metered price with no usage yet) doesn't count.
    first_invoice_free: bool = False

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
        invoice = values.get("latest_invoice")
        if isinstance(invoice, Mapping):
            values.setdefault("first_invoice_free", _discounted_to_zero(invoice))
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
        end, or canceled on request without a comment of ours (one they
        wrote themselves still counts). Our own cancels (after a failed
        payment, of a rejected trial, of a replaced plan) carry our comment,
        and Stripe ending it after a failed payment says ``payment_failed``."""
        comment = self.cancellation_comment or ""
        return self.cancel_at_period_end or (
            self.cancellation_reason == "cancellation_requested"
            and not comment.startswith(OUR_CANCEL_COMMENT_PREFIXES)
        )

    def free_at(self, moment: datetime) -> bool:
        """Its discounts in force at ``moment`` take the whole price, the
        price itself is zero, or its only invoice charged nothing: nothing is
        charged. A partial discount, or an amount off a price that isn't
        flat, still leaves a charge. Stacked discounts apply one after the
        other, in their listed order, as Stripe applies them."""
        if self.gross_amount == 0 or self.first_invoice_free:
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
            remaining = max(remaining - (discount.amount_off or 0), 0.0)
        return remaining <= 0


def _discounted_to_zero(invoice: Mapping[str, Any]) -> bool:
    """The subscription's first invoice, brought to zero by a discount: one
    that took something off, not merely one attached to a $0 invoice."""
    return (
        invoice.get("billing_reason") == "subscription_create"
        and invoice.get("total") == 0
        and any(
            (amount or {}).get("amount")
            for amount in invoice.get("total_discount_amounts") or []
        )
    )


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
