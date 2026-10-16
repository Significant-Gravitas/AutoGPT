"""The invoice fields used to disclose an early trial conversion."""

from datetime import UTC, datetime

from pydantic import BaseModel

from backend.data.subscription_activation_models import ActivationUnavailable
from backend.data.subscription_activation_models import (
    InvoicePreview as BaseInvoicePreview,
)


class InvoicePreview(BaseInvoicePreview):
    lines: "PreviewLines"
    total_tax_amounts: list["TaxAmount"] | None = None
    total_taxes: list["TaxAmount"] | None = None
    tax: int | None = None

    def paid_period(self, subscription_id: str, price_id: str):
        candidates = [
            line.period
            for line in self.lines.data
            if line.matches(subscription_id, price_id)
            and line.period.end > line.period.start
        ]
        if self.lines.has_more or not candidates:
            raise ActivationUnavailable("The paid period could not be established")
        period = max(candidates, key=lambda value: value.start)
        return (
            datetime.fromtimestamp(period.start, UTC),
            datetime.fromtimestamp(period.end, UTC),
        )

    def tax_total(self) -> int | None:
        taxes = self.total_taxes
        if taxes is None:
            taxes = self.total_tax_amounts
        return sum(tax.amount for tax in taxes) if taxes is not None else self.tax


class TaxAmount(BaseModel):
    amount: int


class Period(BaseModel):
    start: int
    end: int


class PriceReference(BaseModel):
    id: str


class PriceDetails(BaseModel):
    price: str


class Pricing(BaseModel):
    price_details: PriceDetails | None = None


class SubscriptionItemDetails(BaseModel):
    subscription: str
    proration: bool = False


class LineParent(BaseModel):
    type: str
    subscription_item_details: SubscriptionItemDetails | None = None


class PreviewLine(BaseModel):
    period: Period
    type: str | None = None
    subscription: str | None = None
    price: PriceReference | None = None
    proration: bool = False
    parent: LineParent | None = None
    pricing: Pricing | None = None

    def matches(self, subscription_id: str, price_id: str) -> bool:
        if self.parent and self.parent.type == "subscription_item_details":
            details = self.parent.subscription_item_details
            price = self.pricing.price_details if self.pricing else None
            return bool(
                details
                and price
                and not details.proration
                and details.subscription == subscription_id
                and price.price == price_id
            )
        return bool(
            self.type == "subscription"
            and not self.proration
            and self.subscription == subscription_id
            and self.price
            and self.price.id == price_id
        )


class PreviewLines(BaseModel):
    data: list[PreviewLine]
    has_more: bool = False


InvoicePreview.model_rebuild()
