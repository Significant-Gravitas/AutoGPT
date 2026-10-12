"""Public charge review and private ownership data for paid Max upgrades."""

from typing import Literal

from pydantic import BaseModel, Field


class SubscriptionUpgradePreviewRequest(BaseModel):
    tier: Literal["MAX"]
    billing_cycle: Literal["monthly", "yearly"] = "monthly"


class UpgradeProrationLine(BaseModel):
    description: str
    amount: int
    period_start: int
    period_end: int
    proration: bool


class SubscriptionUpgradePreview(BaseModel):
    preview_id: str
    amount_due: int
    tax_amount: int | None
    tax_included: bool
    adjustment_amount: int
    currency: str
    proration_date: int
    period_end: int
    expires_at: int
    recurring_amount: int
    recurring_tax_included: bool
    billing_cycle: Literal["monthly", "yearly"]
    lines: list[UpgradeProrationLine]


class StoredUpgradePreview(BaseModel):
    user_id: str
    subscription_id: str
    item_id: str
    current_price_id: str
    target_price_id: str
    quantity: int
    preview: SubscriptionUpgradePreview


class UpgradePreviewUnavailable(ValueError):
    pass


class PreviewPeriod(BaseModel):
    start: int
    end: int


class PreviewProrationParent(BaseModel):
    proration: bool = False
    subscription_item: str | None = None


class PreviewLineParent(BaseModel):
    subscription_item_details: PreviewProrationParent | None = None


class PreviewLinePrice(BaseModel):
    id: str


class PreviewPriceDetails(BaseModel):
    price: str


class PreviewLinePricing(BaseModel):
    price_details: PreviewPriceDetails | None = None


class PreviewLine(BaseModel):
    description: str | None = None
    amount: int
    period: PreviewPeriod
    proration: bool = False
    parent: PreviewLineParent | None = None
    subscription_item: str | None = None
    price: PreviewLinePrice | None = None
    pricing: PreviewLinePricing | None = None

    @property
    def subscription_item_id(self) -> str | None:
        if self.parent and self.parent.subscription_item_details:
            return self.parent.subscription_item_details.subscription_item
        return self.subscription_item

    @property
    def price_id(self) -> str | None:
        if self.pricing and self.pricing.price_details:
            return self.pricing.price_details.price
        return self.price.id if self.price else None

    @property
    def is_proration(self) -> bool:
        return self.proration or bool(
            self.parent
            and self.parent.subscription_item_details
            and self.parent.subscription_item_details.proration
        )


class PreviewLines(BaseModel):
    data: list[PreviewLine]
    has_more: bool = False


class PreviewTax(BaseModel):
    amount: int
    inclusive: bool = False
    tax_behavior: str | None = None

    @property
    def is_inclusive(self) -> bool:
        return self.inclusive or self.tax_behavior == "inclusive"


class UpgradeInvoicePreview(BaseModel):
    customer: str
    currency: str
    amount_due: int
    total: int
    total_tax_amounts: list[PreviewTax] | None = None
    total_taxes: list[PreviewTax] | None = None
    lines: PreviewLines


class UpgradeRecurringPrice(BaseModel):
    interval: Literal["month", "year"]
    interval_count: int = 1


class UpgradePrice(BaseModel):
    id: str
    currency: str
    unit_amount: int
    tax_behavior: str = "unspecified"
    recurring: UpgradeRecurringPrice


class UpgradeItemPrice(BaseModel):
    id: str
    recurring: UpgradeRecurringPrice | None = None


class UpgradeSubscriptionItem(BaseModel):
    id: str
    price: UpgradeItemPrice
    quantity: int = 1
    current_period_end: int | None = None


class UpgradeSubscriptionItems(BaseModel):
    data: list[UpgradeSubscriptionItem]
    has_more: bool = False


class UpgradeSubscription(BaseModel):
    id: str
    customer: str
    status: str
    current_period_end: int | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    items: UpgradeSubscriptionItems
