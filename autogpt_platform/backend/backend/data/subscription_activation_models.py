"""Public contract and persisted terms for explicitly ending a trial on Pro or Max."""

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field, Json, field_validator

ActivationPlan = Literal["PRO", "MAX"]

ActivationStatus = Literal[
    "confirmation_required",
    "payment_required",
    "action_required",
    "processing",
    "ready",
    "not_applicable",
    "failed",
]


class RenewalDiscount(BaseModel):
    amount_off: int | None = None
    percent_off: float | None = None
    currency: str | None = None
    duration: str
    duration_in_months: int | None = None
    ends_at: int | None = None


class RenewalTaxRate(BaseModel):
    display_name: str
    percentage: float
    inclusive: bool
    country: str | None = None
    state: str | None = None


class RenewalTax(BaseModel):
    automatic: bool = False
    price_tax_behavior: str = "unspecified"
    rates: list[RenewalTaxRate] = Field(default_factory=list)


class ActivationTerms(BaseModel):
    plan: ActivationPlan = "PRO"
    price_id: str
    accepted_offer_token: str
    amount_due: int = Field(ge=0)
    currency: str
    billing_interval: Literal["month", "year"]
    billing_interval_count: int = 1
    renewal_unit_amount: int = Field(ge=0)
    renewal_discounts: list[RenewalDiscount] = Field(default_factory=list)
    renewal_tax: RenewalTax = Field(default_factory=RenewalTax)
    charge_timing: Literal["on_confirmation"] = "on_confirmation"
    renewal_terms: str
    expires_at: datetime

    @property
    def token(self) -> str:
        return sha256(self.model_dump_json().encode()).hexdigest()

    def same_charge_as(self, other: "ActivationTerms") -> bool:
        return self.model_dump(exclude={"expires_at"}) == other.model_dump(
            exclude={"expires_at"}
        )


class ActivationConfirmRequest(BaseModel):
    confirmed: Literal[True]
    terms_token: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("confirmed", mode="before")
    @classmethod
    def require_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Explicit confirmation is required")
        return value


class ActivationPreviewRequest(BaseModel):
    plan: ActivationPlan | None = None
    return_to: str = "/settings/billing"

    @field_validator("return_to")
    @classmethod
    def relative_destination(cls, value: str) -> str:
        if (
            not value.startswith("/")
            or value.startswith("//")
            or "\\" in value
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError("return_to must be an application-relative path")
        return value


class PaidActivationResult(BaseModel):
    invoice_id: str
    usage_reset: bool
    activation_id: str | None = None


class ActivationResponse(BaseModel):
    id: str | None = None
    status: ActivationStatus
    terms: ActivationTerms | None = None
    terms_token: str | None = None
    return_to: str = "/settings/billing"
    invoice_id: str | None = None
    hosted_invoice_url: str | None = None
    retry_after_seconds: int | None = None
    error_code: str | None = None
    usage_reset: bool = False
    activation_id: str | None = None


class ActivationAttempt(BaseModel):
    id: str
    user_id: str
    subscription_id: str
    customer_id: str
    terms: ActivationTerms | Json[ActivationTerms]
    return_to: str
    confirmed_at: datetime | None
    invoice_id: str | None = None

    def response(self, status: ActivationStatus, **kwargs) -> ActivationResponse:
        return ActivationResponse(
            id=self.id,
            status=status,
            terms=self.terms,
            terms_token=self.terms.token,
            return_to=self.return_to,
            invoice_id=self.invoice_id,
            **kwargs,
        )


class ActivationUnavailable(ValueError):
    """The user must refresh terms, or has no eligible owned trial."""


class ActivationNotFound(ValueError):
    """No activation owned by this authenticated user."""


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


class CouponProducts(BaseModel):
    products: list[str]


class Coupon(BaseModel):
    applies_to: CouponProducts | None = None
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
    product: str | None = None
    unit_amount: int
    currency: str
    recurring: RecurringPrice
    tax_behavior: str = "unspecified"


class ActivationItem(BaseModel):
    id: str | None = None
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
