"""Public contract and persisted terms for explicitly ending a Pro trial."""

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field, Json, field_validator

ActivationStatus = Literal[
    "confirmation_required",
    "payment_required",
    "action_required",
    "processing",
    "ready",
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
    plan: Literal["PRO"] = "PRO"
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
