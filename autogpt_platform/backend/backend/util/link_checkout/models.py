"""Data that crosses the private checkout's boundaries.

What the agent supplies (``CheckoutPlan``) and what the worker returns
(``WorkerReceipt``) are strict: extra fields are refused, so a script or a card
value can never ride along, and validation errors never echo their input.
Link's responses (``SpendRequest``) are read leniently. Link adds fields and
statuses over time, and a status this code does not know must read as "not
paid" rather than fail the status check that reconciles a payment.
"""

from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

SPEND_REQUEST_ID = r"^lsrq_[A-Za-z0-9_-]+$"
CHECKOUT_ID = r"^[a-f0-9]{32}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


PaymentRole = Literal["number", "cvc", "expiry", "exp_month", "exp_year", "submit"]
Selector = Annotated[str, Field(min_length=1, max_length=250)]


class FieldTarget(StrictModel):
    """One payment control: a selector, inside a frame when ``frame_url`` is set."""

    selector: Selector
    frame_url: str = Field(default="", max_length=2048)


class CheckoutPlan(StrictModel):
    """What the agent supplies to prepare a checkout. Selectors name empty
    fields in the open tab; the card itself never appears here."""

    credentials_id: str = Field(min_length=1, max_length=100)
    payment_method_id: str = Field(pattern=r"^csmrpd_[A-Za-z0-9_-]+$")
    merchant_name: str = Field(min_length=1, max_length=100)
    checkout_url: str = Field(max_length=2048)
    amount: int = Field(ge=1, le=50000, strict=True)
    currency: str = Field(default="usd", pattern=r"^[a-z]{3}$")
    context: str = Field(min_length=100, max_length=2000)
    number: Selector
    cvc: Selector
    expiry: Selector | None = None
    exp_month: Selector | None = None
    exp_year: Selector | None = None
    submit: Selector
    # Fields inside an iframe: role -> the frame's exact URL.
    frame_urls: dict[PaymentRole, Annotated[str, Field(max_length=2048)]] = Field(
        default_factory=dict
    )
    test_mode: bool = True

    def payment_fields(self) -> dict[PaymentRole, FieldTarget | None]:
        selectors: dict[PaymentRole, str | None] = {
            "number": self.number,
            "cvc": self.cvc,
            "expiry": self.expiry,
            "exp_month": self.exp_month,
            "exp_year": self.exp_year,
            "submit": self.submit,
        }
        return {
            role: (
                FieldTarget(selector=selector, frame_url=self.frame_urls.get(role, ""))
                if selector
                else None
            )
            for role, selector in selectors.items()
        }

    def merchant_url(self) -> str:
        """The page Link shows the customer. Query and fragment are dropped:
        a checkout URL often carries a cart or session token."""
        parts = urlsplit(self.checkout_url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @field_validator("checkout_url")
    @classmethod
    def https_checkout(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("Checkout requires an HTTPS merchant URL")
        return value

    @model_validator(mode="after")
    def expiry_fields(self):
        combined = self.expiry is not None
        separate = self.exp_month is not None and self.exp_year is not None
        if combined == separate or (combined and (self.exp_month or self.exp_year)):
            raise ValueError("Provide combined expiry or both month and year fields")
        targets = [t for t in self.payment_fields().values() if t is not None]
        if len({(t.frame_url, t.selector) for t in targets}) != len(targets):
            raise ValueError("Payment fields and submit must be distinct")
        return self


class BoundField(StrictModel):
    role: str
    target_id: str
    frame_id: str
    loader_id: str
    backend_node_id: int


class BrowserBinding(StrictModel):
    endpoint: str
    target_id: str
    url: str
    fields: list[BoundField] = Field(default_factory=list)


ApprovalMode = Literal["link", "in_app"]


class CheckoutIntent(StrictModel):
    id: str = Field(pattern=CHECKOUT_ID)
    user_id: str
    session_id: str
    # Unset while an in-chat approval is pending: the spend request is only
    # created, already approved, once the customer approves in AutoGPT.
    spend_request_id: str | None = Field(default=None, pattern=SPEND_REQUEST_ID)
    approval_url: str = ""
    approval_mode: ApprovalMode = "link"
    expires_at: float
    plan: CheckoutPlan
    browser: BrowserBinding
    attempted: bool = False


class Card(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    number: SecretStr
    cvc: SecretStr
    exp_month: int = Field(ge=1, le=12)
    exp_year: int = Field(ge=2000, le=2100)
    valid_until: datetime | None = None


class NextAction(BaseModel):
    # Unbounded on the way in, bounded where used: an overlong field must not
    # make the whole status unreadable.
    model_config = ConfigDict(hide_input_in_errors=True)
    type: str = ""
    resolution: str = ""
    display_message: str = ""
    action_url: str | None = None


class ActionRequired(BaseModel):
    next_action: NextAction | None = None


class StatusDetails(BaseModel):
    requires_action: ActionRequired | None = None


class SpendRequest(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    id: str = Field(pattern=SPEND_REQUEST_ID)
    status: str
    merchant_url: str | None = None
    amount: int | None = None
    currency: str | None = None
    approval_url: str | None = None
    card: Card | None = Field(default=None, exclude=True, repr=False)
    status_details: StatusDetails | None = None


class ApprovalDetails(StrictModel):
    """Evidence of a customer's approval collected in AutoGPT, sent with a
    delegated spend request (Link's ``approval_details``)."""

    approved_at: int
    approval_method: Literal["click"] = "click"
    app_name: str = "AutoGPT"
    external_user_id: str = Field(min_length=1, max_length=100)
    external_session_id: str = Field(min_length=1, max_length=100)
    agent_log_id: str = Field(pattern=CHECKOUT_ID)
    device_type: Literal["web", "mobile"] = "web"
    ip_address: str | None = Field(default=None, max_length=64)
    user_agent: str | None = Field(default=None, max_length=512)


class WorkerJob(StrictModel):
    action: Literal["create", "create_delegated", "status", "cancel", "pay"] = "pay"
    intent: CheckoutIntent
    access_token: SecretStr
    approval: ApprovalDetails | None = None


class WorkerReceipt(StrictModel):
    status: Literal["submitted", "not_submitted", "outcome_unknown"]
    browser_closed: bool = False
    paid: Literal[False] = False


class WorkerResult(StrictModel):
    receipt: WorkerReceipt | None = None
    spend: SpendRequest | None = None
    # `link_rejected` is Link's definite refusal (a 4xx), which can safely fall
    # back to another route; `link_duplicate` is the refusal of a request that
    # matches one still open, which no other route gets past. Anything else
    # leaves the outcome unknown.
    error: (
        Literal["private_checkout_failed", "link_rejected", "link_duplicate"] | None
    ) = None


class CompleteCheckout(StrictModel):
    checkout_id: str = Field(pattern=CHECKOUT_ID)
