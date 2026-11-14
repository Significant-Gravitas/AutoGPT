"""Requests and responses between the AutoGPT controller and a checkout broker.

The same models serve the in-process broker (self-hosted) and the remote one
(hosted, over mutual TLS), so both run one checkout state machine.
"""

import hashlib

from pydantic import Field, SecretStr, SerializationInfo, field_serializer

from backend.util.link_checkout.models import (
    CHECKOUT_ID,
    Amount,
    ApprovalDetails,
    ApprovalMode,
    CheckoutPlan,
    StrictModel,
    WorkerReceipt,
)


class Principal(StrictModel):
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)


def session_key(principal: Principal) -> str:
    """The broker's name for one user's chat: its browser and checkout state."""
    return hashlib.sha256(
        f"{principal.user_id}\0{principal.session_id}".encode()
    ).hexdigest()


class BrowserCommand(Principal):
    args: list[str] = Field(min_length=1, max_length=8)


class BrowserOutput(StrictModel):
    code: int
    output: str = ""
    error: str = ""
    image: str = ""


def reveal_token(
    value: SecretStr | None, info: SerializationInfo
) -> str | SecretStr | None:
    """The Link token crosses to a remote broker only when the sender asks
    for it explicitly (``context={"reveal_secrets": True}``); any other dump,
    a log line included, keeps it masked."""
    if value is not None and info.context and info.context.get("reveal_secrets"):
        return value.get_secret_value()
    return value


class CreateCheckout(Principal):
    plan: CheckoutPlan
    access_token: SecretStr
    approval_mode: ApprovalMode = "link"

    _serialize_token = field_serializer("access_token")(reveal_token)


class CheckoutReference(Principal):
    checkout_id: str = Field(pattern=CHECKOUT_ID)


class AuthorizedCheckout(CheckoutReference):
    access_token: SecretStr
    # The customer's in-chat approval, when the checkout is waiting for one,
    # and the revision of the purchase it was given for. Link's
    # approval_details names no amount, so this is what ties the approval to a
    # total: one raised since needs an approval of its own.
    approval: ApprovalDetails | None = None
    approved_revision: int | None = Field(default=None, ge=0)

    _serialize_token = field_serializer("access_token")(reveal_token)


class RaiseCheckout(CheckoutReference):
    access_token: SecretStr
    amount: Amount
    # Where the customer approves the new total while nothing exists in Link
    # yet; a request already in Link is always raised and re-approved there.
    approval_mode: ApprovalMode = "link"

    _serialize_token = field_serializer("access_token")(reveal_token)


class ResetCheckout(CheckoutReference):
    # Cancels the Link request of a checkout never attempted. Without one (no
    # usable Link connection) that request is left to expire.
    access_token: SecretStr | None = None

    _serialize_token = field_serializer("access_token")(reveal_token)


class CheckoutView(StrictModel):
    checkout_id: str
    spend_request_id: str | None = None
    credentials_id: str
    payment_method_id: str = ""
    merchant_name: str
    # The page the card goes to, without query or fragment (``merchant_url``).
    merchant_url: str
    amount: int
    currency: str
    test_mode: bool
    approval_mode: str = "link"
    approval_url: str = ""
    action_url: str = ""
    action_message: str = ""
    resolution: str = ""
    status: str
    message: str
    paid: bool = False
    attempted: bool = False
    expires_at: float = 0
    # The purchase's revision: bumped each time its total is raised.
    revision: int = 0
    receipt: WorkerReceipt | None = None
