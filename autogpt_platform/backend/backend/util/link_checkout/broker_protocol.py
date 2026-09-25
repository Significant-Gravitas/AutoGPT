"""Requests and responses between the AutoGPT controller and a checkout broker.

The same models serve the in-process broker (self-hosted) and the remote one
(hosted, over mutual TLS), so both run one checkout state machine.
"""

import hashlib

from pydantic import Field, SecretStr, SerializationInfo, field_serializer

from backend.util.link_checkout.models import (
    CHECKOUT_ID,
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


def reveal_token(value: SecretStr, info: SerializationInfo) -> str | SecretStr:
    """The Link token crosses to a remote broker only when the sender asks
    for it explicitly (``context={"reveal_secrets": True}``); any other dump,
    a log line included, keeps it masked."""
    if info.context and info.context.get("reveal_secrets"):
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
    # The customer's in-chat approval, when the checkout is waiting for one.
    approval: ApprovalDetails | None = None

    _serialize_token = field_serializer("access_token")(reveal_token)


class CheckoutView(StrictModel):
    checkout_id: str
    spend_request_id: str | None = None
    credentials_id: str
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
    receipt: WorkerReceipt | None = None
