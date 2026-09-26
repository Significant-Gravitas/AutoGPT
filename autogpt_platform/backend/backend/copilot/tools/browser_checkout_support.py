"""What the Link checkout tools share: availability, the caller, the Link
credential, and the chat card they return."""

import logging
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import ValidationError

from backend.copilot.credential_selection import selected_credentials
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import ErrorResponse, ResponseType, ToolResponseBase
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth.stripe_link_hosted import (
    STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED,
    is_hosted_link_credential,
)
from backend.util.link_checkout import engine
from backend.util.link_checkout.approval import ApprovalState, ApprovalView
from backend.util.link_checkout.broker_protocol import CheckoutView, Principal
from backend.util.link_checkout.config import hosted_checkout_requested
from backend.util.link_checkout.models import WorkerReceipt
from backend.util.link_checkout.refusals import (
    LINK_ACCOUNT_NOT_CHOSEN,
    LINK_NOT_CONNECTED,
    CheckoutRefused,
)
from backend.util.settings import BehaveAs, Settings

LINK_PAYMENT_SCOPE = "payment_methods.agentic"

# Written out rather than generated: the model sees plain selector strings,
# and ``CheckoutPlan`` enforces the patterns and bounds when the call arrives.
REQUEST_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "credentials_id": {
            "type": "string",
            "description": "Stripe Link credential to pay with. Leave it out to "
            "use the account connected in this chat.",
        },
        "payment_method_id": {
            "type": "string",
            "description": "Link payment method to pay with (csmrpd_...).",
        },
        "merchant_name": {"type": "string", "description": "Merchant the user pays."},
        "checkout_url": {
            "type": "string",
            "description": "Exact HTTPS URL of the open checkout tab.",
        },
        "amount": {
            "type": "integer",
            "description": "Final total in the currency's smallest unit (1250 = 12.50).",
        },
        "currency": {"type": "string", "description": "Lowercase ISO code; usd."},
        "context": {
            "type": "string",
            "description": "What is being bought and why, shown to the user "
            "(at least 100 characters).",
        },
        "number": {"type": "string", "description": "Selector of the card number."},
        "cvc": {"type": "string", "description": "Selector of the CVC."},
        "expiry": {
            "type": "string",
            "description": "Selector of a combined MM/YY field; else give "
            "exp_month and exp_year.",
        },
        "exp_month": {"type": "string", "description": "Selector of the month."},
        "exp_year": {"type": "string", "description": "Selector of the year."},
        "submit": {"type": "string", "description": "Selector of the pay button."},
        "frame_urls": {
            "type": "object",
            "description": "For fields inside an iframe: field name (number, cvc, "
            "expiry, exp_month, exp_year, submit) to the frame's exact URL.",
            "additionalProperties": {"type": "string"},
        },
        "test_mode": {
            "type": "boolean",
            "description": "Test payment with no charge. Default true.",
        },
    },
    "required": [
        "payment_method_id",
        "merchant_name",
        "checkout_url",
        "amount",
        "context",
        "number",
        "cvc",
        "submit",
    ],
}

CHECKOUT_ID_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "checkout_id": {
            "type": "string",
            "description": "checkout_id from tool:browser_request_link_payment.",
        }
    },
    "required": ["checkout_id"],
}

_credentials = IntegrationCredentialsManager()
_settings = Settings()
logger = logging.getLogger(__name__)


class CheckoutResponse(ToolResponseBase):
    type: ResponseType = ResponseType.BROWSER_CHECKOUT
    checkout_id: str
    spend_request_id: str | None = None
    merchant_name: str
    merchant_url: str
    amount: int
    currency: str
    test_mode: bool
    status: str
    # "in_app": the customer approves in this chat; "link": in Link.
    approval_mode: str = "link"
    # For in-chat approval: awaiting / approved / declined / expired.
    approval_state: ApprovalState | None = None
    approval_url: str = ""
    action_url: str = ""
    action_message: str = ""
    resolution: str = ""
    paid: bool = False
    attempted: bool = False
    expires_at: float = 0
    receipt: WorkerReceipt | None = None


def available() -> bool:
    """Whether the checkout tools exist here at all; ``principal_for`` then
    decides for the caller."""
    if not engine.enabled():
        return False
    if _settings.config.behave_as != BehaveAs.CLOUD:
        return True
    # Hosted spending needs every user's browser in their own broker and the
    # registered Link client, and an explicit opt-in on top of both.
    return (
        engine.remote()
        and STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED
        and hosted_checkout_requested()
    )


def principal_for(user_id: str | None, session: ChatSession) -> Principal:
    if (
        not user_id
        or user_id != session.user_id
        or not available()
        or not engine.active_for(user_id)
    ):
        raise ValueError("Private checkout unavailable")
    return Principal(user_id=user_id, session_id=session.session_id)


@asynccontextmanager
async def link_credentials(
    user_id: str, credentials_id: str
) -> AsyncIterator[OAuth2Credentials]:
    """A leased Link credential that can create spend requests."""
    lease = await _credentials.acquire_lease(user_id, credentials_id)
    try:
        await lease.validate()
        credentials = lease.credentials
        if credentials.type != "oauth2" or credentials.provider != "stripe_link":
            raise ValueError("Link OAuth credentials required")
        if LINK_PAYMENT_SCOPE not in credentials.scopes:
            raise ValueError("Link payment authorization required")
        if _settings.config.behave_as == BehaveAs.CLOUD and (
            not is_hosted_link_credential(credentials)
            or credentials.metadata.get("link_client_id")
            != _settings.secrets.stripe_link_client_id
        ):
            # The public device client is not the identity Stripe supports
            # for a hosted agent service.
            raise ValueError("Connect Link through the registered hosted client")
        yield credentials
        await lease.validate()
    finally:
        await lease.release()


async def chat_link_credential(user_id: str, session_id: str) -> str:
    """The Link account a chat pays with when the agent names none: the one
    the user picked in this chat (a connect card records the pick), else their
    only Link connection that can pay. Never a guess between several."""
    connected = [
        credential
        for credential in await _credentials.store.get_creds_by_provider(
            user_id, "stripe_link"
        )
        if isinstance(credential, OAuth2Credentials)
        and LINK_PAYMENT_SCOPE in credential.scopes
    ]
    picked = (await selected_credentials(session_id)).get("stripe_link")
    candidates = [c for c in connected if c.id == picked] or connected
    if len(candidates) == 1:
        return candidates[0].id
    raise CheckoutRefused(LINK_ACCOUNT_NOT_CHOSEN if candidates else LINK_NOT_CONNECTED)


def approval_for(
    approval: ApprovalView | None, view: CheckoutView, principal: Principal
) -> ApprovalView | None:
    """The in-chat approval, only if it is this chat's and names exactly the
    purchase the checkout engine holds."""
    if approval is None:
        return None
    pending = approval.pending
    if (
        pending.user_id != principal.user_id
        or pending.session_id != principal.session_id
        or (
            pending.merchant_name,
            pending.merchant_url,
            pending.amount,
            pending.currency,
            pending.test_mode,
        )
        != (
            view.merchant_name,
            view.merchant_url,
            view.amount,
            view.currency,
            view.test_mode,
        )
    ):
        return None
    return approval


def checkout_response(
    view: CheckoutView,
    session: ChatSession,
    approval: ApprovalView | None = None,
    message: str | None = None,
) -> CheckoutResponse:
    response = CheckoutResponse(
        message=message or view.message,
        session_id=session.session_id,
        **view.model_dump(exclude={"message", "credentials_id"}),
    )
    if approval is not None and view.spend_request_id is None:
        response.approval_state = approval.state
    return response


def invalid_plan(error: ValidationError) -> str:
    """Which arguments were wrong and why, so the agent can fix its call.
    ``CheckoutPlan`` hides inputs in its errors, so only field names and the
    rule they broke come back."""
    problems = "; ".join(
        f"{'.'.join(map(str, detail['loc'])) or 'plan'}: {detail['msg']}"
        for detail in error.errors(include_url=False, include_input=False)[:6]
    )
    return f"Invalid checkout plan. {problems}"[:900]


def log_failure(step: str, error: BaseException) -> None:
    """Record where a checkout step failed for the operator. Only the error's
    type and the line of ours that raised it: its message could carry page
    content, or after approval part of a card, so it is never logged."""
    ours = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if "site-packages" not in frame.filename
    ]
    where = "unknown"
    if ours:
        path = ours[-1].filename.replace("\\", "/").rsplit("/backend/", 1)[-1]
        where = f"{path}:{ours[-1].lineno}"
    logger.warning(f"Link checkout {step} failed: {type(error).__name__} at {where}")


def failure(session: ChatSession, message: str) -> ErrorResponse:
    return ErrorResponse(
        message=message,
        error="private_checkout_unavailable",
        session_id=session.session_id,
    )
