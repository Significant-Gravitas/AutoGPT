"""Link spend-request calls, made from the payment worker.

https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online
Only the ``pay`` path asks for card details (``include=card``); the worker is
the one process that ever holds them.
"""

import time
from datetime import UTC, datetime
from urllib.parse import SplitResult, quote, urlsplit

import httpx
from pydantic import BaseModel, ValidationError

from backend.util.link_checkout.config import https_proxy
from backend.util.link_checkout.models import CheckoutIntent, SpendRequest, WorkerJob

LINK_API_BASE_URL = "https://api.link.com"
LINK_HTTP_TIMEOUT = 15.0
# Hosts Link sends a customer to for approval or a required action.
_LINK_ACTION_DOMAINS = ("link.com", "stripe.com")
_DEFAULT_PORTS = {"https": 443, "http": 80}


class LinkRejected(Exception):
    """Link refused the request outright (HTTP 4xx), so nothing was created."""


class LinkDuplicate(LinkRejected):
    """Link refused a new request because a matching one is still open
    (``spend_request_rate_limited``); asking another way meets the same one."""


async def request_spend(job: WorkerJob, include_card: bool = False) -> SpendRequest:
    method, path, body = _spend_call(job)
    async with httpx.AsyncClient(
        timeout=LINK_HTTP_TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        proxy=https_proxy(),
    ) as client:
        response = await client.request(
            method,
            f"{LINK_API_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {job.access_token.get_secret_value()}"},
            json=body,
            params={"include": "card"} if include_card else None,
        )
    if 400 <= response.status_code < 500:
        if _error_code(response) == "spend_request_rate_limited":
            raise LinkDuplicate()
        raise LinkRejected()
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError("Link request failed")
    return SpendRequest.model_validate_json(response.content)


class _LinkError(BaseModel):
    code: str = ""


class _LinkErrorBody(BaseModel):
    error: _LinkError = _LinkError()


def _error_code(response: httpx.Response) -> str:
    try:
        return _LinkErrorBody.model_validate_json(response.content).error.code
    except ValidationError:
        return ""


def _spend_call(job: WorkerJob) -> tuple[str, str, dict | None]:
    intent = job.intent
    if job.action == "create":
        # One logical purchase per checkout: a retry after a lost response
        # returns the original request instead of creating a second one.
        return "POST", "/spend_requests", _create_body(intent, intent.id)
    if job.action == "create_delegated":
        if job.approval is None:
            raise ValueError("A delegated spend request needs approval details")
        body = _create_body(intent, f"{intent.id}-in-app")
        body["approval_details"] = job.approval.model_dump(exclude_none=True)
        return "POST", "/spend_requests/create_delegated", body
    if not intent.spend_request_id:
        raise ValueError("This checkout has no Link spend request yet")
    path = f"/spend_requests/{quote(intent.spend_request_id, safe='')}"
    if job.action == "cancel":
        # Link cancels from created, pending_approval or approved; an approved
        # request's card stops working with it.
        return "POST", f"{path}/cancel", None
    return "GET", path, None


def _create_body(intent: CheckoutIntent, idempotency_key: str) -> dict:
    plan = intent.plan
    body = {
        "idempotency_key": idempotency_key,
        "payment_details": plan.payment_method_id,
        "merchant_name": plan.merchant_name,
        "merchant_url": plan.merchant_url(),
        "context": plan.context,
        "amount": plan.amount,
        "currency": plan.currency,
        "test": plan.test_mode,
        "metadata": {"autogpt_checkout_id": intent.id},
        "totals": [{"type": "total", "display_text": "Total", "amount": plan.amount}],
    }
    if intent.approval_mode == "link":
        body["request_approval"] = True
    return body


def validate_spend(
    intent: CheckoutIntent,
    spend: SpendRequest,
    require_card: bool = False,
    *,
    check_deadline: bool = True,
) -> None:
    """The spend request must still be the purchase the customer approved."""
    if check_deadline and intent.expires_at <= time.time():
        raise RuntimeError("Checkout approval expired")
    if intent.spend_request_id and spend.id != intent.spend_request_id:
        raise RuntimeError("The Link approval no longer matches this checkout")
    if (
        not _same_page(spend.merchant_url, intent.plan.merchant_url())
        or spend.amount != intent.plan.amount
        or (spend.currency or "").lower() != intent.plan.currency
    ):
        raise RuntimeError("The Link approval no longer matches this checkout")
    if require_card:
        _validate_card(spend)


def _validate_card(spend: SpendRequest) -> None:
    card = spend.card
    if spend.status != "approved" or card is None:
        raise RuntimeError("Link has not approved this payment")
    if card.valid_until is not None and (
        card.valid_until.tzinfo is None or card.valid_until <= datetime.now(UTC)
    ):
        raise RuntimeError("The Link card has expired")
    number = card.number.get_secret_value()
    cvc = card.cvc.get_secret_value()
    if not (number.isascii() and number.isdigit() and 12 <= len(number) <= 19):
        raise RuntimeError("Invalid virtual card")
    if not (cvc.isascii() and cvc.isdigit() and len(cvc) in {3, 4}):
        raise RuntimeError("Invalid virtual card")


def _same_page(returned: str | None, sent: str) -> bool:
    if not returned:
        return False
    a, b = urlsplit(returned), urlsplit(sent)
    return (
        a.scheme == b.scheme
        and (a.hostname or "").lower() == (b.hostname or "").lower()
        and _port(a) == _port(b)
        and a.path.rstrip("/") == b.path.rstrip("/")
    )


def _port(url: SplitResult) -> int | None:
    # An explicit default port names the same page as none: Link may drop it.
    return url.port or _DEFAULT_PORTS.get(url.scheme)


def link_action_url(value: str | None) -> str:
    """A URL from Link that the chat may show as a button, or ""."""
    if not value:
        return ""
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or not any(host == d or host.endswith(f".{d}") for d in _LINK_ACTION_DOMAINS)
    ):
        return ""
    return value
