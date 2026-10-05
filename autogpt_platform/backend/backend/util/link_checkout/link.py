"""Link spend-request calls, made from the payment worker.

https://docs.stripe.com/agentic-commerce/link-agent-wallet/use-link-wallet-pay-online
Only the ``pay`` path asks for the payment credential (``include=card``, or
``include=link_pay_token`` for a Stripe checkout); the worker is the one
process that ever holds it.
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


async def request_spend(
    job: WorkerJob, include_credential: bool = False
) -> SpendRequest:
    async with httpx.AsyncClient(
        timeout=LINK_HTTP_TIMEOUT,
        follow_redirects=False,
        trust_env=False,
        proxy=https_proxy(),
    ) as client:
        if job.action == "raise":
            return await _raise(client, job)
        method, path, body = _spend_call(job)
        response = await _send(
            client,
            job,
            method,
            path,
            body,
            params=(
                {"include": _credential(job.intent)} if include_credential else None
            ),
        )
    return SpendRequest.model_validate_json(response.content)


async def _send(
    client: httpx.AsyncClient,
    job: WorkerJob,
    method: str,
    path: str,
    body: dict | None = None,
    params: dict | None = None,
) -> httpx.Response:
    response = await client.request(
        method,
        f"{LINK_API_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {job.access_token.get_secret_value()}"},
        json=body,
        params=params,
    )
    if 400 <= response.status_code < 500:
        if _error_code(response) == "spend_request_rate_limited":
            raise LinkDuplicate()
        raise LinkRejected()
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError("Link request failed")
    return response


async def _raise(client: httpx.AsyncClient, job: WorkerJob) -> SpendRequest:
    """Link's incremental authorization: the checkout's request takes the new
    total (``job.intent``'s), then the customer approves it again in Link. If
    Link refuses, the request stays usable at its old amount."""
    path = _spend_path(job.intent)
    total = job.intent.plan.amount
    await _send(
        client,
        job,
        "POST",
        path,
        {
            "amount": total,
            "totals": [{"type": "total", "display_text": "Total", "amount": total}],
        },
    )
    requested = await _send(client, job, "POST", f"{path}/request_approval")
    spend = SpendRequest.model_validate_json(
        (await _send(client, job, "GET", path)).content
    )
    spend.approval_url = (
        _ApprovalLink.model_validate_json(requested.content).approval_link
        or spend.approval_url
    )
    return spend


class _ApprovalLink(BaseModel):
    approval_link: str | None = None


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
        # One logical purchase per checkout revision: a retry after a lost
        # response returns the original request instead of creating a second.
        return "POST", "/spend_requests", _create_body(intent, _idempotency(intent))
    if job.action == "create_delegated":
        if job.approval is None:
            raise ValueError("A delegated spend request needs approval details")
        body = _create_body(intent, _idempotency(intent, "in-app"))
        body["approval_details"] = job.approval.model_dump(exclude_none=True)
        return "POST", "/spend_requests/create_delegated", body
    path = _spend_path(intent)
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
        "context": plan.context,
        "amount": plan.amount,
        "currency": plan.currency,
        "metadata": {"autogpt_checkout_id": intent.id},
        "totals": [{"type": "total", "display_text": "Total", "amount": plan.amount}],
    }
    pay_token = intent.browser.pay_token
    if plan.execution == "link_pay_token" and pay_token is not None:
        # Link resolves the merchant from the Stripe account the page named,
        # so the request carries no merchant name or URL, and has no test mode.
        body["execution_method"] = "link_pay_token"
        body["merchant_account_id"] = pay_token.merchant_account_id
    else:
        body["merchant_name"] = plan.merchant_name
        body["merchant_url"] = plan.merchant_url()
        body["test"] = plan.test_mode
    if intent.approval_mode == "link":
        body["request_approval"] = True
    return body


def _credential(intent: CheckoutIntent) -> str:
    return "link_pay_token" if intent.plan.execution == "link_pay_token" else "card"


def _spend_path(intent: CheckoutIntent) -> str:
    if not intent.spend_request_id:
        raise ValueError("This checkout has no Link spend request yet")
    return f"/spend_requests/{quote(intent.spend_request_id, safe='')}"


def _idempotency(intent: CheckoutIntent, route: str = "") -> str:
    """A raised total is a new revision of the purchase, so a retried create
    never returns the request made for the old total."""
    revision = f"r{intent.revision}" if intent.revision else ""
    return "-".join(part for part in (intent.id, route, revision) if part)


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
    pay_token = intent.plan.execution == "link_pay_token"
    if (
        # With a Link Pay Token, Link names the merchant itself, from the
        # Stripe account the page carried; the token pays only that account.
        not (pay_token or _same_page(spend.merchant_url, intent.plan.merchant_url()))
        or spend.amount != intent.plan.amount
        or (spend.currency or "").lower() != intent.plan.currency
    ):
        raise RuntimeError("The Link approval no longer matches this checkout")
    if require_card and pay_token:
        _validate_pay_token(spend)
    elif require_card:
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


def _validate_pay_token(spend: SpendRequest) -> None:
    token = spend.link_pay_token
    if spend.status != "approved" or token is None:
        raise RuntimeError("Link has not approved this payment")
    value = token.get_secret_value()
    if not (value.isascii() and value.isprintable() and 8 <= len(value) <= 512):
        raise RuntimeError("Invalid Link Pay Token")


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
