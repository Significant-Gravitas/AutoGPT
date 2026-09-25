import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from backend.util.link_checkout import link
from backend.util.link_checkout.link import (
    LinkDuplicate,
    LinkRejected,
    link_action_url,
    request_spend,
    validate_spend,
)
from backend.util.link_checkout.models import ApprovalDetails, SpendRequest, WorkerJob


@pytest.fixture
def link_api(monkeypatch):
    sent: list[httpx.Request] = []
    reply: dict = {"status": 200, "body": None}

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(
            reply["status"],
            json=reply["body"]
            or {
                "id": "lsrq_new",
                "status": "pending_approval",
                "merchant_url": "https://shop.example/checkout",
                "amount": 100,
                "currency": "usd",
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        link.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond)),
    )
    return sent, reply


def job(intent, action, approval=None):
    return WorkerJob(
        action=action,
        intent=intent,
        access_token=SecretStr("liwltoken_test"),
        approval=approval,
    )


@pytest.mark.asyncio
async def test_create_is_idempotent_and_asks_the_customer_in_link(intent, link_api):
    sent, _ = link_api
    intent.spend_request_id = None
    intent.plan.checkout_url = "https://shop.example/checkout?cart=secret"

    await request_spend(job(intent, "create"))

    body = json.loads(sent[0].content)
    assert str(sent[0].url) == "https://api.link.com/spend_requests"
    assert sent[0].headers["Authorization"] == "Bearer liwltoken_test"
    assert body["idempotency_key"] == intent.id
    assert body["request_approval"] is True
    assert body["merchant_url"] == "https://shop.example/checkout"
    assert body["payment_details"] == "csmrpd_test"
    assert body["test"] is True
    assert "approval_details" not in body


@pytest.mark.asyncio
async def test_delegated_create_carries_the_customers_approval(intent, link_api):
    sent, _ = link_api
    intent.spend_request_id = None
    intent.approval_mode = "in_app"
    approval = ApprovalDetails(
        approved_at=1_790_000_000,
        external_user_id="owner",
        external_session_id="chat",
        agent_log_id=intent.id,
        user_agent="Mozilla/5.0",
    )

    await request_spend(job(intent, "create_delegated", approval))

    body = json.loads(sent[0].content)
    assert str(sent[0].url) == "https://api.link.com/spend_requests/create_delegated"
    assert body["idempotency_key"] == f"{intent.id}-in-app"
    assert "request_approval" not in body
    assert body["approval_details"] == {
        "approved_at": 1_790_000_000,
        "approval_method": "click",
        "app_name": "AutoGPT",
        "external_user_id": "owner",
        "external_session_id": "chat",
        "agent_log_id": intent.id,
        "device_type": "web",
        "user_agent": "Mozilla/5.0",
    }


@pytest.mark.asyncio
async def test_card_details_are_requested_only_to_pay(intent, link_api):
    sent, _ = link_api
    await request_spend(job(intent, "status"))
    await request_spend(job(intent, "pay"), include_card=True)

    assert "include" not in sent[0].url.params
    assert sent[1].url.params["include"] == "card"
    assert sent[1].url.path == "/spend_requests/lsrq_test"


@pytest.mark.asyncio
async def test_a_4xx_is_a_rejection_and_anything_else_is_unknown(intent, link_api):
    _, reply = link_api
    reply["status"] = 403
    with pytest.raises(LinkRejected):
        await request_spend(job(intent, "status"))
    reply["status"] = 502
    with pytest.raises(RuntimeError):
        await request_spend(job(intent, "status"))


def spend_for(intent, **changes) -> SpendRequest:
    data = {
        "id": intent.spend_request_id,
        "merchant_url": intent.plan.merchant_url(),
        "amount": intent.plan.amount,
        "currency": intent.plan.currency,
        "status": "approved",
        **changes,
    }
    return SpendRequest.model_validate(data)


def test_local_deadline_is_rechecked_after_link_response(intent):
    intent.expires_at = 0
    with pytest.raises(RuntimeError):
        validate_spend(intent, spend_for(intent))


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "lsrq_other"),
        ("merchant_url", "https://evil.example/checkout"),
        ("amount", 101),
        ("currency", "eur"),
        ("merchant_url", None),
    ],
)
def test_changed_link_approval_is_rejected(intent, field, value):
    with pytest.raises(RuntimeError):
        validate_spend(intent, spend_for(intent, **{field: value}))


def test_link_may_normalize_the_merchant_page(intent):
    validate_spend(
        intent,
        spend_for(
            intent, merchant_url="https://SHOP.example/checkout/", currency="USD"
        ),
    )


@pytest.mark.parametrize(
    "sent,returned",
    [
        ("https://shop.example:443/checkout", "https://shop.example/checkout"),
        ("https://shop.example/checkout", "https://shop.example:443/checkout"),
    ],
)
def test_an_explicit_default_port_is_the_same_page(intent, sent, returned):
    intent.plan.checkout_url = sent
    validate_spend(intent, spend_for(intent, merchant_url=returned))


def test_another_port_is_another_page(intent):
    with pytest.raises(RuntimeError):
        validate_spend(
            intent, spend_for(intent, merchant_url="https://shop.example:8443/checkout")
        )


@pytest.mark.parametrize(
    "status,expired",
    [("pending_approval", False), ("denied", False), ("approved", True)],
)
def test_card_requires_approval_and_validity(intent, status, expired):
    spend = spend_for(
        intent,
        status=status,
        card={
            "number": "4242424242424242",
            "cvc": "987",
            "exp_month": 12,
            "exp_year": 2030,
            "valid_until": datetime.now(UTC) + timedelta(minutes=-1 if expired else 1),
        },
    )
    assert "4242424242424242" not in spend.model_dump_json()
    assert "987" not in repr(spend)
    with pytest.raises(RuntimeError):
        validate_spend(intent, spend, require_card=True)


@pytest.mark.parametrize(
    "url,shown",
    [
        ("https://app.link.com/activity/approve/lsrq_1", True),
        ("https://link.com/verify", True),
        ("https://hooks.stripe.com/3d_secure/abc", True),
        ("https://evil.example/app.link.com", False),
        ("https://link.com.evil.example/", False),
        ("http://app.link.com/activity", False),
        ("https://user:pass@app.link.com/", False),
        ("https://app.link.com:8443/", False),
        (None, False),
    ],
)
def test_only_link_and_stripe_pages_become_buttons(url, shown):
    assert (link_action_url(url) == url) is shown


@pytest.mark.asyncio
async def test_a_request_matching_an_open_one_is_told_apart(intent, link_api):
    """Link's answer to a duplicate, as link-cli reads it."""
    _, reply = link_api
    intent.spend_request_id = None
    reply.update(
        status=429,
        body={
            "error": {
                "code": "spend_request_rate_limited",
                "message": "You cannot submit duplicate spend requests.",
                "duplicate_spend_request": {"id": "lsrq_open", "status": "created"},
            }
        },
    )
    with pytest.raises(LinkDuplicate):
        await request_spend(job(intent, "create"))

    reply.update(status=400, body={"error": {"code": "invalid_request"}})
    with pytest.raises(LinkRejected) as refused:
        await request_spend(job(intent, "create"))
    assert not isinstance(refused.value, LinkDuplicate)


@pytest.mark.asyncio
async def test_cancel_posts_to_the_spend_requests_cancel_endpoint(intent, link_api):
    sent, reply = link_api
    reply["body"] = {
        "id": intent.spend_request_id,
        "status": "canceled",
        "merchant_url": "https://shop.example/checkout",
        "amount": 100,
        "currency": "usd",
    }

    spend = await request_spend(job(intent, "cancel"))

    assert sent[0].method == "POST"
    assert str(sent[0].url) == (
        f"https://api.link.com/spend_requests/{intent.spend_request_id}/cancel"
    )
    assert sent[0].content == b""
    assert spend.status == "canceled"
