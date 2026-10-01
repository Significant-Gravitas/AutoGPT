"""Raising an unpaid checkout's total (Link's incremental authorization), end
to end through the copilot tools over the in-process broker."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools import browser_checkout as request_tools
from backend.copilot.tools import browser_checkout_raise as raise_tools
from backend.copilot.tools import browser_checkout_status as status_tools
from backend.copilot.tools import browser_checkout_support as support
from backend.copilot.tools.browser_checkout_support import CheckoutResponse
from backend.util.link_checkout import approval
from backend.util.link_checkout.preflight import PurchaseBlocker
from backend.util.link_checkout.refusals import RAISE_NOT_HIGHER, RAISE_REFUSED

REASON = "Sales tax was added on the final checkout step."


@asynccontextmanager
async def fake_link_credentials(user_id: str, credentials_id: str):
    yield MagicMock(access_token=SecretStr("synthetic-token"))


@pytest.fixture
def tools(local_broker, fake_redis, monkeypatch):
    monkeypatch.setattr(support, "available", lambda: True)
    for module in (request_tools, status_tools, raise_tools):
        monkeypatch.setattr(module, "link_credentials", fake_link_credentials)
    for module in (request_tools, raise_tools):
        monkeypatch.setattr(module, "purchase_blocker", AsyncMock(return_value=None))
        monkeypatch.setattr(
            module, "in_app_approval_allowed", AsyncMock(return_value=False)
        )
    monkeypatch.setattr(request_tools, "validate_url_host", AsyncMock())
    monkeypatch.setattr("backend.copilot.tools.base._record_activity", AsyncMock())
    return local_broker


async def run(name: str, **arguments):
    session = MagicMock(
        spec=ChatSession,
        session_id="chat",
        user_id="owner",
        metadata=MagicMock(origin="scheduled"),
    )
    return await TOOL_REGISTRY[name].execute("owner", session, "call", **arguments)


def parsed(result) -> CheckoutResponse:
    return CheckoutResponse.model_validate_json(result.output)


async def request(plan) -> CheckoutResponse:
    return parsed(await run("browser_request_link_payment", **plan.model_dump()))


async def raise_to(checkout_id: str, amount: int):
    return await run(
        "browser_raise_link_payment",
        checkout_id=checkout_id,
        amount=amount,
        reason=REASON,
    )


async def complete(checkout_id: str) -> CheckoutResponse:
    return parsed(await run("browser_complete_link_payment", checkout_id=checkout_id))


@pytest.mark.asyncio
async def test_a_higher_total_is_reapproved_in_link_then_paid_once(tools, plan):
    created = await request(plan)

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert (raised.amount, raised.revision) == (250, 1)
    assert raised.status == "pending_approval"
    assert raised.approval_mode == "link"
    assert raised.approval_url.startswith("https://app.link.com/")
    assert tools.calls == ["create", "raise"]
    assert tools.amounts == [100, 250]

    paid = await complete(created.checkout_id)

    assert paid.status == "submitted"
    assert tools.calls == ["create", "raise", "status", "pay"]
    assert tools.amounts[-1] == 250


@pytest.mark.asyncio
async def test_a_raise_in_the_chat_needs_its_own_approval(tools, plan, monkeypatch):
    for module in (request_tools, raise_tools):
        monkeypatch.setattr(
            module, "in_app_approval_allowed", AsyncMock(return_value=True)
        )
    created = await request(plan)
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None
    )

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert raised.status == "awaiting_approval"
    assert raised.approval_state == "awaiting"
    record = await approval.read_approval(created.checkout_id)
    assert record is not None
    pending = record.pending
    assert (pending.amount, pending.previous_amount, pending.revision) == (250, 100, 1)
    assert pending.reason == REASON
    # The approval of the old total does not pay the new one.
    assert (await complete(created.checkout_id)).status == "awaiting_approval"
    assert tools.calls == []

    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None, revision=1
    )
    paid = await complete(created.checkout_id)

    assert paid.status == "submitted"
    assert tools.calls == ["create_delegated", "pay"]
    assert tools.amounts == [250, 250]


@pytest.mark.asyncio
async def test_a_raise_beyond_the_chat_policy_moves_to_link(tools, plan, monkeypatch):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = await request(plan)

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert raised.approval_mode == "link"
    assert raised.status == "pending_approval"
    assert tools.calls == ["create"]
    assert tools.amounts == [250]
    # No card left in the chat can approve it there any more.
    record = await approval.read_approval(created.checkout_id)
    assert record is not None and record.state == "expired"


@pytest.mark.asyncio
async def test_link_refusing_a_raise_keeps_the_approved_total(tools, plan):
    created = await request(plan)
    tools.raise_error = "link_rejected"

    refused = await raise_to(created.checkout_id, 250)

    assert RAISE_REFUSED in refused.output
    paid = await complete(created.checkout_id)
    assert paid.status == "submitted"
    assert paid.amount == 100
    assert tools.amounts[-1] == 100


@pytest.mark.asyncio
async def test_a_raise_must_be_higher(tools, plan):
    created = await request(plan)

    result = await raise_to(created.checkout_id, 100)

    assert RAISE_NOT_HIGHER in result.output
    assert tools.calls == ["create"]


@pytest.mark.asyncio
async def test_an_attempted_checkout_cannot_be_raised(tools, plan):
    created = await request(plan)
    await complete(created.checkout_id)
    calls = list(tools.calls)

    result = await raise_to(created.checkout_id, 250)

    assert "already attempted" in result.output
    assert tools.calls == calls


@pytest.mark.asyncio
async def test_a_raise_over_a_link_limit_changes_nothing(tools, plan, monkeypatch):
    created = await request(plan)
    monkeypatch.setattr(
        raise_tools,
        "purchase_blocker",
        AsyncMock(return_value=PurchaseBlocker(message="Over the agent limit.")),
    )

    result = await raise_to(created.checkout_id, 250)

    assert "Over the agent limit." in result.output
    assert tools.calls == ["create"]
    status = parsed(
        await run("browser_link_payment_status", checkout_id=created.checkout_id)
    )
    assert (status.amount, status.revision) == (100, 0)


@pytest.mark.asyncio
async def test_a_request_over_a_link_limit_is_stopped_before_approval(
    tools, plan, monkeypatch
):
    monkeypatch.setattr(
        request_tools,
        "purchase_blocker",
        AsyncMock(return_value=PurchaseBlocker(message="Verify your identity.")),
    )

    result = await run("browser_request_link_payment", **plan.model_dump())

    assert "Verify your identity." in result.output
    assert tools.calls == []
