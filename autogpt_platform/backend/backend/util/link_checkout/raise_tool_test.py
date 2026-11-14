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
from backend.copilot.tools.models import ErrorResponse
from backend.util.link_checkout import approval
from backend.util.link_checkout.preflight import PurchaseBlocker
from backend.util.link_checkout.refusals import (
    APPROVAL_SUPERSEDED,
    RAISE_CLOSED,
    RAISE_NOT_HIGHER,
)

REASON = "Sales tax was added on the final checkout step."


@asynccontextmanager
async def fake_link_credentials(
    user_id: str, credentials_id: str, *, expert_id: str | None
):
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
        expert_id=None,
        organization_id=None,
        external_caller=False,
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
    # Link raises only an approved request, so its status is read first.
    assert tools.calls == ["create", "status", "raise"]
    assert tools.amounts == [100, 100, 250]

    paid = await complete(created.checkout_id)

    assert paid.status == "submitted"
    assert tools.calls == ["create", "status", "raise", "status", "pay"]
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
async def test_a_raise_landing_mid_completion_is_not_paid_on_the_old_approval(
    tools, plan, monkeypatch
):
    """Tools run in parallel, so a raise can land after the complete tool read
    the approval and before the broker's lock. The broker checks the approval's
    revision under that lock, so Link never gets the raised total already
    approved on the old one's approval."""
    for module in (request_tools, raise_tools):
        monkeypatch.setattr(
            module, "in_app_approval_allowed", AsyncMock(return_value=True)
        )
    created = await request(plan)
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None
    )
    complete_now = request_tools.engine.complete

    async def raised_first(checkout):
        await raise_to(created.checkout_id, 250)
        return await complete_now(checkout)

    monkeypatch.setattr(request_tools.engine, "complete", raised_first)

    result = await run("browser_complete_link_payment", checkout_id=created.checkout_id)

    assert (
        ErrorResponse.model_validate_json(result.output).message == APPROVAL_SUPERSEDED
    )
    assert tools.calls == []


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
async def test_link_refusing_a_raise_asks_again_for_the_new_total(tools, plan):
    """Link's guide: an increase can fail; cancel the request and create one
    for the new total, which the user approves in Link."""
    created = await request(plan)
    tools.raise_error = "link_rejected"

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert raised.status == "pending_approval"
    assert raised.amount == 250
    assert tools.calls == ["create", "status", "raise", "cancel", "create"]
    assert tools.amounts[-1] == 250
    paid = await complete(created.checkout_id)
    assert paid.status == "submitted"
    assert tools.amounts[-1] == 250


@pytest.mark.asyncio
async def test_a_request_not_yet_approved_is_asked_for_again(tools, plan):
    """Link raises only an approved request; one still waiting for approval
    is canceled and asked for again at the new total."""
    created = await request(plan)
    tools.status = "pending_approval"

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert raised.amount == 250
    assert tools.calls == ["create", "status", "cancel", "create"]


@pytest.mark.asyncio
async def test_a_purchase_declined_in_link_cannot_be_raised(tools, plan):
    created = await request(plan)
    tools.status = "denied"

    result = await raise_to(created.checkout_id, 250)

    assert RAISE_CLOSED in result.output
    assert tools.calls == ["create", "status"]


@pytest.mark.asyncio
async def test_a_raise_must_be_higher(tools, plan):
    created = await request(plan)

    result = await raise_to(created.checkout_id, 100)

    assert RAISE_NOT_HIGHER in result.output
    assert tools.calls == ["create"]


@pytest.mark.asyncio
async def test_a_raise_whose_new_request_failed_is_finished_by_a_retry(tools, plan):
    """A request not yet approved is canceled and asked for again at the new
    total. When that new request fails, the checkout is already saved at the
    new total, so the advised retry asks Link again rather than finding the
    old request canceled."""
    created = await request(plan)
    tools.status = "pending_approval"
    tools.create_error = "link_rejected"
    failed = await raise_to(created.checkout_id, 250)
    assert "call this with the same new total" in failed.output
    tools.create_error = None

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert (raised.amount, raised.revision) == (250, 1)
    assert raised.status == "pending_approval"
    assert tools.calls == ["create", "status", "cancel", "create", "create"]
    tools.status = "approved"
    assert (await complete(created.checkout_id)).status == "submitted"
    assert tools.amounts[-1] == 250


@pytest.mark.asyncio
async def test_a_raise_moved_to_link_whose_request_failed_is_finished_by_a_retry(
    tools, plan, monkeypatch
):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = await request(plan)
    tools.create_error = "link_rejected"
    await raise_to(created.checkout_id, 250)
    tools.create_error = None

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert (raised.amount, raised.revision) == (250, 1)
    assert (raised.approval_mode, raised.status) == ("link", "pending_approval")
    assert tools.calls == ["create", "create"]


@pytest.mark.asyncio
async def test_a_raise_whose_chat_record_failed_is_recorded_by_a_retry(
    tools, plan, monkeypatch
):
    """The checkout is raised before the chat's approval record is. When that
    record fails, the retry at the same total writes it, so the customer can
    approve the new total in the chat."""
    for module in (request_tools, raise_tools):
        monkeypatch.setattr(
            module, "in_app_approval_allowed", AsyncMock(return_value=True)
        )
    created = await request(plan)
    reopen = raise_tools.reopen_approval
    monkeypatch.setattr(
        raise_tools, "reopen_approval", AsyncMock(side_effect=RuntimeError("down"))
    )
    await raise_to(created.checkout_id, 250)
    monkeypatch.setattr(raise_tools, "reopen_approval", reopen)

    raised = parsed(await raise_to(created.checkout_id, 250))

    assert raised.approval_state == "awaiting"
    record = await approval.read_approval(created.checkout_id)
    assert record is not None
    pending = record.pending
    assert (pending.amount, pending.previous_amount, pending.revision) == (250, 100, 1)
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None, revision=1
    )
    assert (await complete(created.checkout_id)).status == "submitted"
    assert tools.amounts == [250, 250]


@pytest.mark.asyncio
async def test_a_retry_after_a_finished_raise_leaves_the_chat_record_alone(
    tools, plan, monkeypatch
):
    """The card keeps showing the raise from the old total, and the customer's
    decision on it stands."""
    for module in (request_tools, raise_tools):
        monkeypatch.setattr(
            module, "in_app_approval_allowed", AsyncMock(return_value=True)
        )
    created = await request(plan)
    await raise_to(created.checkout_id, 250)
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None, revision=1
    )

    again = parsed(await raise_to(created.checkout_id, 250))

    assert again.approval_state == "approved"
    record = await approval.read_approval(created.checkout_id)
    assert record is not None
    assert (record.pending.previous_amount, record.pending.revision) == (100, 1)
    assert (await complete(created.checkout_id)).status == "submitted"


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
