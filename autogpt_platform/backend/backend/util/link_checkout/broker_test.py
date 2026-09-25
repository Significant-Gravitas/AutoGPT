import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from backend.util.link_checkout import broker_commands, broker_link, broker_service
from backend.util.link_checkout.models import ApprovalDetails, WorkerReceipt
from backend.util.link_checkout.refusals import (
    ATTEMPT_UNRECONCILED,
    DUPLICATE_REQUEST,
    LIVE_PAYMENTS_DISABLED,
)

SECRET = "test-controller-credential-" + "x" * 32
PRINCIPAL = {"user_id": "owner", "session_id": "chat"}


@pytest.fixture
def broker(local_broker, monkeypatch):
    monkeypatch.setattr(
        broker_commands,
        "browser_command",
        AsyncMock(return_value=(0, "safe snapshot", "")),
    )
    return broker_service.create_app("owner", SECRET.encode())


def client(app, authenticated=True):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="https://broker.example",
        headers={"Authorization": f"Bearer {SECRET}"} if authenticated else {},
    )


async def create(api, plan, approval_mode="link") -> dict:
    result = await api.post(
        "/v1/checkout/create",
        json={
            **PRINCIPAL,
            "plan": plan.model_dump(),
            "access_token": "synthetic",
            "approval_mode": approval_mode,
        },
    )
    assert result.status_code == 200, result.text
    return result.json()


def approval(checkout_id: str) -> dict:
    return ApprovalDetails(
        approved_at=int(time.time()),
        external_user_id="owner",
        external_session_id="chat",
        agent_log_id=checkout_id,
    ).model_dump()


@pytest.mark.asyncio
async def test_missing_auth_and_cross_tenant_requests_never_reach_browser(broker):
    async with client(broker, False) as api:
        missing = await api.post(
            "/v1/browser", json={**PRINCIPAL, "args": ["get", "url"]}
        )
        assert missing.status_code == 401
    async with client(broker) as api:
        other = await api.post(
            "/v1/browser",
            json={"user_id": "other", "session_id": "chat", "args": ["get", "url"]},
        )
        assert other.status_code == 403
    broker_commands.browser_command.assert_not_called()


@pytest.mark.asyncio
async def test_failure_never_logs_or_returns_exception_secrets(
    broker, monkeypatch, caplog
):
    monkeypatch.setattr(
        broker_commands,
        "browser_command",
        AsyncMock(side_effect=RuntimeError("canary-4242424242424242")),
    )
    async with client(broker) as api:
        result = await api.post(
            "/v1/browser", json={**PRINCIPAL, "args": ["get", "url"]}
        )
    assert result.status_code == 409
    assert "canary" not in result.text + caplog.text


@pytest.mark.asyncio
async def test_validation_never_echoes_secret_input(broker):
    async with client(broker) as api:
        result = await api.post(
            "/v1/checkout/create",
            json={"access_token": "secret-canary-4242424242424242"},
        )
        assert result.status_code == 422
        assert "canary" not in result.text


@pytest.mark.asyncio
async def test_link_approval_pays_once_then_waits_for_links_final_status(
    broker, local_broker, plan
):
    async with client(broker) as api:
        created = await create(api, plan)
        assert created["status"] == "pending_approval"
        assert created["approval_url"].startswith("https://app.link.com/")
        assert created["merchant_url"] == "https://shop.example/checkout"
        reference = {**PRINCIPAL, "checkout_id": created["checkout_id"]}
        authorized = {**reference, "access_token": "synthetic"}

        first, replay = await asyncio.gather(
            api.post("/v1/checkout/complete", json=authorized),
            api.post("/v1/checkout/complete", json=authorized),
        )
        assert first.status_code == replay.status_code == 200
        assert first.json()["attempted"] is replay.json()["attempted"] is True
        assert local_broker.calls.count("pay") == 1

        # The browser that held the card was retired, so browsing resumes in a
        # fresh one; another checkout waits for this one's final status.
        browsed = await api.post(
            "/v1/browser", json={**PRINCIPAL, "args": ["get", "url"]}
        )
        assert browsed.status_code == 200
        blocked = await api.post(
            "/v1/checkout/create",
            json={**PRINCIPAL, "plan": plan.model_dump(), "access_token": "synthetic"},
        )
        assert blocked.status_code == 422
        assert blocked.json()["detail"] == ATTEMPT_UNRECONCILED
        assert (await api.post("/v1/checkout/reset", json=reference)).status_code == 422

        local_broker.status = "submitted"
        status = await api.post("/v1/checkout/status", json=authorized)
        assert status.json()["paid"] is False
        assert (await api.post("/v1/checkout/reset", json=reference)).status_code == 422

        local_broker.status = "succeeded"
        status = await api.post("/v1/checkout/status", json=authorized)
        assert status.json()["paid"] is True
        assert local_broker.calls.count("pay") == 1
        again = await create(api, plan)
        assert again["checkout_id"] != created["checkout_id"]
        gone = await api.post("/v1/checkout/complete", json=authorized)
        assert gone.status_code == 409


@pytest.mark.asyncio
async def test_a_browser_that_could_not_be_retired_stays_sealed_until_reset(
    broker, local_broker, plan, monkeypatch
):
    local_broker.receipt = WorkerReceipt(status="outcome_unknown", browser_closed=False)
    monkeypatch.setattr(
        broker_link, "retire_payment_browser", AsyncMock(return_value=False)
    )
    async with client(broker) as api:
        created = await create(api, plan)
        reference = {**PRINCIPAL, "checkout_id": created["checkout_id"]}
        authorized = {**reference, "access_token": "synthetic"}
        await api.post("/v1/checkout/complete", json=authorized)

        sealed = await api.post(
            "/v1/browser", json={**PRINCIPAL, "args": ["snapshot", "-i", "-c"]}
        )
        assert sealed.status_code == 409
        broker_commands.browser_command.assert_not_called()
        assert (await api.post("/v1/checkout/reset", json=reference)).status_code == 422

        local_broker.status = "succeeded"
        await api.post("/v1/checkout/status", json=authorized)
        assert (await api.post("/v1/checkout/reset", json=reference)).status_code == 200
        browsed = await api.post(
            "/v1/browser", json={**PRINCIPAL, "args": ["get", "url"]}
        )
        assert browsed.status_code == 200


@pytest.mark.asyncio
async def test_an_attempt_that_never_reached_the_page_cancels_its_request(
    broker, local_broker, plan
):
    local_broker.receipt = WorkerReceipt(status="not_submitted", browser_closed=True)
    async with client(broker) as api:
        created = await create(api, plan)
        result = await api.post(
            "/v1/checkout/complete",
            json={
                **PRINCIPAL,
                "checkout_id": created["checkout_id"],
                "access_token": "synthetic",
            },
        )
        body = result.json()
        assert body["status"] == "canceled"
        assert body["paid"] is False
        assert "Nothing was charged" in body["message"]
        assert local_broker.calls == ["create", "status", "pay", "cancel"]
        # Nothing is left to reconcile, so the chat can buy again at once.
        await create(api, plan)


@pytest.mark.asyncio
async def test_in_chat_approval_creates_an_approved_request_then_pays(
    broker, local_broker, plan
):
    async with client(broker) as api:
        created = await create(api, plan, approval_mode="in_app")
        assert created["status"] == "awaiting_approval"
        assert created["spend_request_id"] is None
        assert local_broker.calls == []
        reference = {**PRINCIPAL, "checkout_id": created["checkout_id"]}

        waiting = await api.post(
            "/v1/checkout/complete", json={**reference, "access_token": "synthetic"}
        )
        assert waiting.json()["status"] == "awaiting_approval"
        assert local_broker.calls == []

        paid = await api.post(
            "/v1/checkout/complete",
            json={
                **reference,
                "access_token": "synthetic",
                "approval": approval(created["checkout_id"]),
            },
        )
        assert paid.status_code == 200, paid.text
        assert paid.json()["status"] == "submitted"
        assert local_broker.calls == ["create_delegated", "pay"]


@pytest.mark.asyncio
async def test_link_refusing_the_chat_approval_falls_back_to_link(
    broker, local_broker, plan
):
    local_broker.delegated_error = "link_rejected"
    async with client(broker) as api:
        created = await create(api, plan, approval_mode="in_app")
        fallback = await api.post(
            "/v1/checkout/complete",
            json={
                **PRINCIPAL,
                "checkout_id": created["checkout_id"],
                "access_token": "synthetic",
                "approval": approval(created["checkout_id"]),
            },
        )
        body = fallback.json()
        assert body["status"] == "pending_approval"
        assert body["approval_mode"] == "link"
        assert body["approval_url"].startswith("https://app.link.com/")
        assert local_broker.calls == ["create_delegated", "create"]


@pytest.mark.asyncio
async def test_a_duplicate_request_is_explained_not_retried_another_way(
    broker, local_broker, plan
):
    """Link refuses a request matching one still open; asking for Link's own
    approval instead would meet the same refusal."""
    local_broker.delegated_error = "link_duplicate"
    async with client(broker) as api:
        created = await create(api, plan, approval_mode="in_app")
        refused = await api.post(
            "/v1/checkout/complete",
            json={
                **PRINCIPAL,
                "checkout_id": created["checkout_id"],
                "access_token": "synthetic",
                "approval": approval(created["checkout_id"]),
            },
        )
        assert refused.status_code == 422
        assert refused.json()["detail"] == DUPLICATE_REQUEST
        assert local_broker.calls == ["create_delegated"]

        local_broker.create_error = "link_duplicate"
        refused = await api.post(
            "/v1/checkout/create",
            json={**PRINCIPAL, "plan": plan.model_dump(), "access_token": "synthetic"},
        )
        assert refused.json()["detail"] == DUPLICATE_REQUEST


@pytest.mark.asyncio
async def test_a_live_card_is_only_filled_behind_the_egress_proxy(
    broker, local_broker, plan, monkeypatch
):
    live = {
        **PRINCIPAL,
        "plan": plan.model_copy(update={"test_mode": False}).model_dump(),
        "access_token": "synthetic",
    }
    monkeypatch.setenv("COPILOT_LINK_LIVE_PAYMENTS", "true")
    monkeypatch.delenv("CHECKOUT_HTTPS_PROXY", raising=False)
    async with client(broker) as api:
        refused = await api.post("/v1/checkout/create", json=live)
        assert refused.status_code == 422
        assert refused.json()["detail"] == LIVE_PAYMENTS_DISABLED
        assert local_broker.calls == []

        monkeypatch.setenv("CHECKOUT_HTTPS_PROXY", "http://checkout-egress:3128")
        accepted = await api.post("/v1/checkout/create", json=live)
        assert accepted.status_code == 200, accepted.text


@pytest.mark.asyncio
async def test_an_expired_checkout_is_closed_and_its_request_canceled(
    broker, local_broker, plan, monkeypatch
):
    async with client(broker) as api:
        created = await create(api, plan)
        monkeypatch.setattr(time, "time", lambda: created["expires_at"] + 1)
        result = await api.post(
            "/v1/checkout/complete",
            json={
                **PRINCIPAL,
                "checkout_id": created["checkout_id"],
                "access_token": "synthetic",
            },
        )
        assert result.json()["status"] == "expired"
        assert local_broker.calls == ["create", "cancel"]


@pytest.mark.asyncio
async def test_a_new_checkout_replaces_an_unpaid_one_and_cancels_its_request(
    broker, local_broker, plan
):
    async with client(broker) as api:
        first = await create(api, plan)
        second = await create(api, plan)
        assert second["checkout_id"] != first["checkout_id"]
        assert local_broker.calls == ["create", "cancel", "create"]
        stale = await api.post(
            "/v1/checkout/get", json={**PRINCIPAL, "checkout_id": first["checkout_id"]}
        )
        assert stale.status_code == 409


@pytest.mark.asyncio
async def test_an_unpaid_checkout_can_be_reset_without_a_link_status(
    broker, local_broker, plan
):
    async with client(broker) as api:
        created = await create(api, plan)
        reset = await api.post(
            "/v1/checkout/reset",
            json={**PRINCIPAL, "checkout_id": created["checkout_id"]},
        )
        assert reset.status_code == 200
        assert reset.json()["status"] == "browser_reset"
