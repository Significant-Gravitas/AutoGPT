"""HTTP-to-Redis retry journeys with isolated keys in the real Redis cluster.

Session ownership and active-turn discovery are stubbed; request validation,
pending-message admission, Redis scripts, and draining run unchanged.
"""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from autogpt_libs import auth
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.features.chat import routes
from backend.copilot import pending_messages
from backend.copilot.expert_kickoff import scoped_client_message_id
from backend.copilot.model import ChatSessionInfo
from backend.copilot.pending_message_receipts import (
    pending_message_receipt_key,
    pending_message_was_received,
)
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    _client_message_key,
    accept_client_message,
    claim_client_message,
    drain_pending_messages,
    push_pending_message,
)
from backend.copilot.stream_registry import get_session_meta_key
from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import RedisListCapacityExceeded


@pytest_asyncio.fixture
async def pending_client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[AsyncClient, str, str]]:
    user_id, session_id = str(uuid4()), str(uuid4())
    now = datetime.now(timezone.utc)
    session = ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        usage=[],
        started_at=now,
        updated_at=now,
    )
    monkeypatch.setattr(
        routes, "_validate_and_get_writable_session", AsyncMock(return_value=session)
    )
    monkeypatch.setattr(routes, "is_turn_in_flight", AsyncMock(return_value=True))
    monkeypatch.setattr(
        routes, "chat_message_has_assistant_reply", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(routes, "enforce_payment_paywall", AsyncMock())
    monkeypatch.setattr(
        "backend.copilot.briefing.scheduling.ensure_morning_briefing_scheduled",
        AsyncMock(),
    )
    redis = await get_redis_async()
    await redis.hset(get_session_meta_key(session_id), mapping={"status": "running"})
    await redis.expire(get_session_meta_key(session_id), 120)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=_test_app(user_id)), base_url="http://test"
        ) as client:
            yield client, user_id, session_id
    finally:
        async for key in redis.scan_iter(match=f"*{{{session_id}}}*"):
            await redis.delete(key)
        async for key in redis.scan_iter(
            match=f"copilot:pending:client-msg:{session_id}:*"
        ):
            await redis.delete(key)
        await redis.delete(f"copilot:pending:calls:{user_id}")


def _test_app(user_id: str) -> FastAPI:
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[auth.get_user_id] = lambda: user_id
    app.dependency_overrides[auth.get_request_context] = lambda: auth.RequestContext(
        user_id=user_id,
        org_id="test-org",
        team_id=None,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
    return app


@pytest.mark.parametrize("endpoint", ["messages/pending", "stream"])
async def test_pending_http_retry_is_one_message(
    pending_client: tuple[AsyncClient, str, str],
    endpoint: str,
) -> None:
    client, user_id, session_id = pending_client
    body = {"message": "Include the weekly totals", "message_id": str(uuid4())}

    responses = await asyncio.gather(
        *[
            client.post(f"/sessions/{session_id}/{endpoint}", json=body)
            for _ in range(2)
        ]
    )

    assert [response.status_code for response in responses] == [200, 200]
    drained = await drain_pending_messages(session_id)
    assert len(drained) == 1
    assert drained[0].id == scoped_client_message_id(
        user_id, session_id, body["message_id"]
    )


async def test_separate_pending_sends_can_have_identical_text(
    pending_client: tuple[AsyncClient, str, str],
) -> None:
    client, _, session_id = pending_client
    for _ in range(2):
        response = await client.post(
            f"/sessions/{session_id}/messages/pending",
            json={"message": "Yes", "message_id": str(uuid4())},
        )
        assert response.status_code == 200

    drained = await drain_pending_messages(session_id)
    assert [message.content for message in drained] == ["Yes", "Yes"]
    assert len({message.id for message in drained}) == 2


@pytest.mark.parametrize("claim_expired", [False, True])
async def test_pending_retry_after_drain_and_completion_does_not_requeue(
    pending_client: tuple[AsyncClient, str, str],
    monkeypatch: pytest.MonkeyPatch,
    claim_expired: bool,
) -> None:
    client, user_id, session_id = pending_client
    body = {"message": "Include the weekly totals", "message_id": str(uuid4())}
    path = f"/sessions/{session_id}/messages/pending"
    assert (await client.post(path, json=body)).status_code == 200
    assert len(await drain_pending_messages(session_id)) == 1

    redis = await get_redis_async()
    if claim_expired:
        message_id = scoped_client_message_id(user_id, session_id, body["message_id"])
        await redis.delete(_client_message_key(session_id, message_id))
    await redis.hset(get_session_meta_key(session_id), mapping={"status": "completed"})
    monkeypatch.setattr(routes, "is_turn_in_flight", AsyncMock(return_value=False))
    response = await client.post(path, json=body)

    assert response.status_code == 200
    assert response.json()["turn_in_flight"] is True
    stream_retry = await client.post(f"/sessions/{session_id}/stream", json=body)
    assert stream_retry.status_code == 200
    assert "data-follow-up-queued" in stream_retry.text
    assert "[DONE]" in stream_retry.text
    assert await drain_pending_messages(session_id) == []


async def test_persisted_pending_retry_is_accepted_after_receipt_expiry(
    pending_client: tuple[AsyncClient, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, session_id = pending_client
    monkeypatch.setattr(routes, "is_turn_in_flight", AsyncMock(return_value=False))
    monkeypatch.setattr(
        routes, "chat_message_has_assistant_reply", AsyncMock(return_value=False)
    )

    response = await client.post(
        f"/sessions/{session_id}/messages/pending",
        json={"message": "Already saved", "message_id": str(uuid4())},
    )

    assert response.status_code == 200
    assert response.json()["turn_in_flight"] is True
    assert await drain_pending_messages(session_id) == []


async def test_full_pending_queue_rejects_new_send_without_eviction(
    pending_client: tuple[AsyncClient, str, str],
) -> None:
    client, user_id, session_id = pending_client
    bodies = [
        {"message": f"Send {i}", "message_id": str(uuid4())}
        for i in range(MAX_PENDING_MESSAGES + 1)
    ]
    path = f"/sessions/{session_id}/messages/pending"
    for body in bodies[:-1]:
        assert (await client.post(path, json=body)).status_code == 200

    rejected = await client.post(path, json=bodies[-1])
    assert rejected.status_code == 429
    assert rejected.headers["X-Error-Code"] == "pending_buffer_full"
    assert rejected.headers["Retry-After"] == "5"
    assert (await client.post(path, json=bodies[0])).status_code == 200
    drained = await drain_pending_messages(session_id)
    assert [m.id for m in drained] == [
        scoped_client_message_id(user_id, session_id, body["message_id"])
        for body in bodies[:-1]
    ]
    assert (await client.post(path, json=bodies[-1])).status_code == 200
    assert len(await drain_pending_messages(session_id)) == 1


async def test_pending_retry_after_receipt_expiry_keeps_one_copy(
    pending_client: tuple[AsyncClient, str, str],
) -> None:
    client, user_id, session_id = pending_client
    body = {"message": "First", "message_id": str(uuid4())}
    path = f"/sessions/{session_id}/messages/pending"
    assert (await client.post(path, json=body)).status_code == 200
    assert (
        await client.post(path, json={"message": "Second", "message_id": str(uuid4())})
    ).status_code == 200
    redis = await get_redis_async()
    message_id = scoped_client_message_id(user_id, session_id, body["message_id"])
    await redis.delete(_client_message_key(session_id, message_id))
    await redis.pexpire(pending_message_receipt_key(session_id, message_id), 1)
    await asyncio.sleep(0.01)
    assert (await client.post(path, json=body)).status_code == 200
    drained = await drain_pending_messages(session_id)
    assert [message.content for message in drained] == ["First", "Second"]


async def test_internal_send_cannot_evict_http_send(
    pending_client: tuple[AsyncClient, str, str],
) -> None:
    client, _, session_id = pending_client
    path = f"/sessions/{session_id}/messages/pending"
    for _ in range(MAX_PENDING_MESSAGES):
        assert (
            await client.post(
                path, json={"message": "Keep", "message_id": str(uuid4())}
            )
        ).status_code == 200
    with pytest.raises(RedisListCapacityExceeded):
        await push_pending_message(session_id, PendingMessage(content="Overflow"))
    assert all(m.content == "Keep" for m in await drain_pending_messages(session_id))


@pytest.mark.parametrize("endpoint", ["messages/pending", "stream"])
@pytest.mark.parametrize("expire_before_settlement", [False, True])
async def test_restored_receipt_expires_with_undrained_queue(
    pending_client: tuple[AsyncClient, str, str],
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    expire_before_settlement: bool,
) -> None:
    client, user_id, session_id = pending_client
    path = f"/sessions/{session_id}/messages/pending"
    body = {"message": "Expiring send", "message_id": str(uuid4())}
    assert (await client.post(path, json=body)).status_code == 200
    message_id = scoped_client_message_id(user_id, session_id, body["message_id"])
    redis = await get_redis_async()
    await redis.delete(_client_message_key(session_id, message_id))
    await redis.delete(pending_message_receipt_key(session_id, message_id))
    queue_key = f"copilot:pending:{{{session_id}}}"
    await redis.pexpire(queue_key, 2000)
    original_cas = pending_messages.string_compare_and_set

    async def poll_for_expiry() -> None:
        while await redis.exists(queue_key):
            await asyncio.sleep(0.01)

    async def wait_for_expiry() -> None:
        await asyncio.wait_for(poll_for_expiry(), timeout=10)

    async def delayed_settlement(*args: Any, **kwargs: Any) -> bool:
        if expire_before_settlement:
            await wait_for_expiry()
        return await original_cas(*args, **kwargs)

    with monkeypatch.context() as delay:
        delay.setattr(pending_messages, "string_compare_and_set", delayed_settlement)
        assert (await client.post(path, json=body)).status_code == 200
    await wait_for_expiry()
    assert await drain_pending_messages(session_id) == []
    assert not await pending_message_was_received(session_id, message_id)
    assert (
        await client.post(f"/sessions/{session_id}/{endpoint}", json=body)
    ).status_code == 200
    assert [message.id for message in await drain_pending_messages(session_id)] == [
        message_id
    ]


async def test_missing_receipt_does_not_leave_an_accepted_claim(
    pending_client: tuple[AsyncClient, str, str],
) -> None:
    client, user_id, session_id = pending_client
    body = {"message": "Retry after expiry", "message_id": str(uuid4())}
    message_id = scoped_client_message_id(user_id, session_id, body["message_id"])
    assert await claim_client_message(session_id, message_id, "first") == "claimed"
    await accept_client_message(session_id, message_id, "first", pending=True)

    response = await client.post(f"/sessions/{session_id}/messages/pending", json=body)

    assert response.status_code == 200
    assert [message.id for message in await drain_pending_messages(session_id)] == [
        message_id
    ]
