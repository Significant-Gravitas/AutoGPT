"""Queued retries through real Redis, Prisma writes, and session caching."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from autogpt_libs import auth
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prisma.models import ChatMessage as StoredMessage
from prisma.models import ChatSession as StoredSession
from prisma.models import User

from backend.api.features.chat import routes
from backend.copilot import db as chat_db
from backend.copilot import pending_message_helpers as helpers
from backend.copilot import stream_registry
from backend.copilot.expert_kickoff import scoped_client_message_id
from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.pending_message_helpers import persist_pending_as_user_rows
from backend.copilot.pending_message_receipts import pending_message_receipt_key
from backend.copilot.pending_message_reconciliation import reconcile_pending_rows
from backend.copilot.pending_messages import (
    PendingMessage,
    _client_message_key,
    drain_pending_messages,
    push_pending_message,
)
from backend.data import db
from backend.data.redis_client import get_redis_async


@pytest_asyncio.fixture
async def persisted_session() -> AsyncIterator[ChatSession]:
    connected = db.is_connected()
    if not connected:
        await db.connect()
    user_id, session_id = str(uuid4()), str(uuid4())
    now = datetime.now(UTC)
    await User.prisma().create(
        data={"id": user_id, "email": f"pending-retry-{user_id}@example.com"}
    )
    await StoredSession.prisma().create(data={"id": session_id, "userId": user_id})
    try:
        yield ChatSession(
            session_id=session_id,
            user_id=user_id,
            usage=[],
            messages=[],
            started_at=now,
            updated_at=now,
        )
    finally:
        await User.prisma().delete(where={"id": user_id})
        redis = await get_redis_async()
        async for key in redis.scan_iter(match=f"*{session_id}*"):
            await redis.delete(key)
        if not connected:
            await db.disconnect()


async def test_duplicate_pending_batch_persists_once_and_restores_history(
    persisted_session: ChatSession,
) -> None:
    session = persisted_session
    first = PendingMessage(content="First")
    second = PendingMessage(content="Second")
    pending = [first, second, first]

    assert await persist_pending_as_user_rows(session, None, pending, log_prefix="test")
    rows = await StoredMessage.prisma().find_many(
        where={"sessionId": session.session_id}, order={"sequence": "asc"}
    )
    assert [(row.id, row.content) for row in rows] == [
        (first.id, "First"),
        (second.id, "Second"),
    ]
    restored = await get_chat_session(session.session_id)
    assert restored is not None
    assert [(m.id, m.sequence) for m in restored.messages] == [
        (row.id, row.sequence) for row in rows
    ]
    assert await drain_pending_messages(session.session_id) == []


async def test_already_persisted_pending_id_is_not_requeued_forever(
    persisted_session: ChatSession,
) -> None:
    session = persisted_session
    pending = PendingMessage(content="Persisted before the retry")
    await chat_db.add_chat_messages_batch(
        session.session_id,
        [{"id": pending.id, "role": "user", "content": pending.content}],
        start_sequence=0,
    )
    await push_pending_message(session.session_id, pending)
    drained = await drain_pending_messages(session.session_id)

    assert await persist_pending_as_user_rows(session, None, drained, log_prefix="test")
    assert await drain_pending_messages(session.session_id) == []
    assert (
        await StoredMessage.prisma().count(where={"sessionId": session.session_id}) == 1
    )
    assert (
        await chat_db.chat_message_has_assistant_reply(pending.id, session.session_id)
        is False
    )


async def test_competing_writer_pk_collision_recovers_on_next_drain(
    persisted_session: ChatSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = persisted_session
    pending = PendingMessage(content="Racing writer")
    real_reconcile = reconcile_pending_rows

    async def reconcile_then_compete(session, messages, content_of):
        rows = await real_reconcile(session, messages, content_of)
        await chat_db.add_chat_messages_batch(
            session.session_id,
            [{"id": pending.id, "role": "user", "content": pending.content}],
            start_sequence=0,
        )
        return rows

    with monkeypatch.context() as race:
        race.setattr(helpers, "reconcile_pending_rows", reconcile_then_compete)
        assert not await persist_pending_as_user_rows(
            session, None, [pending], log_prefix="test"
        )
    assert session.messages == []
    retry = await drain_pending_messages(session.session_id)
    assert [message.id for message in retry] == [pending.id]
    assert await persist_pending_as_user_rows(session, None, retry, log_prefix="test")
    assert await drain_pending_messages(session.session_id) == []
    restored = await get_chat_session(session.session_id)
    assert restored is not None
    assert [(message.id, message.sequence) for message in restored.messages] == [
        (pending.id, 0)
    ]
    assert (
        await StoredMessage.prisma().count(where={"sessionId": session.session_id}) == 1
    )


async def test_http_retry_uses_persisted_row_after_redis_receipt_expires(
    persisted_session: ChatSession,
) -> None:
    session = persisted_session
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[auth.get_user_id] = lambda: session.user_id
    await stream_registry.create_session(
        session_id=session.session_id,
        user_id=session.user_id,
        tool_call_id="retry-test",
        tool_name="retry-test",
    )
    body = {"message": "One real pending send", "message_id": str(uuid4())}
    path = f"/sessions/{session.session_id}/messages/pending"
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post(path, json=body)).status_code == 200
        pending = await drain_pending_messages(session.session_id)
        assert await persist_pending_as_user_rows(
            session, None, pending, log_prefix="test"
        )
        redis = await get_redis_async()
        message_id = scoped_client_message_id(
            session.user_id, session.session_id, body["message_id"]
        )
        await redis.delete(_client_message_key(session.session_id, message_id))
        await redis.delete(pending_message_receipt_key(session.session_id, message_id))
        await redis.hset(
            stream_registry.get_session_meta_key(session.session_id),
            mapping={"status": "completed"},
        )
        response = await client.post(path, json=body)
        assert response.status_code == 200
        assert response.json()["turn_in_flight"] is True
        assert await drain_pending_messages(session.session_id) == []
        assert (
            await StoredMessage.prisma().count(where={"sessionId": session.session_id})
            == 1
        )
