from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.copilot import pending_message_recovery as recovery
from backend.copilot.gate.held import HeldResult
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import persist_unfolded_rows
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    _buffer_key,
    drain_pending_messages,
    push_pending_message,
)
from backend.copilot.tool_image import ToolImage
from backend.data.redis_client import get_redis_async


@pytest_asyncio.fixture
async def session_id() -> AsyncIterator[str]:
    value = str(uuid4())
    try:
        yield value
    finally:
        redis = await get_redis_async()
        await redis.delete(_buffer_key(value))


async def test_recovery_is_ordered_idempotent_and_preserves_longer_expiry(
    session_id: str,
) -> None:
    first, second, newer = [PendingMessage(content=text) for text in ("a", "b", "c")]
    await push_pending_message(session_id, newer)
    await push_pending_message(session_id, first)
    redis = await get_redis_async()
    key = _buffer_key(session_id)
    await redis.rpush(key, "malformed entry")
    await redis.expire(key, 7200)

    for _ in range(2):
        await recovery.restore_pending_messages(session_id, [first, second, first])

    assert 7100 < await redis.ttl(key) <= 7200
    assert await redis.lrange(key, 0, -1) == [
        message.model_dump_json() for message in (first, second, newer)
    ] + ["malformed entry"]


async def test_notification_failure_does_not_undo_recovery(
    session_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis = await get_redis_async()
    transport = Mock(wraps=redis)
    transport.execute_command = AsyncMock(
        side_effect=OSError("notification unavailable")
    )
    monkeypatch.setattr(recovery, "get_redis_async", AsyncMock(return_value=transport))
    pending = PendingMessage(content="Keep this")

    await recovery.restore_pending_messages(session_id, [pending])

    assert await drain_pending_messages(session_id) == [pending]


async def test_empty_recovery_does_not_touch_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_redis = AsyncMock()
    monkeypatch.setattr(recovery, "get_redis_async", get_redis)

    await recovery.restore_pending_messages("unused", [])

    get_redis.assert_not_awaited()


@pytest.mark.parametrize("newer_count", [0, MAX_PENDING_MESSAGES])
async def test_unfolded_image_batch_is_requeued_once_in_order(
    session_id: str, newer_count: int
) -> None:
    session = ChatSession.new("user-1", dry_run=False)
    session.session_id = session_id
    pending = [
        PendingMessage(content="before"),
        HeldResult(
            content="Approved image: " + "x" * 40_000,
            metadata={"held_call": {"review_id": "image-review"}},
            model_image=ToolImage(mime_type="image/png", data_base64="cGl4ZWxz"),
        ),
        PendingMessage(content="after"),
    ]
    expected = list(pending)
    newer = [PendingMessage(content=f"newer-{index}") for index in range(newer_count)]
    for message in newer:
        await push_pending_message(session_id, message)

    await persist_unfolded_rows(session, pending, log_prefix="test")
    await persist_unfolded_rows(session, pending, log_prefix="test")

    assert not pending
    assert not session.messages
    recovered = await drain_pending_messages(session_id)
    recovered.extend(await drain_pending_messages(session_id))
    assert recovered == expected + newer
