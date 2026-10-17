from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.copilot import pending_message_recovery as recovery
from backend.copilot.pending_messages import (
    PendingMessage,
    _buffer_key,
    drain_pending_messages,
    push_pending_message,
)
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
