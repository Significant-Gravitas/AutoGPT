import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
from prisma.models import User

from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    PendingMessageContext,
    _buffer_key,
    drain_pending_messages,
    peek_pending_messages,
    push_pending_message,
)
from backend.data import db
from backend.data.redis_client import get_redis_async
from backend.data.user import get_or_create_user

from .recording import persisted_session
from .scripted import sdk_service_turn


@pytest.fixture
async def user_id() -> AsyncIterator[str]:
    await db.connect()
    value = str(uuid.uuid4())
    await get_or_create_user({"sub": value, "email": f"{value}@recovery.invalid"})
    try:
        yield value
    finally:
        await User.prisma().delete_many(where={"id": value})


@pytest.mark.parametrize("recursive_error", [False, True])
async def test_auto_continue_restores_accepted_batch_ahead_of_full_queue(
    user_id: str, recursive_error: bool
) -> None:
    session = await persisted_session(user_id, "Finish the current task")
    session.title = "Existing conversation"
    older = [
        PendingMessage(
            content=f"Accepted {i}",
            file_ids=[f"file-{i}"],
            context=PendingMessageContext(url=f"https://example.test/{i}"),
            metadata={"from_session_id": "sender"},
        )
        for i in range(3)
    ]
    newer = [PendingMessage(content=f"New {i}") for i in range(MAX_PENDING_MESSAGES)]
    locks = 0

    def lock(*_: object, owner_id: str = "", **__: object) -> MagicMock:
        nonlocal locks
        locks += 1
        first = locks == 1

        async def acquire() -> str:
            if first:
                return owner_id
            for pending in newer:
                await push_pending_message(session.session_id, pending)
            if recursive_error:
                raise RuntimeError("recursive turn failed")
            return "competing-turn"

        return MagicMock(
            try_acquire=AsyncMock(side_effect=acquire),
            refresh=AsyncMock(),
            release=AsyncMock(),
        )

    async def run_turn() -> None:
        async for _ in sdk_service_turn(
            session,
            str(uuid.uuid4()),
            [
                [
                    AssistantMessage(content=[TextBlock(text="Done")], model="drift"),
                    ResultMessage(
                        subtype="success",
                        result="Done",
                        duration_ms=100,
                        duration_api_ms=50,
                        is_error=False,
                        num_turns=1,
                        session_id="drift",
                    ),
                ]
            ],
            queued_after_first=older,
            lock_factory=lock,
        ):
            pass

    try:
        for pending in older:
            await push_pending_message(session.session_id, pending)
        assert await drain_pending_messages(session.session_id) == older
        if recursive_error:
            with pytest.raises(RuntimeError, match="recursive turn failed"):
                await run_turn()
        else:
            await run_turn()
        assert locks == 2
        assert await peek_pending_messages(session.session_id) == older + newer
    finally:
        redis = await get_redis_async()
        await redis.delete(_buffer_key(session.session_id))
