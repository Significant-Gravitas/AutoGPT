import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import pytest
import pytest_asyncio

from backend.copilot import pending_message_helpers, pending_message_recovery
from backend.copilot.baseline import service
from backend.copilot.baseline.pending_recovery_test import baseline_io as baseline_io
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_message_recovery import restore_pending_messages
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    _buffer_key,
    drain_pending_messages,
    peek_pending_count,
    push_pending_message,
)
from backend.data.redis_client import get_redis_async

__all__ = ["baseline_io"]


@pytest_asyncio.fixture
async def queued_session() -> AsyncIterator[ChatSession]:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Existing conversation"
    session.messages = [
        ChatMessage(role="user", content="Earlier task"),
        ChatMessage(role="assistant", content="Earlier reply"),
        ChatMessage(role="user", content="Continue"),
    ]
    try:
        yield session
    finally:
        redis = await get_redis_async()
        await redis.delete(_buffer_key(session.session_id))


@pytest.mark.parametrize("phase", ["_resolve_baseline_model", "_build_system_prompt"])
@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_preflight_failure_restores_backlog_ahead_of_new_arrivals(
    queued_session: ChatSession,
    baseline_io: None,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    failure: str,
) -> None:
    session_id = queued_session.session_id
    pending = [PendingMessage(content=f"Accepted {i}") for i in range(13)]
    await restore_pending_messages(session_id, pending)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_preflight(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await release.wait()
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(service, phase, fail_preflight)
    stream = service.stream_chat_completion_baseline(
        session_id, user_id="user-1", session=queued_session, is_user_message=False
    )
    turn = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert await peek_pending_count(session_id) == 0
        newer = [
            PendingMessage(content=f"New {i}") for i in range(MAX_PENDING_MESSAGES)
        ]
        for message in newer:
            await push_pending_message(session_id, message)
        if failure == "cancel":
            turn.cancel()
        else:
            release.set()
        with pytest.raises(
            asyncio.CancelledError if failure == "cancel" else RuntimeError
        ):
            await asyncio.wait_for(turn, timeout=10)

        recovered = await pending_message_helpers.drain_pending_safe(
            session_id, drain_backlog=True
        )
        assert recovered == pending + newer
    finally:
        release.set()
        turn.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await turn
        await stream.aclose()


async def test_repeated_cancellation_does_not_interrupt_queue_restoration(
    queued_session: ChatSession, baseline_io: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = queued_session.session_id
    pending = PendingMessage(content="Keep this followup")
    await push_pending_message(session_id, pending)
    entered = asyncio.Event()
    recovering = asyncio.Event()
    release = asyncio.Event()

    async def preflight(*_args: object, **_kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    async def restore(session_id: str, messages: list[PendingMessage]) -> None:
        recovering.set()
        await release.wait()
        await restore_pending_messages(session_id, messages)

    monkeypatch.setattr(service, "_resolve_baseline_model", preflight)
    monkeypatch.setattr(pending_message_recovery, "restore_pending_messages", restore)
    stream = service.stream_chat_completion_baseline(
        session_id, user_id="user-1", session=queued_session, is_user_message=False
    )
    turn = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        turn.cancel()
        await asyncio.wait_for(recovering.wait(), timeout=10)
        turn.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, timeout=10)
        assert await drain_pending_messages(session_id) == [pending]
    finally:
        release.set()
        turn.cancel()
        with suppress(asyncio.CancelledError):
            await turn
        await stream.aclose()


async def test_cancellation_during_backlog_drain_preserves_every_accepted_message(
    queued_session: ChatSession, baseline_io: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = queued_session.session_id
    pending = [PendingMessage(content=f"Accepted {i}") for i in range(13)]
    await restore_pending_messages(session_id, pending)
    popped = asyncio.Event()
    release = asyncio.Event()

    async def delayed_pop(
        session_id: str, *, limit: int = MAX_PENDING_MESSAGES
    ) -> list[PendingMessage]:
        batch = await drain_pending_messages(session_id, limit=limit)
        if not popped.is_set():
            popped.set()
            await release.wait()
        return batch

    monkeypatch.setattr(pending_message_helpers, "drain_pending_messages", delayed_pop)
    stream = service.stream_chat_completion_baseline(
        session_id, user_id="user-1", session=queued_session, is_user_message=False
    )
    turn = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(popped.wait(), timeout=10)
        newer = [PendingMessage(content=f"New {i}") for i in range(7)]
        for message in newer:
            await push_pending_message(session_id, message)
        turn.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(turn, timeout=10)

        recovered = await pending_message_helpers.drain_pending_safe(
            session_id, drain_backlog=True
        )
        assert recovered == pending + newer
    finally:
        release.set()
        turn.cancel()
        with suppress(asyncio.CancelledError):
            await turn
        await stream.aclose()
