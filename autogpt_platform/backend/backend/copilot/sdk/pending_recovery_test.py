import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest

from backend.copilot import pending_message_helpers
from backend.copilot.baseline.pending_preflight_recovery_test import (
    queued_session as queued_session,
)
from backend.copilot.budget_gate import AutoContinueGate
from backend.copilot.model import ChatSession
from backend.copilot.pending_message_recovery import restore_pending_messages
from backend.copilot.pending_messages import (
    PendingMessage,
    drain_pending_messages,
    push_pending_message,
)
from backend.copilot.response_model import StreamError, StreamFinish
from backend.copilot.sdk import service
from backend.copilot.sdk.cli_prelaunch_turn_test import _patched_turn, _turns

__all__ = ["queued_session"]


async def test_sdk_consumes_recovered_backlog_and_accepts_new_followups(
    queued_session: ChatSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = queued_session.session_id
    pending = [PendingMessage(content=f"Accepted {i:02}") for i in range(20)]
    newer = PendingMessage(content="Arrived during this turn")
    await restore_pending_messages(session_id, pending)
    client = _turns._make_client_mock(result_message=_turns._make_result_message())

    async def persist(*_args: object, **_kwargs: object) -> bool:
        await push_pending_message(session_id, newer)
        return True

    with _patched_turn(
        lambda **_kwargs: client, queued_session
    ), monkeypatch.context() as scoped:
        scoped.setattr(
            service, "drain_pending_safe", pending_message_helpers.drain_pending_safe
        )
        scoped.setattr(service, "persist_pending_as_user_rows", persist)
        scoped.setattr(
            service,
            "gate_auto_continue",
            AsyncMock(return_value=AutoContinueGate(proceed=False)),
        )
        events = [
            event
            async for event in service.stream_chat_completion_sdk(
                session_id=session_id,
                message="Continue",
                user_id="user-1",
                session=queued_session,
                is_user_message=False,
            )
        ]

    assert any(isinstance(event, StreamFinish) for event in events)
    assert not any(isinstance(event, StreamError) for event in events)
    query = str(client.__aenter__.return_value.query.call_args)
    for message in pending:
        assert message.content in query
    assert await drain_pending_messages(session_id) == [newer]


async def test_empty_sdk_turn_leaves_accepted_followups_queued(
    queued_session: ChatSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = queued_session.session_id
    queued_session.messages[-1].content = "   "
    pending = PendingMessage(content="Keep this followup")
    await push_pending_message(session_id, pending)
    with _patched_turn(AsyncMock(), queued_session), monkeypatch.context() as scoped:
        scoped.setattr(
            service, "drain_pending_safe", pending_message_helpers.drain_pending_safe
        )
        events = [
            event
            async for event in service.stream_chat_completion_sdk(
                session_id=session_id,
                message="   ",
                user_id="user-1",
                session=queued_session,
                is_user_message=False,
            )
        ]

    assert any(
        isinstance(event, StreamError) and event.code == "empty_prompt"
        for event in events
    )
    assert await drain_pending_messages(session_id) == [pending]


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_sdk_restores_backlog_when_startup_persistence_is_interrupted(
    queued_session: ChatSession, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    session_id = queued_session.session_id
    pending = [PendingMessage(content=f"Accepted {i:02}") for i in range(13)]
    await restore_pending_messages(session_id, pending)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_persist(*_args: object, **_kwargs: object) -> bool:
        entered.set()
        await release.wait()
        raise RuntimeError("persistence interrupted")

    async def collect() -> None:
        async for _event in service.stream_chat_completion_sdk(
            session_id=session_id,
            message="Continue",
            user_id="user-1",
            session=queued_session,
            is_user_message=False,
        ):
            pass

    with _patched_turn(AsyncMock(), queued_session), monkeypatch.context() as scoped:
        scoped.setattr(
            service, "drain_pending_safe", pending_message_helpers.drain_pending_safe
        )
        scoped.setattr(service, "persist_pending_as_user_rows", fail_persist)
        turn = asyncio.create_task(collect())
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            newer = PendingMessage(content="Arrived before the interruption")
            await push_pending_message(session_id, newer)
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
            assert recovered == pending + [newer]
        finally:
            release.set()
            turn.cancel()
            with suppress(asyncio.CancelledError, RuntimeError):
                await turn
