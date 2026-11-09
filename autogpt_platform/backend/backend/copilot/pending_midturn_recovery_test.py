import asyncio
from unittest.mock import AsyncMock

import pytest
from prisma.models import ChatMessage as StoredMessage

from backend.copilot import db as chat_db
from backend.copilot import pending_message_helpers as helpers
from backend.copilot.baseline import service
from backend.copilot.baseline.pending_recovery_test import _final_stream
from backend.copilot.baseline.pending_recovery_test import baseline_io as baseline_io
from backend.copilot.model import ChatMessage, ChatSession, get_chat_session
from backend.copilot.pending_messages import (
    MAX_PENDING_MESSAGES,
    PendingMessage,
    drain_pending_for_persist,
    drain_pending_messages,
    push_pending_message,
    stash_pending_for_persist,
)
from backend.copilot.pending_retries_db_test import persisted_session
from backend.copilot.tool_image import ToolImage
from backend.copilot.transcript_builder import TranscriptBuilder

__all__ = ["persisted_session", "baseline_io"]


@pytest.mark.parametrize("with_image", [False, True])
async def test_cancel_after_commit_delivers_recovered_content_once_to_next_model_turn(
    persisted_session: ChatSession,
    baseline_io: None,
    monkeypatch: pytest.MonkeyPatch,
    with_image: bool,
) -> None:
    session = persisted_session
    pending = PendingMessage(
        content="Recovered accounting follow-up",
        model_image=(
            ToolImage(mime_type="image/png", data_base64="cGl4ZWxz")
            if with_image
            else None
        ),
    )
    await push_pending_message(session.session_id, pending)
    drained = await drain_pending_messages(session.session_id)
    committed = asyncio.Event()
    real_persist = helpers.persist_session_safe

    async def stop_after_commit(*args):
        result = await real_persist(*args)
        assert result.messages[-1].sequence is not None
        committed.set()
        await asyncio.Event().wait()

    with monkeypatch.context() as interrupted:
        interrupted.setattr(helpers, "persist_session_safe", stop_after_commit)
        task = asyncio.create_task(
            helpers.persist_pending_as_user_rows(
                session, None, drained, log_prefix="test"
            )
        )
        try:
            await asyncio.wait_for(committed.wait(), 10)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    restored = await get_chat_session(session.session_id)
    assert restored is not None
    restored.title = "Existing conversation"
    restored.messages.append(ChatMessage(role="user", content="Continue", sequence=1))
    provider = AsyncMock(return_value=_final_stream())
    monkeypatch.setattr(service, "call_provider_stream", provider)
    async for _ in service.stream_chat_completion_baseline(
        session.session_id,
        user_id=session.user_id,
        session=restored,
        is_user_message=False,
    ):
        pass
    assert provider.await_args is not None
    messages = provider.await_args.kwargs["messages"]
    assert sum(pending.content in str(row["content"]) for row in messages) == 1
    assert str(messages).count("data:image/png;base64,cGl4ZWxz") == int(with_image)
    assert await drain_pending_messages(session.session_id) == []
    assert (
        await StoredMessage.prisma().count(
            where={"sessionId": session.session_id, "id": pending.id}
        )
        == 1
    )


@pytest.mark.parametrize("interrupted", ["lookup", "write", "committed"])
@pytest.mark.parametrize("cancel_again", [False, True])
async def test_cancelled_midturn_batch_is_recovered_and_persisted_once(
    persisted_session: ChatSession,
    monkeypatch: pytest.MonkeyPatch,
    interrupted: str,
    cancel_again: bool,
) -> None:
    session = persisted_session
    accepted = [PendingMessage(content=f"Accepted {i}") for i in range(2)]
    for message in accepted:
        await push_pending_message(session.session_id, message)
    await stash_pending_for_persist(
        session.session_id, await drain_pending_messages(session.session_id)
    )
    drained = await drain_pending_for_persist(session.session_id)
    newer = [PendingMessage(content=f"New {i}") for i in range(MAX_PENDING_MESSAGES)]
    for message in newer:
        await push_pending_message(session.session_id, message)

    transcript = TranscriptBuilder()
    transcript.append_user("Earlier turn")
    before = transcript.to_jsonl()
    blocked = asyncio.Event()
    real_persist = helpers.persist_session_safe
    real_restore = helpers.restore_pending_messages
    restoring = asyncio.Event()
    release_restore = asyncio.Event()

    async def blocked_restore(*args):
        restoring.set()
        await release_restore.wait()
        await real_restore(*args)

    async def blocked_lookup(*_args):
        blocked.set()
        await asyncio.Event().wait()

    async def blocked_persist(*args):
        if interrupted == "committed":
            await real_persist(*args)
        blocked.set()
        await asyncio.Event().wait()

    with monkeypatch.context() as interruption:
        if cancel_again:
            interruption.setattr(helpers, "restore_pending_messages", blocked_restore)
        if interrupted == "lookup":
            interruption.setattr(chat_db, "get_chat_messages_by_ids", blocked_lookup)
        else:
            interruption.setattr(helpers, "persist_session_safe", blocked_persist)
        task = asyncio.create_task(
            helpers.persist_pending_as_user_rows(
                session, transcript, drained, log_prefix="test"
            )
        )
        try:
            await asyncio.wait_for(blocked.wait(), timeout=10)
        finally:
            task.cancel()
            if cancel_again:
                try:
                    await asyncio.wait_for(restoring.wait(), timeout=10)
                    task.cancel()
                finally:
                    release_restore.set()
            with pytest.raises(asyncio.CancelledError):
                await task

    recovered = await drain_pending_messages(session.session_id)
    recovered.extend(await drain_pending_messages(session.session_id))
    assert recovered == accepted + newer
    assert transcript.to_jsonl() == before
    assert await helpers.persist_pending_as_user_rows(
        session, transcript, recovered[: len(accepted)], log_prefix="test"
    )
    rows = await StoredMessage.prisma().find_many(
        where={"sessionId": session.session_id}, order={"sequence": "asc"}
    )
    assert [(row.id, row.content) for row in rows] == [
        (message.id, message.content) for message in accepted
    ]
    assert await drain_pending_for_persist(session.session_id) == []
