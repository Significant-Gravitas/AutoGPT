import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import ChatMessage as StoredMessage

from backend.copilot import db as chat_db
from backend.copilot import pending_message_helpers as helpers
from backend.copilot.baseline import service
from backend.copilot.context import set_execution_context
from backend.copilot.gate.held import HeldResult
from backend.copilot.gate.held_engines_test import _baseline_patches
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import clear_pending_messages_unsafe
from backend.copilot.pending_retries_db_test import persisted_session
from backend.copilot.response_model import StreamStatus
from backend.copilot.tree import root_envelope

__all__ = ["persisted_session"]


@pytest.mark.parametrize("phase", ["lookup", "write"])
async def test_stop_preserves_completed_approval_results_after_queue_cleanup(
    persisted_session: ChatSession, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = persisted_session
    session.title = "Existing conversation"
    results = [
        HeldResult(
            content=f'<held_call_result tool="{tool}">{result}</held_call_result>',
            metadata={"held_call": {"tool_name": tool}},
        )
        for tool, result in [
            ("post_to_chat_platform", "posted"),
            ("send_email", "sent"),
        ]
    ]
    delivered = []

    async def resolve(_user_id, _session, deliver, **_kwargs):
        for result in results:
            deliver(result)
            delivered.append(result.id)
            yield StreamStatus(message="Action finished")

    async def model_loop(**_kwargs):
        raise AssertionError("A cancelled turn must not reach the model")
        yield

    entered = asyncio.Event()
    real_lookup = chat_db.get_chat_messages_by_ids
    real_persist = helpers.persist_session_safe

    async def interrupt_lookup(*args):
        if phase == "lookup" and not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await real_lookup(*args)

    async def interrupt_write(*args):
        if phase == "write" and not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return await real_persist(*args)

    monkeypatch.setattr(chat_db, "get_chat_messages_by_ids", interrupt_lookup)
    monkeypatch.setattr(helpers, "persist_session_safe", interrupt_write)
    with (
        _baseline_patches(resolve, helpers.persist_pending_as_user_rows, model_loop),
        patch.object(service, "get_or_create_sandbox", AsyncMock(return_value=None)),
    ):

        async def run_turn():
            try:
                async for _ in service.stream_chat_completion_baseline(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    session=session,
                    is_user_message=False,
                    envelope=root_envelope("held-stop", session_id=session.session_id),
                ):
                    pass
            finally:
                set_execution_context(None, None)

        task = asyncio.create_task(run_turn())
        try:
            await asyncio.wait_for(entered.wait(), timeout=30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=30)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    await clear_pending_messages_unsafe(session.session_id)
    rows = await StoredMessage.prisma().find_many(
        where={"sessionId": session.session_id}, order={"sequence": "asc"}
    )
    assert delivered == [result.id for result in results]
    assert [(row.id, row.content) for row in rows] == [
        (result.id, result.content) for result in results
    ]
