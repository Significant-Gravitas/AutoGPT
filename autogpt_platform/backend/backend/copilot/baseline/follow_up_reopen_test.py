"""A redelivered turn that reruns on the baseline engine reopens the follow-up
queue its dead SDK attempt closed (#17247), on the real Lua scripts."""

import contextlib
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis.aioredis import FakeRedis

from backend.copilot import pending_messages, stream_registry
from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.context import set_current_turn_id
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import (
    PendingMessage,
    drain_pending_or_close,
    push_pending_message_if_session_running,
)
from backend.util.tool_call_loop import LLMLoopResponse

MESSAGE = "summarise my week"
FOLLOW_UP = "and next week too"


@pytest.fixture
async def redis() -> AsyncIterator[FakeRedis]:
    async with FakeRedis(decode_responses=True) as fake:
        with contextlib.ExitStack() as stack:
            for module in (pending_messages, stream_registry):
                stack.enter_context(
                    patch.object(
                        module, "get_redis_async", AsyncMock(return_value=fake)
                    )
                )
            chat_db = stack.enter_context(patch.object(stream_registry, "chat_db"))
            chat_db.return_value.set_turn_duration = AsyncMock()
            yield fake


def _baseline_stubs(llm) -> contextlib.ExitStack:
    """Everything a baseline turn reaches outside Redis, stubbed; the pending
    queue's drains stay real."""
    values = {
        "fetch_warm_context": None,
        "download_transcript": None,
        "build_expert_identity_suffix": "",
        "_resolve_baseline_model": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "get_or_create_sandbox": None,
        "_build_system_prompt": ("system prompt", None),
        "build_builder_system_prompt_suffix": "",
        "build_budget_ctx": "",
        "build_session_context": "",
        "build_skills_context": "",
        "build_skills_update_notice": "",
        "build_turn_budget_block": "",
        "persist_and_record_usage": None,
        "persist_pending_as_user_rows": True,
        "enqueue_conversation_turn": None,
        "upload_transcript": None,
    }
    stack = contextlib.ExitStack()
    stack.enter_context(
        patch("backend.copilot.gate.held.answered", new=AsyncMock(return_value=[]))
    )
    for name, value in values.items():
        stack.enter_context(
            patch(
                f"backend.copilot.baseline.service.{name}",
                new=AsyncMock(return_value=value),
            )
        )
    for name in ("upsert_chat_session", "persist_session_safe"):
        stack.enter_context(
            patch(
                f"backend.copilot.baseline.service.{name}",
                new=AsyncMock(side_effect=lambda s, *_a, **_k: s),
            )
        )
    stack.enter_context(
        patch("backend.copilot.baseline.service._baseline_llm_caller", new=llm)
    )
    stack.enter_context(
        patch(
            "backend.copilot.graphiti.context_refresh.refresh_warm_context",
            new=AsyncMock(return_value=None),
        )
    )
    return stack


@pytest.mark.asyncio
async def test_a_rerun_on_the_baseline_engine_reopens_the_queue_its_dead_attempt_closed(
    redis: FakeRedis,
):
    # Its own user, so nothing another test left in a shared Redis applies.
    user_id = f"reopen-{uuid.uuid4()}"
    session = ChatSession.new(user_id, dry_run=False)
    session.title = "Existing chat"
    # An existing chat: a first message also builds its expert context.
    session.messages = [
        ChatMessage(role="user", content="hello", sequence=0),
        ChatMessage(role="assistant", content="hi", sequence=1),
        ChatMessage(role="user", content=MESSAGE, sequence=2),
    ]
    meta = stream_registry.get_session_meta_key(session.session_id)
    await stream_registry.create_session(
        session.session_id, None, "chat_stream", "chat", turn_id="turn-1"
    )
    # The SDK attempt took its last look, closing the queue, then its pod died.
    await drain_pending_or_close(
        session.session_id, session_meta_key=meta, turn_id="turn-1"
    )
    sent_while_answering: list[int | None] = []

    async def llm(messages, tools, *, state):
        if not sent_while_answering:
            sent_while_answering.append(
                await push_pending_message_if_session_running(
                    session.session_id,
                    PendingMessage(content=FOLLOW_UP),
                    session_meta_key=meta,
                )
            )
            # Taken back off, so the rest of the turn runs as it would have:
            # only whether the queue let it in matters here.
            await redis.delete(pending_messages._buffer_key(session.session_id))
        return LLMLoopResponse(response_text="done", tool_calls=[], raw_response=None)

    set_current_turn_id("turn-1")
    try:
        with _baseline_stubs(llm):
            async for _ in stream_chat_completion_baseline(
                session.session_id, MESSAGE, user_id=user_id, session=session
            ):
                pass
    finally:
        set_current_turn_id(None)

    assert sent_while_answering == [1]
