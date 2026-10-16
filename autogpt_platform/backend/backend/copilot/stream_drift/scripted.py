"""Copilot engines driven by a script instead of a model, for the drift suite."""

import contextlib
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from claude_agent_sdk import Message

from backend.copilot.budget_gate import AutoContinueGate
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamBaseResponse, StreamStart
from backend.copilot.sdk import service as sdk
from backend.copilot.sdk.compaction import CompactionTracker
from backend.copilot.sdk.response_adapter import SDKResponseAdapter
from backend.copilot.stream_checkpoint import turn_checkpoint
from backend.copilot.tree import root_envelope
from backend.copilot.turn_budget import TurnBudget

# Its presence alone picks the OpenRouter transport the fixtures were recorded
# under; without a key (fork PRs have no secrets) the turns record differently.
UNSENT_MODEL_KEY = "test-placeholder-not-a-key"


def session_with_prompt(prompt: str) -> ChatSession:
    session = ChatSession.new("drift-user", dry_run=False)
    session.title = "Drift fixture"
    session.messages.append(ChatMessage(role="user", content=prompt))
    return session


async def sdk_turn(
    session: ChatSession, messages: list[Message]
) -> AsyncGenerator[StreamBaseResponse, None]:
    """The SDK engine's consume loop, flushes included, over a scripted CLI;
    then the turn-end persist and checkpoint of its ``finally``.

    Persistence goes through ``sdk.upsert_chat_session``; patch it to record.
    """
    message_id = str(uuid.uuid4())
    turn_start = len(session.messages)
    ctx = sdk._StreamContext(
        session=session,
        session_id=session.session_id,
        log_prefix="[drift]",
        sdk_cwd="/tmp/drift",
        current_message="",
        file_ids=None,
        message_id=message_id,
        attachments=MagicMock(image_blocks=[]),
        compaction=CompactionTracker(),
        lock=MagicMock(refresh=AsyncMock()),
        turn_start=turn_start,
    )
    state = sdk._RetryState(
        options=MagicMock(),
        query_message="",
        compaction_stats=None,
        use_resume=False,
        transcript_msg_count=0,
        adapter=SDKResponseAdapter(
            message_id=message_id, session_id=session.session_id
        ),
        transcript_builder=MagicMock(),
        usage=sdk._TokenUsage(),
    )
    acc = sdk._StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )
    now = time.monotonic()
    loop_state = sdk._SDKLoopState(last_real_msg_time=now, last_flush_time=now)

    async def scripted_cli(*_: Any, **__: Any) -> AsyncGenerator[Message, None]:
        for message in messages:
            yield message

    yield StreamStart(messageId=message_id, sessionId=session.session_id)
    with patch.object(sdk, "_iter_sdk_messages", scripted_cli):
        async for event in sdk._consume_sdk_until_done(
            MagicMock(), ctx, state, acc, loop_state
        ):
            yield event
    await sdk.upsert_chat_session(session)
    checkpoint = turn_checkpoint(session.messages, turn_start)
    if checkpoint is not None:
        yield checkpoint


async def sdk_service_turn(
    session: ChatSession,
    turn_id: str,
    queries: list[list[Message]],
    *,
    queued_after_first: list[PendingMessage],
    held_results: list[PendingMessage] | None = None,
    queued_at_start: list[PendingMessage] | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """The whole SDK engine over a scripted CLI, one list of messages per
    query, with ``queued_after_first`` arriving while the first one runs:
    its turn-end persist and auto-continue included. ``held_results`` are the
    answered cards' results the turn opens with, ``queued_at_start`` the
    follow-ups drained before its first query. Only what reaches beyond the
    process (CLI, sandbox, RPC services, billing) is stubbed."""
    scripted = iter(queries)

    def cli(*_: Any, **__: Any) -> AsyncMock:
        messages = next(scripted)

        async def receive() -> AsyncGenerator[Message, None]:
            for message in messages:
                yield message

        client = MagicMock(receive_response=receive, query=AsyncMock())
        connection = AsyncMock()
        connection.__aenter__.return_value = client
        return connection

    def lock(*_: Any, owner_id: str = "", **__: Any) -> MagicMock:
        return MagicMock(
            try_acquire=AsyncMock(return_value=owner_id),
            refresh=AsyncMock(),
            release=AsyncMock(),
        )

    stubs: dict[str, Any] = {
        "ClaudeSDKClient": MagicMock(side_effect=cli),
        "AsyncClusterLock": MagicMock(side_effect=lock),
        "_make_sdk_cwd": MagicMock(return_value=tempfile.mkdtemp()),
        "propagate_attributes": MagicMock(),
        "_build_system_prompt": AsyncMock(return_value="System prompt"),
        "fetch_business_understanding": AsyncMock(return_value=None),
        "download_transcript": AsyncMock(return_value=None),
        "upload_transcript": AsyncMock(),
        "create_copilot_mcp_server": MagicMock(),
        "create_security_hooks": MagicMock(),
        "get_copilot_tool_names": MagicMock(return_value=[]),
        "get_sdk_disallowed_tools": MagicMock(return_value=[]),
        "route_launch_env": MagicMock(return_value={}),
        "set_execution_context": MagicMock(),
        "get_user_tier": AsyncMock(return_value=None),
        "resolve_turn_budget": AsyncMock(
            return_value=TurnBudget(usd=100.0, limiting="task")
        ),
        "gate_auto_continue": AsyncMock(return_value=AutoContinueGate(proceed=True)),
        "drain_pending_safe": AsyncMock(
            side_effect=[queued_at_start or [], queued_after_first]
            + [[]] * len(queries)
        ),
        "resolve_answered": AsyncMock(return_value=held_results or []),
        "build_session_context": AsyncMock(return_value=""),
        "skills_context_block": AsyncMock(return_value=None),
        "_maybe_prepend_skills_update": AsyncMock(side_effect=lambda *a: a[-1]),
        "build_turn_budget_block": AsyncMock(return_value=""),
        "persist_and_record_usage": AsyncMock(),
        "clear_pending_question": AsyncMock(),
        "config": sdk.config.model_copy(
            update={
                "platform_route": "openrouter",
                "openrouter_api_key": UNSENT_MODEL_KEY,
                "use_e2b_sandbox": False,
                "claude_agent_max_transient_retries": 1,
            }
        ),
    }
    with contextlib.ExitStack() as stack:
        for name, value in stubs.items():
            stack.enter_context(patch.object(sdk, name, value))
        async for event in sdk.stream_chat_completion_sdk(
            session_id=session.session_id,
            message=session.messages[-1].content,
            is_user_message=True,
            user_id=session.user_id,
            session=session,
            envelope=root_envelope(turn_id, session_id=session.session_id),
        ):
            yield event
