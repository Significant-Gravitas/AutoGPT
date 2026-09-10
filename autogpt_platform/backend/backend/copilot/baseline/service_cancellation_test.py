"""Saved display names survive cancellation and stay out of provider requests."""

import asyncio
from collections.abc import Iterator
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk

from backend.copilot.baseline import service
from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.response_model import (
    StreamToolDisplayAvailable,
    StreamToolOutputAvailable,
)
from backend.copilot.tool_display import emit_tool_display_name


@pytest.mark.asyncio
async def test_cancelled_stream_persists_named_calls_and_completed_sibling(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Run two workflows"
    session.messages.append(ChatMessage(role="user", content="Run both workflows"))
    provider_stream = _tool_call_stream()
    monkeypatch.setattr(
        service, "call_provider_stream", AsyncMock(return_value=provider_stream)
    )
    interrupted = asyncio.Event()

    async def execute(
        *, tool_call_id: str, parameters: dict[str, str], **kwargs: object
    ) -> StreamToolOutputAvailable:
        assert parameters == {"library_agent_id": tool_call_id}
        emit_tool_display_name(f"Workflow {tool_call_id}")
        if tool_call_id == "pending":
            try:
                await asyncio.Future[None]()
            finally:
                interrupted.set()
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id, output="actual completed result"
        )

    monkeypatch.setattr(service, "execute_tool", AsyncMock(side_effect=execute))
    completed = asyncio.Event()
    names: dict[str, str] = {}

    async def consume() -> None:
        async for event in service.stream_chat_completion_baseline(
            session.session_id, user_id="user-1", session=session, is_user_message=False
        ):
            if isinstance(event, StreamToolDisplayAvailable):
                names[event.data.toolCallId] = event.data.displayName
            if isinstance(event, StreamToolOutputAvailable):
                assert event.toolCallId == "completed"
                completed.set()

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(completed.wait(), timeout=5)
        assert names == {
            "pending": "Workflow pending",
            "completed": "Workflow completed",
        }
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=5)
    finally:
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=5)

    assert interrupted.is_set()
    provider_stream.close.assert_awaited_once()
    [persisted] = baseline_io
    assert [message.role for message in persisted.messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assistant, result = persisted.messages[1:]
    assert assistant.content == "Running both workflows."
    assert assistant.tool_calls is not None
    assert [call["id"] for call in assistant.tool_calls] == ["pending", "completed"]
    assert {call["id"]: call["display_name"] for call in assistant.tool_calls} == names
    assert result.tool_call_id == "completed"
    assert result.content == "actual completed result"


@pytest.mark.asyncio
async def test_resumed_stream_sends_clean_tool_calls_and_retains_saved_names(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    call = {
        "id": "saved-call",
        "type": "function",
        "function": {"name": "run_agent", "arguments": '{"library_agent_id":"id-1"}'},
    }
    assistant = ChatMessage(
        role="assistant", tool_calls=[{**call, "display_name": "Daily report"}]
    )
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Existing conversation"
    session.messages = [
        ChatMessage(role="user", content="Run the workflow"),
        assistant,
        ChatMessage(role="tool", tool_call_id="saved-call", content="Completed"),
        ChatMessage(role="user", content="Continue"),
    ]
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)
    monkeypatch.setattr(service, "download_transcript", AsyncMock(return_value=None))

    async with asyncio.timeout(5):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id, user_id="user-1", session=session, is_user_message=False
        ):
            pass

    provider.assert_awaited_once()
    assert provider.await_args is not None
    [sent_assistant] = [
        message
        for message in provider.await_args.kwargs["messages"]
        if message["role"] == "assistant"
    ]
    assert sent_assistant["tool_calls"] == [call]
    assert assistant.tool_calls == [{**call, "display_name": "Daily report"}]
    [persisted] = baseline_io
    assert persisted.messages[1].tool_calls == assistant.tool_calls


@pytest.fixture
def baseline_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ChatSession]]:
    persisted: list[ChatSession] = []

    async def save(session: ChatSession) -> ChatSession:
        persisted.append(session.model_copy(deep=True))
        return session

    monkeypatch.setattr(
        service,
        "config",
        service.config.model_copy(
            update={"use_e2b_sandbox": False, "use_local": False}
        ),
    )
    for name, value in {
        "drain_pending_safe": [],
        "resolve_model_route": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "_build_system_prompt": ("System prompt", None),
        "is_enabled_for_user": False,
        "is_feature_enabled": False,
        "persist_and_record_usage": None,
    }.items():
        monkeypatch.setattr(service, name, AsyncMock(return_value=value))
    monkeypatch.setattr(service, "_get_main_client", MagicMock())
    monkeypatch.setattr(service, "upsert_chat_session", AsyncMock(side_effect=save))
    try:
        yield persisted
    finally:
        set_execution_context(None, None)


def _tool_call_stream() -> MagicMock:
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "round-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "anthropic/claude-sonnet-4-6",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "delta": {
                        "content": "Running both workflows.",
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "run_agent",
                                    "arguments": f'{{"library_agent_id":"{call_id}"}}',
                                },
                            }
                            for index, call_id in enumerate(("pending", "completed"))
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }
    )
    stream = MagicMock()
    stream.__aiter__.return_value = [chunk]
    stream.close = AsyncMock()
    return stream
