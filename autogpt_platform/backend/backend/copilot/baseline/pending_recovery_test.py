import asyncio
from collections import deque
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk

from backend.copilot.baseline import service
from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.pending_messages import MAX_PENDING_MESSAGES, PendingMessage
from backend.copilot.response_model import StreamFinish


@pytest.mark.parametrize("count", [MAX_PENDING_MESSAGES + 3, 2 * MAX_PENDING_MESSAGES])
async def test_final_baseline_turn_consumes_all_recovered_messages(
    monkeypatch: pytest.MonkeyPatch, baseline_io: None, count: int
) -> None:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Existing conversation"
    session.messages = [
        ChatMessage(role="user", content="Earlier task"),
        ChatMessage(role="assistant", content="Earlier reply"),
        ChatMessage(role="user", content="Continue"),
    ]
    pending = [PendingMessage(content=f"Queued payload {i:02}") for i in range(count)]
    queue = deque(pending)

    async def drain(*_: object) -> list[PendingMessage]:
        return [queue.popleft() for _ in range(min(len(queue), MAX_PENDING_MESSAGES))]

    monkeypatch.setattr(service, "drain_pending_safe", AsyncMock(side_effect=drain))
    mid_turn_drain = AsyncMock(return_value=[])
    monkeypatch.setattr(service, "drain_pending_messages", mid_turn_drain)
    persist = AsyncMock(return_value=True)
    monkeypatch.setattr(service, "persist_pending_as_user_rows", persist)
    provider = AsyncMock(return_value=_final_stream())
    monkeypatch.setattr(service, "call_provider_stream", provider)

    async with asyncio.timeout(10):
        events = [
            event
            async for event in service.stream_chat_completion_baseline(
                session.session_id,
                user_id="user-1",
                session=session,
                is_user_message=False,
            )
        ]

    assert isinstance(events[-1], StreamFinish)
    assert not queue
    mid_turn_drain.assert_not_awaited()
    persist.assert_awaited_once()
    assert persist.await_args is not None
    assert persist.await_args.args[2] == pending
    provider.assert_awaited_once()
    assert provider.await_args is not None
    messages = provider.await_args.kwargs["messages"]
    for message in pending:
        assert sum(message.content in str(row["content"]) for row in messages) == 1


@pytest.fixture
def baseline_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(
        service,
        "config",
        service.config.model_copy(
            update={"use_e2b_sandbox": False, "use_local": False}
        ),
    )
    for name, value in {
        "resolve_answered": [],
        "resolve_model_route": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "_build_system_prompt": ("System prompt", None),
        "is_enabled_for_user": False,
        "is_feature_enabled": False,
        "persist_and_record_usage": None,
        "download_transcript": None,
        "_upload_final_transcript": None,
    }.items():
        monkeypatch.setattr(service, name, AsyncMock(return_value=value))
    monkeypatch.setattr(service, "_get_main_client", MagicMock())
    monkeypatch.setattr(
        service, "upsert_chat_session", AsyncMock(side_effect=lambda session: session)
    )
    try:
        yield
    finally:
        set_execution_context(None, None)


def _final_stream() -> MagicMock:
    stream = MagicMock()
    stream.__aiter__.return_value = [
        ChatCompletionChunk.model_validate(
            {
                "id": "final",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "anthropic/claude-sonnet-4-6",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "delta": {"content": "Done"}}
                ],
            }
        )
    ]
    stream.close = AsyncMock()
    return stream
