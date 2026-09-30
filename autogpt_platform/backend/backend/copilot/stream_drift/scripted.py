"""Copilot engines driven by a script instead of a model, for the drift suite."""

import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from claude_agent_sdk import Message
from openai.types.chat import ChatCompletionChunk

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamBaseResponse, StreamStart
from backend.copilot.sdk.response_adapter import SDKResponseAdapter
from backend.copilot.sdk.service import _dispatch_response, _StreamAccumulator


def session_with_prompt(prompt: str) -> ChatSession:
    session = ChatSession.new("drift-user", dry_run=False)
    session.title = "Drift fixture"
    session.messages.append(ChatMessage(role="user", content=prompt))
    return session


async def baseline_turn(
    session: ChatSession,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """The baseline engine; its provider and I/O come from the ``baseline_io`` fixture."""
    async for event in baseline.stream_chat_completion_baseline(
        session.session_id,
        user_id="drift-user",
        session=session,
        is_user_message=False,
    ):
        yield event


def provider_round(
    deltas: list[str], *, tool_call: dict[str, str] | None = None
) -> MagicMock:
    """One scripted OpenAI-compatible response: text deltas, then a tool call."""
    chunks = [_chunk({"content": delta}) for delta in deltas]
    if tool_call:
        chunks.append(
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": f'{{"url": "{tool_call["url"]}"}}',
                            },
                        }
                    ]
                }
            )
        )
    chunks.append(
        _chunk({}, finish_reason="tool_calls" if tool_call else "stop", usage=True)
    )
    stream = MagicMock()
    stream.__aiter__.return_value = chunks
    stream.close = AsyncMock()
    return stream


async def sdk_turn(
    session: ChatSession, messages: list[Message]
) -> AsyncGenerator[StreamBaseResponse, None]:
    """The SDK engine's own adapter and row builder over a scripted CLI."""
    message_id = str(uuid.uuid4())
    adapter = SDKResponseAdapter(message_id=message_id, session_id=session.session_id)
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )
    ctx = MagicMock(session=session, log_prefix="[drift]")
    yield StreamStart(messageId=message_id, sessionId=session.session_id)
    for message in messages:
        for response in adapter.convert_message(message):
            dispatched = _dispatch_response(
                response, acc, ctx, MagicMock(), False, "[drift]"
            )
            if dispatched is not None:
                yield dispatched


def _chunk(
    delta: dict[str, Any], *, finish_reason: str | None = None, usage: bool = False
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "round",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "anthropic/claude-sonnet-4-6",
            "choices": [{"index": 0, "finish_reason": finish_reason, "delta": delta}],
            **(
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    }
                }
                if usage
                else {}
            ),
        }
    )
