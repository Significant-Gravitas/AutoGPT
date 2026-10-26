"""A continuation note that ends the chat reaches the model as the user's turn.

The note is stored as an assistant row, so a request built from history would
end on an assistant message: a prefill, which Claude 4.6+ rejects with a 400.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline.service_cancellation_test import (
    _no_answered_cards,
    baseline_io,
)
from backend.copilot.continuation import KEEP_GOING_NOTE
from backend.copilot.model import ChatMessage, ChatSession

__all__ = ["baseline_io"]

CHECKPOINT = "Checkpoint: two of three sections done."


@pytest.mark.asyncio
async def test_a_trailing_note_is_sent_as_the_user_turn_without_its_marker(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    session = _chat(
        ChatMessage(role="assistant", content=CHECKPOINT),
        ChatMessage(role="assistant", content=KEEP_GOING_NOTE),
    )

    sent = await _sent_messages(monkeypatch, session, message=KEEP_GOING_NOTE)

    assert sent[-2:] == [
        {"role": "assistant", "content": CHECKPOINT},
        {
            "role": "user",
            "content": "Keep going: pick the task up from the checkpoint above.",
        },
    ]
    [persisted] = baseline_io
    [note] = [m for m in persisted.messages if m.content == KEEP_GOING_NOTE]
    assert note.role == "assistant"


@pytest.mark.asyncio
async def test_other_trailing_assistant_text_is_sent_as_it_is(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    session = _chat(ChatMessage(role="assistant", content=CHECKPOINT))

    sent = await _sent_messages(monkeypatch, session, message=None)

    assert sent[-1] == {"role": "assistant", "content": CHECKPOINT}


def _chat(*rows: ChatMessage) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Quarterly report"
    session.messages = [ChatMessage(role="user", content="Build the report"), *rows]
    return session


async def _sent_messages(
    monkeypatch: pytest.MonkeyPatch, session: ChatSession, *, message: str | None
) -> list[dict[str, Any]]:
    """Run a non-user turn and return the messages the provider was sent."""
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)
    monkeypatch.setattr(service, "resolve_answered", _no_answered_cards)

    async def consume() -> None:
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            message,
            is_user_message=False,
            user_id="user-1",
            session=session,
        ):
            pass

    await asyncio.wait_for(consume(), timeout=5)

    assert provider.await_args is not None
    return provider.await_args.kwargs["messages"]
