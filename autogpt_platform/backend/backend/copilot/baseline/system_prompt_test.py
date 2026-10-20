"""The system message the baseline engine sends carries the shared workflow notes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.model import ChatMessage, ChatSession

__all__ = ["baseline_io"]


@pytest.mark.asyncio
async def test_system_message_tells_the_model_formulas_render(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Math"
    session.messages.append(ChatMessage(role="user", content="Derive it"))
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)

    async with asyncio.timeout(5):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id, user_id="user-1", session=session, is_user_message=False
        ):
            pass

    assert provider.await_args is not None
    [system] = [
        m for m in provider.await_args.kwargs["messages"] if m["role"] == "system"
    ]
    assert "`$…$` inline, `$$…$$` for display" in str(system["content"])
