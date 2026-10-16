"""The system message the baseline engine sends carries the shared workflow notes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.tools import skills
from backend.copilot.tools.skills import ParsedSkill

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


@pytest.mark.asyncio
async def test_baseline_names_tools_by_the_short_names_it_declares(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    """The baseline engine declares the short names, so its skills index
    names the bare ``run_capability`` and its system message carries none of
    the SDK path's ``mcp__copilot__`` names."""
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Changelog"
    session.messages.append(ChatMessage(role="user", content="Draft the changelog"))
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    inject = AsyncMock(return_value=None)
    skill = ParsedSkill(name="changelog", description="Drafts the changelog", body="x")
    monkeypatch.setattr(service, "call_provider_stream", provider)
    monkeypatch.setattr(service, "inject_user_context", inject)
    monkeypatch.setattr(service, "build_budget_ctx", AsyncMock(return_value=""))
    monkeypatch.setattr(service, "build_session_context", AsyncMock(return_value=""))
    # Both read Redis; neither is what this test is about.
    monkeypatch.setattr(
        service, "build_skills_update_notice", AsyncMock(return_value="")
    )
    monkeypatch.setattr(service, "resolve_answered", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        skills, "is_skills_feature_enabled", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(skills, "list_all_skills", AsyncMock(return_value=[skill]))

    async with asyncio.timeout(5):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            message="Draft the changelog",
            user_id="user-1",
            session=session,
            is_user_message=True,
        ):
            pass

    skills_ctx = inject.await_args.kwargs["skills_ctx"]
    assert '`run_capability(id="tool:read_skill", input={"name": ...})`' in skills_ctx
    assert "mcp__copilot__" not in skills_ctx
    assert provider.await_args is not None
    [system] = [
        m for m in provider.await_args.kwargs["messages"] if m["role"] == "system"
    ]
    assert "mcp__copilot__" not in str(system["content"])
