import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.baseline import service
from backend.copilot.baseline import service_cancellation_test as baseline_fixtures
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError
from backend.copilot.sdk import service as sdk_service
from backend.copilot.sdk.retry_scenarios_test import (
    TestStreamChatCompletionRetryIntegration as _SdkHarness,
)
from backend.copilot.sdk.retry_scenarios_test import (
    _build_transcript,
    _make_sdk_patches,
)
from backend.copilot.tool_image import ToolImage
from backend.copilot.tools import skill_matching_test as skill_fixtures

baseline_io = baseline_fixtures.baseline_io
available_skills = skill_fixtures.available_skills


@pytest.mark.asyncio
@pytest.mark.parametrize("resume,retry", [(True, False), (False, False), (False, True)])
async def test_sdk_matching_survives_resume_disabled_and_context_retry(
    available_skills: AsyncMock, resume: bool, retry: bool
) -> None:
    harness = _SdkHarness()
    session = harness._make_session()
    session.expert_id = "expert-a"
    session.title = "Partner research"
    task = "Find a partner contact"
    session.messages = [
        ChatMessage(role="user", content="Earlier"),
        ChatMessage(role="assistant", content="Ready"),
        ChatMessage(role="user", content=task),
    ]
    queries: list[str] = []

    def client_factory(*args, **kwargs):
        client = harness._make_client_mock(
            result_message=harness._make_result_message()
        )

        async def query(prompt, **kwargs):
            queries.append(prompt)
            if retry and len(queries) == 1:
                raise RuntimeError("prompt is too long (context_length_exceeded)")

        client.__aenter__.return_value.query = query
        return client

    transcript = _build_transcript([("user", "Earlier"), ("assistant", "Ready")])
    patches = _make_sdk_patches(
        session,
        original_transcript=transcript,
        compacted_transcript=None,
        client_side_effect=client_factory,
    )
    with ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        stack.enter_context(
            patch.object(sdk_service.config, "claude_agent_use_resume", resume)
        )
        held_statuses = MagicMock()
        held_statuses.__aiter__.return_value = []
        stack.enter_context(
            patch.object(sdk_service, "resolve_answered", return_value=held_statuses)
        )
        for name, result in {
            "build_expert_identity_suffix": "",
            "build_skills_update_notice": "",
            "clear_pending_question": None,
            "persist_and_record_usage": None,
            "active_mode": None,
        }.items():
            stack.enter_context(
                patch.object(sdk_service, name, AsyncMock(return_value=result))
            )
        async with asyncio.timeout(10):
            events = [
                event
                async for event in sdk_service.stream_chat_completion_sdk(
                    session_id=session.session_id,
                    message=task,
                    user_id="test-user",
                    session=session,
                )
            ]
    assert not [event for event in events if isinstance(event, StreamError)]
    assert len(queries) == (2 if retry else 1)
    for query in queries:
        assert "<matched_skills>" in query
        assert "skill:partner-first-touch-outreach" in query
        assert query.rindex(task) < query.index("<matched_skills>")
    assert all("<matched_skills>" not in (m.content or "") for m in session.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("provided_message", [True, False])
@pytest.mark.parametrize("pending_image", [True, False])
async def test_followup_surfaces_matching_skills_without_the_first_turn_index(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    available_skills: AsyncMock,
    provided_message: bool,
    pending_image: bool,
) -> None:
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.title = "Partner research"
    session.messages = [
        ChatMessage(role="user", content="Help me with partnerships"),
        ChatMessage(role="assistant", content="Ready."),
    ]
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)
    monkeypatch.setattr(service, "download_transcript", AsyncMock(return_value=None))
    monkeypatch.setattr(
        service, "build_expert_identity_suffix", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        service, "build_skills_update_notice", AsyncMock(return_value="")
    )
    monkeypatch.setattr(service, "clear_pending_question", AsyncMock())

    task = (
        "Add Northwind to my partner list in Airtable and go find me a partner contact"
    )
    starting_message = "Use this approved image" if pending_image else task
    image = ToolImage(mime_type="image/png", data_base64="aW1hZ2U=")
    if pending_image:
        pending = PendingMessage(content=task, model_image=image)
        monkeypatch.setattr(
            service, "drain_pending_safe", AsyncMock(return_value=[pending])
        )
        monkeypatch.setattr(
            service, "persist_pending_as_user_rows", AsyncMock(return_value=True)
        )
    if not provided_message:
        session.messages.append(ChatMessage(role="user", content=starting_message))
    async with asyncio.timeout(10):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            message=starting_message if provided_message else None,
            user_id="user-1",
            session=session,
        ):
            pass

    provider.assert_awaited_once()
    assert provider.await_args is not None
    current = provider.await_args.kwargs["messages"][-1]["content"]
    if pending_image:
        assert image.openai() in current
        current = "".join(part["text"] for part in current if part["type"] == "text")
    assert "<matched_skills>" in current
    assert "partner-first-touch-outreach" in current
    assert current.index(task) < current.index("<matched_skills>")
    assert all("<matched_skills>" not in (m.content or "") for m in session.messages)
