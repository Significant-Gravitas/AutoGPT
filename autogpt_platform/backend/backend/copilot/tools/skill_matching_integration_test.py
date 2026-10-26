import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from backend.copilot.tools import skill_matching_helpers_test as baseline
from backend.copilot.tools import skill_matching_test as skill_fixtures
from backend.util import json

baseline_io = baseline_fixtures.baseline_io
available_skills = skill_fixtures.available_skills


@pytest.mark.asyncio
@pytest.mark.parametrize("resume,retry", [(True, False), (False, False), (False, True)])
async def test_sdk_matching_survives_resume_disabled_and_context_retry(
    available_skills: AsyncMock, resume: bool, retry: bool
) -> None:
    session = _sdk_session()
    task = "Find a partner contact"
    queries: list[str] = []
    patches = _make_sdk_patches(
        session,
        original_transcript=_build_transcript(
            [("user", "Earlier"), ("assistant", "Ready")]
        ),
        compacted_transcript=None,
        client_side_effect=_sdk_client_factory(queries, retry),
    )
    with ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        _patch_sdk_context(stack, resume)
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
@pytest.mark.parametrize("first_turn", [False, True])
@pytest.mark.parametrize("request_arrival_at", [0, 100])
@pytest.mark.parametrize("provided_message", [True, False])
@pytest.mark.parametrize(
    "pending_kind,pending_image",
    [("none", False), ("task", False), ("task", True), ("note", False), ("note", True)],
)
async def test_task_and_pending_messages_reach_matching_and_transcript(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    available_skills: AsyncMock,
    first_turn: bool,
    request_arrival_at: float,
    provided_message: bool,
    pending_kind: str,
    pending_image: bool,
) -> None:
    session = baseline.new_session(first_turn)
    provider, upload, matcher = baseline.patch_provider(monkeypatch)
    prefix = baseline.patch_first_turn_context(monkeypatch) if first_turn else ""
    starting_message = baseline.NOTE if pending_kind == "task" else baseline.TASK
    pending = baseline.pending_message(pending_kind, pending_image)
    baseline.patch_pending(monkeypatch, pending)
    expected_request = baseline.expected_request(pending_kind, request_arrival_at)
    if not provided_message:
        session.messages.append(ChatMessage(role="user", content=starting_message))
    await baseline.run_turn(
        session, starting_message if provided_message else None, request_arrival_at
    )
    user_texts = baseline.assert_matched_request(
        provider, matcher, expected_request, pending_image
    )
    baseline.assert_saved_rows(session, pending)
    await baseline.assert_transcript(upload, session, prefix + expected_request)
    if first_turn:
        assert session.messages[0].content == prefix + starting_message
        assert user_texts[0].startswith(prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [None, ""])
@pytest.mark.parametrize(
    "pending_text",
    [None, "Use the approved image", "Find a partner contact for Contoso"],
)
async def test_empty_turn_does_not_reuse_an_answered_task(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    available_skills: AsyncMock,
    message: str | None,
    pending_text: str | None,
) -> None:
    old_task = "Find a partner contact for Northwind"
    session = baseline.new_session()
    provider, upload, matcher = baseline.patch_provider(monkeypatch, reply="Ready")
    baseline.patch_answered_transcript(monkeypatch, session, old_task)
    baseline.patch_pending(
        monkeypatch,
        PendingMessage(content=pending_text) if pending_text is not None else None,
    )
    await baseline.run_turn(session, message)
    baseline.assert_empty_turn(provider, upload, matcher, old_task, pending_text)


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [None, ""])
@pytest.mark.parametrize("pending_text", [None, baseline.NOTE, baseline.TASK])
async def test_empty_first_turn_does_not_record_context_as_a_request(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    available_skills: AsyncMock,
    message: str | None,
    pending_text: str | None,
) -> None:
    session = baseline.new_session(first_turn=True)
    session.messages.append(ChatMessage(role="user", content="", sequence=0))
    _, upload, _ = baseline.patch_provider(monkeypatch)
    prefix = baseline.patch_first_turn_context(monkeypatch)
    baseline.patch_pending(
        monkeypatch,
        PendingMessage(content=pending_text) if pending_text is not None else None,
    )
    await baseline.run_turn(session, message)
    upload.assert_awaited_once()
    assert upload.await_args is not None
    transcript = upload.await_args.kwargs["content"].decode("utf-8")
    rows = [json.loads(line) for line in transcript.splitlines()]
    assert [row["message"]["content"] for row in rows if row["type"] == "user"] == (
        [prefix + pending_text] if pending_text is not None else []
    )
    assert "<matched_skills>" not in transcript


def _sdk_session() -> ChatSession:
    session = _SdkHarness()._make_session()
    session.expert_id = "expert-a"
    session.title = "Partner research"
    session.messages = [
        ChatMessage(role="user", content="Earlier"),
        ChatMessage(role="assistant", content="Ready"),
        ChatMessage(role="user", content="Find a partner contact"),
    ]
    return session


def _sdk_client_factory(queries: list[str], retry: bool):
    harness = _SdkHarness()

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

    return client_factory


def _patch_sdk_context(stack: ExitStack, resume: bool) -> None:
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
