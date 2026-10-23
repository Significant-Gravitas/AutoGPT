import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat import ChatCompletionChunk

from backend.copilot import pending_message_helpers
from backend.copilot import service as shared_service
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
from backend.util import json

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
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.title = "Partner research"
    if not first_turn:
        session.messages = [
            ChatMessage(role="user", content="Help me with partnerships"),
            ChatMessage(role="assistant", content="Ready."),
        ]
    stream = MagicMock()
    stream.__aiter__.return_value = [
        ChatCompletionChunk.model_validate(
            {
                "id": "reply",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "anthropic/claude-sonnet-4-6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta": {"content": "I will follow the matching procedure."},
                    }
                ],
            }
        )
    ]
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)
    monkeypatch.setattr(service, "download_transcript", AsyncMock(return_value=None))
    upload = AsyncMock()
    monkeypatch.setattr(service, "upload_transcript", upload)
    monkeypatch.setattr(
        service, "build_expert_identity_suffix", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        service, "build_skills_update_notice", AsyncMock(return_value="")
    )
    monkeypatch.setattr(service, "clear_pending_question", AsyncMock())
    monkeypatch.setattr(service, "start_refresh", MagicMock(return_value=None))
    matcher = AsyncMock(wraps=service.build_matched_skills_notice)
    monkeypatch.setattr(service, "build_matched_skills_notice", matcher)
    prefix = ""
    if first_turn:
        monkeypatch.setattr(
            service,
            "_build_system_prompt",
            AsyncMock(return_value=("System", MagicMock())),
        )
        monkeypatch.setattr(
            service, "build_budget_ctx", AsyncMock(return_value="Budget")
        )
        monkeypatch.setattr(
            service, "build_session_context", AsyncMock(return_value="Session")
        )
        monkeypatch.setattr(
            service, "build_skills_context", AsyncMock(return_value="Partner skills")
        )
        monkeypatch.setattr(
            shared_service, "build_expert_context", AsyncMock(return_value="")
        )
        monkeypatch.setattr(
            shared_service, "format_understanding_for_prompt", lambda _: "Business"
        )
        monkeypatch.setattr(
            "backend.copilot.rate_limit.get_user_tier",
            AsyncMock(return_value=MagicMock(value="free")),
        )
        monkeypatch.setattr(
            shared_service,
            "chat_db",
            MagicMock(
                return_value=MagicMock(update_message_content_by_sequence=AsyncMock())
            ),
        )
        prefix = (
            "<available_skills>\nPartner skills\n</available_skills>\n\n"
            "<session_context>\nSession\n</session_context>\n\n"
            "<budget_context>\nBudget\n</budget_context>\n\n"
            "<user_context>\nBusiness\nPlan: free\n</user_context>\n\n"
        )

    task = (
        "Add Northwind to my partner list in Airtable and go find me a partner contact"
    )
    note = "Use this approved image"
    starting_message = note if pending_kind == "task" else task
    expected_request = starting_message
    image = ToolImage(mime_type="image/png", data_base64="aW1hZ2U=")
    pending = None
    if pending_kind != "none":
        pending = PendingMessage(
            content=task if pending_kind == "task" else note,
            model_image=image if pending_image else None,
            enqueued_at=50 if pending_kind == "task" else 150,
        )
        monkeypatch.setattr(
            service, "drain_pending_safe", AsyncMock(return_value=[pending])
        )

        async def save_pending(session: ChatSession) -> ChatSession:
            for sequence, row in enumerate(session.messages):
                row.sequence = sequence
            return session

        monkeypatch.setattr(
            pending_message_helpers,
            "upsert_chat_session",
            AsyncMock(side_effect=save_pending),
        )
        expected_request = f"{task}\n\n{note}"
        if request_arrival_at == 0 and pending_kind == "note":
            expected_request = f"{note}\n\n{task}"
    if not provided_message:
        session.messages.append(ChatMessage(role="user", content=starting_message))
    async with asyncio.timeout(10):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            message=starting_message if provided_message else None,
            user_id="user-1",
            session=session,
            request_arrival_at=request_arrival_at,
        ):
            pass

    provider.assert_awaited_once()
    assert provider.await_args is not None
    sent = provider.await_args.kwargs["messages"]
    current = sent[-1]["content"]
    if pending_image:
        assert image.openai() in current
    user_texts = [
        (
            m["content"]
            if isinstance(m["content"], str)
            else "".join(
                part["text"] for part in m["content"] if part["type"] == "text"
            )
        )
        for m in sent
        if m["role"] == "user"
    ]
    current = user_texts[-1]
    assert "<matched_skills>" in current
    assert "partner-first-touch-outreach" in current
    text = "\n".join(user_texts)
    assert text.index(task) < text.index("<matched_skills>")
    matcher.assert_awaited_once()
    assert matcher.await_args is not None
    assert matcher.await_args.args[2] == expected_request
    if pending is not None:
        saved = [m for m in session.messages if m.content == pending.content]
        assert len(saved) == 1
        assert saved[0].sequence is not None
    assert all("<matched_skills>" not in (m.content or "") for m in session.messages)
    upload.assert_awaited_once()
    assert upload.await_args is not None
    transcript = upload.await_args.kwargs["content"].decode("utf-8")
    rows = [json.loads(line) for line in transcript.splitlines()]
    assert rows[-2]["type"] == "user"
    expected_transcript = prefix + expected_request
    assert rows[-2]["message"]["content"] == expected_transcript
    assert "<matched_skills>" not in transcript
    restored = await service.extract_context_messages(
        service.TranscriptDownload(
            content=transcript.encode("utf-8"),
            message_count=len(session.messages),
            mode="baseline",
        ),
        [*session.messages, ChatMessage(role="user", content="Continue")],
    )
    assert restored[-2].content == expected_transcript
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
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.title = "Partner research"
    session.messages = [
        ChatMessage(role="user", content=old_task, sequence=0),
        ChatMessage(role="assistant", content="Done", sequence=1),
    ]
    previous = _build_transcript([("user", old_task), ("assistant", "Done")])
    monkeypatch.setattr(
        service,
        "download_transcript",
        AsyncMock(
            return_value=service.TranscriptDownload(
                content=previous, message_count=2, mode="baseline"
            )
        ),
    )
    stream = MagicMock()
    stream.__aiter__.return_value = [
        ChatCompletionChunk.model_validate(
            {
                "id": "reply",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "anthropic/claude-sonnet-4-6",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "delta": {"content": "Ready"}}
                ],
            }
        )
    ]
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)
    upload = AsyncMock()
    monkeypatch.setattr(service, "upload_transcript", upload)
    for name in ("build_expert_identity_suffix", "build_skills_update_notice"):
        monkeypatch.setattr(service, name, AsyncMock(return_value=""))
    monkeypatch.setattr(service, "start_refresh", MagicMock(return_value=None))
    matcher = AsyncMock(wraps=service.build_matched_skills_notice)
    monkeypatch.setattr(service, "build_matched_skills_notice", matcher)
    if pending_text is not None:
        monkeypatch.setattr(
            service,
            "drain_pending_safe",
            AsyncMock(return_value=[PendingMessage(content=pending_text)]),
        )

        async def save_pending(session: ChatSession) -> ChatSession:
            for sequence, row in enumerate(session.messages):
                row.sequence = sequence
            return session

        monkeypatch.setattr(
            pending_message_helpers,
            "upsert_chat_session",
            AsyncMock(side_effect=save_pending),
        )
    async with asyncio.timeout(10):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id, message=message, user_id="user-1", session=session
        ):
            pass

    upload.assert_awaited_once()
    assert upload.await_args is not None
    transcript = upload.await_args.kwargs["content"].decode("utf-8")
    rows = [json.loads(line) for line in transcript.splitlines()]
    assert [row["message"]["content"] for row in rows if row["type"] == "user"] == [
        old_task,
        *([pending_text] if pending_text is not None else []),
    ]
    matcher.assert_awaited_once()
    assert matcher.await_args is not None
    assert matcher.await_args.args[2] == (pending_text or "")
    provider.assert_awaited_once()
    assert provider.await_args is not None
    sent = provider.await_args.kwargs["messages"]
    hinted = [
        m
        for m in sent
        if m["role"] == "user" and "<matched_skills>" in str(m.get("content", ""))
    ]
    assert bool(hinted) == bool(pending_text and "partner" in pending_text)
    assert "<matched_skills>" not in transcript
