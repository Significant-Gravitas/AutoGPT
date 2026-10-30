import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk

from backend.copilot import pending_message_helpers
from backend.copilot import service as shared_service
from backend.copilot.baseline import service
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.retry_scenarios_test import _build_transcript
from backend.copilot.tool_image import ToolImage
from backend.util import json

TASK = "Add Northwind to my partner list in Airtable and go find me a partner contact"
NOTE = "Use this approved image"
IMAGE = ToolImage(mime_type="image/png", data_base64="aW1hZ2U=")


def new_session(first_turn: bool = False) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False, expert_id="expert-a")
    session.title = "Partner research"
    if not first_turn:
        session.messages = [
            ChatMessage(role="user", content="Help me with partnerships"),
            ChatMessage(role="assistant", content="Ready."),
        ]
    return session


def patch_provider(
    monkeypatch: pytest.MonkeyPatch,
    reply: str = "I will follow the matching procedure.",
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    stream = MagicMock()
    stream.__aiter__.return_value = [_final_chunk(reply)]
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    upload = AsyncMock()
    matcher = AsyncMock(wraps=service.build_matched_skills_notice)
    for name, mock in {
        "call_provider_stream": provider,
        "download_transcript": AsyncMock(return_value=None),
        "upload_transcript": upload,
        "build_expert_identity_suffix": AsyncMock(return_value=""),
        "build_skills_update_notice": AsyncMock(return_value=""),
        "clear_pending_question": AsyncMock(),
        "start_refresh": MagicMock(return_value=None),
        "build_matched_skills_notice": matcher,
    }.items():
        monkeypatch.setattr(service, name, mock)
    return provider, upload, matcher


def patch_first_turn_context(monkeypatch: pytest.MonkeyPatch) -> str:
    for name, value in {
        "_build_system_prompt": ("System", MagicMock()),
        "build_budget_ctx": "Budget",
        "build_session_context": "Session",
        "build_skills_context": "Partner skills",
    }.items():
        monkeypatch.setattr(service, name, AsyncMock(return_value=value))
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
    return (
        "<available_skills>\nPartner skills\n</available_skills>\n\n"
        "<session_context>\nSession\n</session_context>\n\n"
        "<budget_context>\nBudget\n</budget_context>\n\n"
        "<user_context>\nBusiness\nPlan: free\n</user_context>\n\n"
    )


def pending_message(kind: str, image: bool) -> PendingMessage | None:
    if kind == "none":
        return None
    return PendingMessage(
        content=TASK if kind == "task" else NOTE,
        model_image=IMAGE if image else None,
        enqueued_at=50 if kind == "task" else 150,
    )


def patch_pending(
    monkeypatch: pytest.MonkeyPatch, pending: PendingMessage | None
) -> None:
    if pending is None:
        return
    monkeypatch.setattr(
        service, "drain_pending_safe", AsyncMock(return_value=[pending])
    )
    monkeypatch.setattr(
        pending_message_helpers,
        "upsert_chat_session",
        AsyncMock(side_effect=_save_pending),
    )


def expected_request(kind: str, request_arrival_at: float) -> str:
    if kind == "none":
        return TASK
    if request_arrival_at == 0 and kind == "note":
        return f"{NOTE}\n\n{TASK}"
    return f"{TASK}\n\n{NOTE}"


async def run_turn(
    session: ChatSession, message: str | None, request_arrival_at: float = 0
) -> None:
    async with asyncio.timeout(10):
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            message=message,
            user_id="user-1",
            session=session,
            request_arrival_at=request_arrival_at,
        ):
            pass


def assert_matched_request(
    provider: AsyncMock, matcher: AsyncMock, expected: str, pending_image: bool
) -> list[str]:
    user_texts = _provider_user_texts(provider)
    assert provider.await_args is not None
    if pending_image:
        assert IMAGE.openai() in provider.await_args.kwargs["messages"][-1]["content"]
    assert "<matched_skills>" in user_texts[-1]
    assert "partner-first-touch-outreach" in user_texts[-1]
    text = "\n".join(user_texts)
    assert text.index(TASK) < text.index("<matched_skills>")
    _assert_matching_input(matcher, expected)
    return user_texts


def assert_saved_rows(session: ChatSession, pending: PendingMessage | None) -> None:
    if pending is not None:
        saved = [m for m in session.messages if m.content == pending.content]
        assert len(saved) == 1
        assert saved[0].sequence is not None
    assert all("<matched_skills>" not in (m.content or "") for m in session.messages)


async def assert_transcript(
    upload: AsyncMock, session: ChatSession, expected: str
) -> None:
    transcript = _uploaded_text(upload)
    rows = [json.loads(line) for line in transcript.splitlines()]
    assert rows[-2]["type"] == "user"
    assert rows[-2]["message"]["content"] == expected
    restored = await service.extract_context_messages(
        service.TranscriptDownload(
            content=transcript.encode("utf-8"),
            message_count=len(session.messages),
            mode="baseline",
        ),
        [*session.messages, ChatMessage(role="user", content="Continue")],
    )
    assert restored[-2].content == expected


def patch_answered_transcript(
    monkeypatch: pytest.MonkeyPatch, session: ChatSession, old_task: str
) -> None:
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


def assert_empty_turn(
    provider: AsyncMock,
    upload: AsyncMock,
    matcher: AsyncMock,
    old_task: str,
    pending_text: str | None,
) -> None:
    rows = [json.loads(line) for line in _uploaded_text(upload).splitlines()]
    assert [row["message"]["content"] for row in rows if row["type"] == "user"] == [
        old_task,
        *([pending_text] if pending_text is not None else []),
    ]
    _assert_matching_input(matcher, pending_text or "")
    hinted = [
        text for text in _provider_user_texts(provider) if "<matched_skills>" in text
    ]
    assert bool(hinted) == bool(pending_text and "partner" in pending_text)


def _final_chunk(reply: str) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "reply",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "anthropic/claude-sonnet-4-6",
            "choices": [
                {"index": 0, "finish_reason": "stop", "delta": {"content": reply}}
            ],
        }
    )


async def _save_pending(session: ChatSession) -> ChatSession:
    for sequence, row in enumerate(session.messages):
        row.sequence = sequence
    return session


def _provider_user_texts(provider: AsyncMock) -> list[str]:
    provider.assert_awaited_once()
    assert provider.await_args is not None
    return [
        (
            m["content"]
            if isinstance(m["content"], str)
            else "".join(
                part["text"] for part in m["content"] if part["type"] == "text"
            )
        )
        for m in provider.await_args.kwargs["messages"]
        if m["role"] == "user"
    ]


def _assert_matching_input(matcher: AsyncMock, expected: str) -> None:
    matcher.assert_awaited_once()
    assert matcher.await_args is not None
    assert matcher.await_args.args[2] == expected


def _uploaded_text(upload: AsyncMock) -> str:
    upload.assert_awaited_once()
    assert upload.await_args is not None
    transcript = upload.await_args.kwargs["content"].decode("utf-8")
    assert "<matched_skills>" not in transcript
    return transcript
