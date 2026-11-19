from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionChunk

from backend.copilot import pending_message_helpers, pending_message_reconciliation
from backend.copilot.baseline import service
from backend.copilot.baseline import service_cancellation_test as baseline_fixtures
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tools import skill_matching_helpers_test as baseline
from backend.copilot.tools import skill_matching_test as skill_fixtures
from backend.util import json

baseline_io = baseline_fixtures.baseline_io
available_skills = skill_fixtures.available_skills


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_image", [False, True])
@pytest.mark.parametrize("extra_note", [False, True])
@pytest.mark.parametrize("persist_ok", [False, True])
async def test_pending_task_is_matched_before_next_tool_round(
    monkeypatch: pytest.MonkeyPatch,
    baseline_io: list[ChatSession],
    available_skills: AsyncMock,
    pending_image: bool,
    extra_note: bool,
    persist_ok: bool,
) -> None:
    session = baseline.new_session()
    provider, upload, matcher = baseline.patch_provider(monkeypatch)
    sent = _patch_two_rounds(monkeypatch, provider)
    pending = _pending_batch(pending_image, extra_note)
    restore = _patch_pending_persistence(monkeypatch, pending, persist_ok)
    await baseline.run_turn(session, "Continue the current task")
    assert len(sent) == 2
    assert "<matched_skills>" not in _user_content(sent[0])
    _assert_next_round(sent[1], pending_image, persist_ok)
    assert [call.args[2] for call in matcher.await_args_list] == [
        "Continue the current task",
        *(["\n\n".join(pm.content for pm in pending)] if persist_ok else []),
    ]
    _assert_saved_messages(session, upload, pending, persist_ok)
    if persist_ok:
        restore.assert_not_awaited()
    else:
        restore.assert_awaited_once_with(session.session_id, pending)


def _patch_two_rounds(monkeypatch: pytest.MonkeyPatch, provider: AsyncMock) -> list:
    sent = []
    final = provider.return_value

    async def call(**kwargs):
        sent.append(json.loads(json.dumps(kwargs["messages"])))
        return _tool_stream() if len(sent) == 1 else final

    async def execute(*, tool_call_id: str, **kwargs):
        return StreamToolOutputAvailable(toolCallId=tool_call_id, output="Ready")

    provider.side_effect = call
    monkeypatch.setattr(service, "execute_tool", AsyncMock(side_effect=execute))
    return sent


def _pending_batch(image: bool, extra_note: bool) -> list[PendingMessage]:
    pending = [
        PendingMessage(
            content=baseline.TASK, model_image=baseline.IMAGE if image else None
        )
    ]
    if extra_note:
        pending.append(PendingMessage(content=baseline.NOTE))
    return pending


def _patch_pending_persistence(
    monkeypatch: pytest.MonkeyPatch, pending: list[PendingMessage], persist_ok: bool
) -> AsyncMock:
    async def save(session: ChatSession) -> ChatSession:
        if persist_ok:
            for sequence, message in enumerate(session.messages):
                message.sequence = sequence
        return session

    monkeypatch.setattr(
        service, "drain_pending_messages", AsyncMock(return_value=pending)
    )
    monkeypatch.setattr(
        pending_message_helpers, "upsert_chat_session", AsyncMock(side_effect=save)
    )
    store = AsyncMock()
    store.get_chat_messages_by_ids.return_value = []
    monkeypatch.setattr(pending_message_reconciliation, "chat_db", lambda: store)
    restore = AsyncMock()
    monkeypatch.setattr(pending_message_helpers, "restore_pending_messages", restore)
    return restore


def _assert_next_round(messages: list, image: bool, persist_ok: bool) -> None:
    content = _user_content(messages)
    assert ("<matched_skills>" in content) == persist_ok
    assert ("skill:partner-first-touch-outreach" in content) == persist_ok
    assert (baseline.TASK in content) == persist_ok
    assert (baseline.IMAGE.data_base64 in content) == (image and persist_ok)
    if persist_ok:
        assert messages[-1]["role"] == "user"
        assert "<matched_skills>" in json.dumps(messages[-1]["content"])


def _assert_saved_messages(
    session: ChatSession,
    upload: AsyncMock,
    pending: list[PendingMessage],
    persist_ok: bool,
) -> None:
    upload.assert_awaited_once()
    assert upload.await_args is not None
    transcript = upload.await_args.kwargs["content"].decode("utf-8")
    assert "<matched_skills>" not in transcript
    assert all("<matched_skills>" not in (m.content or "") for m in session.messages)
    for pm in pending:
        assert sum(m.content == pm.content for m in session.messages) == int(persist_ok)
        assert (pm.content in transcript) == persist_ok


def _user_content(messages: list) -> str:
    return json.dumps([m["content"] for m in messages if m["role"] == "user"])


def _tool_stream() -> MagicMock:
    stream = MagicMock()
    stream.close = AsyncMock()
    stream.__aiter__.return_value = [
        ChatCompletionChunk.model_validate(
            {
                "id": "tool-round",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "anthropic/claude-sonnet-4-6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "read-file",
                                    "type": "function",
                                    "function": {
                                        "name": "read_workspace_file",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                    }
                ],
            }
        )
    ]
    return stream
