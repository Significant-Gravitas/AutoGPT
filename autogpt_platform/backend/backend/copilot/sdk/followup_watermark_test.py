"""The CLI session uploaded after a turn covers the turn's mid-turn follow-up.

A follow-up the user sends while a tool runs reaches the model as the
PostToolUse hook's additional context, which the CLI keeps in its session
file and replays on ``--resume`` (followup_resume_cli_test.py). So the
watermark stored with that file must cover the follow-up's row like any
other: the next uncovered sequence over the whole turn. A lower watermark
re-sends the rows after it to the next turn, without the follow-up.

Drives the real ``stream_chat_completion_sdk`` up to the upload, with the
CLI client, the DB and Redis stubbed.
"""

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk

_SVC = "backend.copilot.sdk.service"


@pytest.mark.asyncio
async def test_upload_watermark_covers_a_midturn_followup_row(tmp_path: Path):
    session = _session()
    upload = AsyncMock()

    await _run_turn_with_a_followup(session, upload, tmp_path)

    assert [m.content for m in session.messages if m.role == "user"][-1] == (
        "FOLLOW-UP"
    )
    last = max(m.sequence for m in session.messages if m.sequence is not None)
    assert upload.await_args is not None
    assert upload.await_args.kwargs["message_count"] == last + 1


def _session() -> ChatSession:
    return ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="prior", sequence=0),
            ChatMessage(role="assistant", content="prior answer", sequence=1),
            ChatMessage(role="user", content="hello", sequence=2),
        ],
    )


async def _run_turn_with_a_followup(
    session: ChatSession, upload: AsyncMock, tmp_path: Path
) -> None:
    transcript = build_test_transcript([("user", "prior"), ("assistant", "ans")])
    patches = [
        (target, kwargs)
        for target, kwargs in _make_sdk_patches(
            session, transcript, None, lambda *a, **k: _client(_tool_turn())
        )
        if not target.endswith((".upload_transcript", ".upsert_chat_session"))
    ] + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.upsert_chat_session", dict(new=AsyncMock(side_effect=_backfill))),
        (f"{_SVC}.upload_transcript", dict(new=upload)),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
        (
            f"{_SVC}.drain_pending_for_persist",
            dict(
                new=AsyncMock(
                    side_effect=[[PendingMessage(content="FOLLOW-UP")]] + [[]] * 5
                )
            ),
        ),
        (f"{_SVC}.persist_pending_as_user_rows", dict(side_effect=_persist)),
        (
            f"{_SVC}.read_cli_session_from_disk",
            dict(return_value=transcript.encode()),
        ),
    ]
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for _ in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="hello",
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass


def _tool_turn() -> list:
    return [
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="WebSearch", input={"q": "x"})],
            model="claude-sonnet-4-6",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="OUT")]),
        AssistantMessage(content=[TextBlock(text="FINAL")], model="claude-sonnet-4-6"),
        ResultMessage(
            subtype="success",
            result="FINAL",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="test-session-id",
        ),
    ]


def _backfill(session: ChatSession) -> ChatSession:
    """Give rows the next DB sequences, as a save does."""
    next_seq = 1 + max(m.sequence for m in session.messages if m.sequence is not None)
    for m in session.messages:
        if m.sequence is None:
            m.sequence, next_seq = next_seq, next_seq + 1
    return session


async def _persist(session, _builder, pending, *, log_prefix):
    session.messages.extend(
        ChatMessage(role="user", content=p.content) for p in pending
    )
    _backfill(session)
    return True


def _client(messages: list) -> AsyncMock:
    async def _receive():
        for message in messages:
            yield message

    client = MagicMock(receive_response=_receive, query=AsyncMock())
    client._transport.write = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm
