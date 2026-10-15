"""An auto-continue turn's queued messages reach the model once each.

When a turn ends with messages queued, the turn continues with them: each
becomes its own user row and, joined, the continued turn's message. The
history the query is built from must end before those rows, on a resumed
session (the gap) and on one that cannot resume (the history block), or every
queued message but the last is sent twice.

Drives the real ``stream_chat_completion_sdk`` up to the CLI query, with the
CLI client, the DB and Redis stubbed.
"""

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import _RestoreResult, stream_chat_completion_sdk

_SVC = "backend.copilot.sdk.service"
_QUEUED = ["QUEUED ONE", "QUEUED TWO", "QUEUED THREE"]


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [True, False], ids=["resumed", "not-resumed"])
@pytest.mark.parametrize("queued", [_QUEUED[:1], _QUEUED[:2], _QUEUED])
async def test_an_auto_continue_turn_sends_each_queued_message_once(
    tmp_path: Path, queued: list[str], resumed: bool
):
    client = _client()

    await _continue_turn(tmp_path, client, queued, resumed=resumed)

    [query] = [call.args[0] for call in client.query.await_args_list]
    assert [query.count(text) for text in queued] == [1] * len(queued)
    assert ("User: prior" in query) is not resumed


async def _continue_turn(
    tmp_path: Path, client: MagicMock, queued: list[str], *, resumed: bool
) -> None:
    session = ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        messages=[
            ChatMessage(role="user", content="prior", sequence=0),
            ChatMessage(role="assistant", content="prior answer", sequence=1),
        ],
    )
    transcript = build_test_transcript([("user", "prior"), ("assistant", "ans")])
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    patches = _make_sdk_patches(session, transcript, None, lambda *a, **k: cm) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
        (f"{_SVC}.persist_pending_as_user_rows", dict(side_effect=_persist)),
        (f"{_SVC}._rebuild_first_turn_context", dict(new=AsyncMock(return_value=[]))),
    ]
    if resumed:
        # The previous turn's session file covers its rows: sequences 0 and 1.
        restored = _RestoreResult(
            use_resume=True, resume_file="test-session-id", transcript_msg_count=2
        )
        patches.append(
            (
                f"{_SVC}._restore_cli_session_for_turn",
                dict(new=AsyncMock(return_value=restored)),
            )
        )
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for _ in stream_chat_completion_sdk(
            session_id="test-session-id",
            message="\n\n".join(queued),
            is_user_message=True,
            user_id="test-user",
            session=session,
            continued_pending=[PendingMessage(content=text) for text in queued],
        ):
            pass


async def _persist(session, _builder, pending, *, log_prefix):
    """Append the queued messages as user rows with the next sequences."""
    next_seq = 1 + max(m.sequence for m in session.messages if m.sequence is not None)
    for offset, message in enumerate(pending):
        session.messages.append(
            ChatMessage(
                role="user", content=message.content, sequence=next_seq + offset
            )
        )
    return True


def _client() -> MagicMock:
    async def _receive():
        yield AssistantMessage(
            content=[TextBlock(text="done")], model="claude-sonnet-4-6"
        )
        yield ResultMessage(
            subtype="success",
            result="done",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="test-session-id",
        )

    client = MagicMock(receive_response=_receive, query=AsyncMock())
    client._transport.write = AsyncMock()
    return client
