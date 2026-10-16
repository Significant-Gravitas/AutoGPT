"""A paused turn's marker row keeps the turn's taint, for the turn that picks
the task back up (Keep going, or the resume at the budget reset).

Drives the real ``stream_chat_completion_sdk`` until the CLI stops the turn at
its step backstop, with the CLI client, the DB and Redis stubbed.
"""

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.conftest import build_test_transcript
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.tree import root_envelope
from backend.copilot.turn_pause import continues_tainted, pause_of

_SVC = "backend.copilot.sdk.service"
_MESSAGE = "Carry on with the second section"


@pytest.mark.asyncio
@pytest.mark.parametrize("tainted", [True, False], ids=["tainted", "untainted"])
async def test_a_paused_turns_marker_keeps_its_taint(tmp_path: Path, tainted: bool):
    session = ChatSession(
        session_id="test-session-id",
        user_id="test-user",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        # A later turn (the first writes its context blocks), whose message
        # the route already saved.
        messages=[
            ChatMessage(role="user", content="Build the report", sequence=0),
            ChatMessage(role="assistant", content="On it.", sequence=1),
            ChatMessage(role="user", content=_MESSAGE, sequence=2),
        ],
    )

    await _turn_stopped_at_its_backstop(tmp_path, session, tainted=tainted)

    assert pause_of(session.messages[-1]) is not None
    assert continues_tainted(session.messages) is tainted


async def _turn_stopped_at_its_backstop(
    tmp_path: Path, session: ChatSession, *, tainted: bool
) -> None:
    transcript = build_test_transcript(
        [("user", "Build the report"), ("assistant", "On it.")]
    )
    cm = AsyncMock()
    cm.__aenter__.return_value = _client()
    cm.__aexit__.return_value = None
    patches = _make_sdk_patches(session, transcript, None, lambda *a, **k: cm) + [
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}.resolve_answered", dict(new_callable=AsyncMock, return_value=[])),
        (f"{_SVC}._update_title_async", dict(new_callable=AsyncMock)),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.persist_and_record_usage", dict(new_callable=AsyncMock)),
        (f"{_SVC}._rebuild_first_turn_context", dict(new=AsyncMock(return_value=[]))),
    ]
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for _ in stream_chat_completion_sdk(
            session_id="test-session-id",
            message=_MESSAGE,
            is_user_message=True,
            user_id="test-user",
            session=session,
            envelope=root_envelope("turn-1", tainted=tainted),
        ):
            pass


def _client() -> MagicMock:
    async def _receive():
        yield AssistantMessage(
            content=[TextBlock(text="Two of three sections done.")],
            model="claude-sonnet-4-6",
        )
        yield ResultMessage(
            subtype="error_max_turns",
            result="",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="test-session-id",
        )

    client = MagicMock(receive_response=_receive, query=AsyncMock())
    client._transport.write = AsyncMock()
    return client
