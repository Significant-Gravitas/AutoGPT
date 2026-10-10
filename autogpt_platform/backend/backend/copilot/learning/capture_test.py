"""The capture hook must see the persisted initiating user row.

Both engines hand the hook only generated rows; the user's "that worked"
lives on the row persisted before the slice. Without it a real
confirmation would be silently missed while a hand-built probe passes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from backend.copilot.model import ChatMessage, ChatSession

from . import capture
from .capture import capture_chat_turn, turn_with_initiating_user_row

USER = "user-1"


def _session(messages: list[ChatMessage]) -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id="sess-1",
        user_id=USER,
        usage=[],
        started_at=now,
        updated_at=now,
        expert_id="expert-1",
        messages=messages,
    )


def _tool(seq: int) -> ChatMessage:
    return ChatMessage(
        role="tool",
        content=json.dumps({"type": "bash_exec", "exit_code": 0}),
        sequence=seq,
    )


def test_sdk_shaped_slice_gains_the_persisted_user_row():
    earlier_user = ChatMessage(role="user", content="import it", sequence=0)
    earlier_assistant = ChatMessage(role="assistant", content="done", sequence=1)
    confirmation = ChatMessage(role="user", content="That worked, thanks.", sequence=2)
    generated = [ChatMessage(role="assistant", content="Great.", sequence=3)]
    session = _session([earlier_user, earlier_assistant, confirmation, *generated])
    # SDK: session.messages[pre_attempt_msg_count:] — generated rows only.
    turn = turn_with_initiating_user_row(session, generated, "That worked, thanks.")
    assert [m.sequence for m in turn] == [2, 3]


def test_baseline_shaped_slice_gains_the_persisted_user_row():
    confirmation = ChatMessage(role="user", content="Approved, thanks", sequence=4)
    generated = [_tool(5), ChatMessage(role="assistant", content="ok", sequence=6)]
    session = _session(
        [ChatMessage(role="user", content="x", sequence=0), confirmation, *generated]
    )
    turn = turn_with_initiating_user_row(session, generated, "Approved, thanks")
    assert [m.sequence for m in turn] == [4, 5, 6]


def test_unpersisted_user_row_is_never_referenced():
    unsaved = ChatMessage(role="user", content="That worked", sequence=None)
    generated = [ChatMessage(role="assistant", content="ok", sequence=7)]
    session = _session([unsaved, *generated])
    assert turn_with_initiating_user_row(session, generated, "That worked") == generated


def test_slice_that_already_contains_the_user_row_is_unchanged():
    user_row = ChatMessage(role="user", content="That worked", sequence=1)
    generated = [user_row, ChatMessage(role="assistant", content="ok", sequence=2)]
    session = _session(
        [ChatMessage(role="user", content="That worked", sequence=0), *generated]
    )
    assert turn_with_initiating_user_row(session, generated, "That worked") == generated


@pytest.mark.asyncio
async def test_capture_records_the_users_confirmation_signal(fake_store, monkeypatch):
    monkeypatch.setattr(capture, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(capture, "_spawn", lambda coro, name: coro.close())
    confirmation = ChatMessage(role="user", content="That worked, thanks.", sequence=2)
    generated = [_tool(3), ChatMessage(role="assistant", content="Great.", sequence=4)]
    session = _session(
        [
            ChatMessage(role="user", content="import", sequence=0),
            confirmation,
            *generated,
        ]
    )
    record = await capture_chat_turn(USER, session, generated, "That worked, thanks.")
    assert record is not None
    kinds = {(s["kind"], s["ref"]) for s in record.outcome_signals}
    assert ("user_confirmation", "msg:2") in kinds
    assert ("tool_result", "msg:3") in kinds
    assert all(":?" not in r["ref"] for r in record.evidence_refs)
    assert record.revision == "000000000004"
