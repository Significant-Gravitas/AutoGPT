"""The wait/poll blocks answer compactly by default (SECRT-2790).

A turn's raw transcript rows run to hundreds of kilobytes, which pushed every
poll through the copilot's digest and a sandbox re-parse. The blocks now emit
`reply`/`latest_reply`, the status, `next_after` and a row count by default
and the raw rows only when `include_messages` is on.
"""

import json
from typing import Any
from unittest import mock

import pytest

from backend.blocks.conductor.create_session import ConductorCreateSessionBlock
from backend.blocks.conductor.create_workspace import ConductorCreateWorkspaceBlock
from backend.blocks.conductor.get_session import ConductorGetSessionBlock
from backend.blocks.conductor.send_message import ConductorSendMessageBlock
from backend.blocks.conductor.test_fixtures import (
    CLAUDE_TOOL_USE,
    OLD_TURN,
    RECEIPT,
    TEST_CREDENTIALS_INPUT,
    FakeTranscript,
    agent_row,
    claude_text,
    claude_turn,
    collect,
    mock_block,
    user_row,
    wait_client,
    wait_clock,
)
from backend.copilot.tools.base import _DIGEST_THRESHOLD

COMPACT_WAIT_OUTPUTS = {
    "session_status",
    "reply",
    "next_after",
    "message_count",
    "timed_out",
    "truncated",
    "error_message",
}


def long_turn(rows: int = 1000) -> list[dict[str, Any]]:
    """A prompt followed by tool calls and interim text, the shape of a real
    coding-agent turn, long enough to fill the kept-row budget."""
    turn = [user_row("row-0", RECEIPT, 0)]
    for i in range(1, rows):
        raw = CLAUDE_TOOL_USE if i % 2 else claude_text(f"Step {i} of the work.")
        turn.append(agent_row(f"row-{i}", RECEIPT, i, raw))
    return turn


def waited_with(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "session_status": "idle",
        "error_message": "",
        "messages": rows,
        "reply": "Done.",
        "timed_out": False,
        "truncated": False,
        "prompt_row_id": rows[0]["id"],
    }


def serialized_size(outputs: dict[str, Any]) -> int:
    """Size of the outputs as the copilot's run_capability tool serialises a
    block result (each output name mapped to its list of values)."""
    return len(json.dumps({name: [value] for name, value in outputs.items()}))


SENT = {"messageId": RECEIPT, "state": "sent", "deepLink": "conductor://m/1"}
CREATED_SESSION = {
    "id": "sess_1",
    "deepLink": "conductor://s/1",
    "initialMessage": SENT,
}
CREATED_WORKSPACE = {
    "workspaceId": "ws_1",
    "sessionId": "sess_1",
    "deepLink": "conductor://w/1",
    "initialMessage": SENT,
}


def waiting_blocks(rows: list[dict[str, Any]]):
    waited = waited_with(rows)
    cases = [
        (
            ConductorSendMessageBlock(),
            {"_send": lambda *a, **k: SENT, "_wait": lambda *a, **k: waited},
            {"session_id": "s1", "message": "go"},
        ),
        (
            ConductorCreateSessionBlock(),
            {
                "_create": lambda *a, **k: CREATED_SESSION,
                "_wait": lambda *a, **k: waited,
            },
            {"workspace_id": "ws_1", "message": "go", "wait_for_reply": True},
        ),
        (
            ConductorCreateWorkspaceBlock(),
            {
                "_create": lambda *a, **k: CREATED_WORKSPACE,
                "_wait": lambda *a, **k: waited,
            },
            {
                "repository_url": "https://github.com/x/y",
                "message": "go",
                "wait_for_reply": True,
            },
        ),
    ]
    for block, mocks, inputs in cases:
        mock_block(block, mocks)
        yield block, {"credentials": TEST_CREDENTIALS_INPUT, **inputs}


@pytest.mark.asyncio
async def test_waiting_blocks_omit_raw_rows_by_default():
    rows = long_turn(5)
    for block, inputs in waiting_blocks(rows):
        outputs = await collect(block, inputs)

        assert "messages" not in outputs, type(block).__name__
        assert COMPACT_WAIT_OUTPUTS <= set(outputs), type(block).__name__
        assert outputs["reply"] == "Done."
        assert outputs["next_after"] == "row-0"
        assert outputs["message_count"] == 5


@pytest.mark.asyncio
async def test_waiting_blocks_return_raw_rows_on_request():
    rows = long_turn(5)
    for block, inputs in waiting_blocks(rows):
        outputs = await collect(block, {**inputs, "include_messages": True})

        assert outputs["messages"] == rows, type(block).__name__
        assert outputs["message_count"] == 5


@pytest.mark.asyncio
async def test_default_wait_outputs_stay_under_the_digest_threshold():
    rows = long_turn(1000)
    for block, inputs in waiting_blocks(rows):
        compact = await collect(block, inputs)
        raw = await collect(block, {**inputs, "include_messages": True})

        assert serialized_size(compact) < _DIGEST_THRESHOLD, type(block).__name__
        assert serialized_size(raw) > _DIGEST_THRESHOLD, type(block).__name__


def get_session_block(rows: list[dict[str, Any]]) -> ConductorGetSessionBlock:
    block = ConductorGetSessionBlock()
    mock_block(
        block,
        {
            "_fetch": lambda *a, **k: {
                "session": {"id": "sess_1", "deepLink": "conductor://s/1"},
                "status": {"status": "working"},
                "messages": {"data": rows, "hasMore": True},
            }
        },
    )
    return block


@pytest.mark.asyncio
async def test_get_session_omits_raw_rows_by_default():
    rows = long_turn(500)
    outputs = await collect(
        get_session_block(rows),
        {"credentials": TEST_CREDENTIALS_INPUT, "session_id": "s1"},
    )

    assert "messages" not in outputs
    assert outputs["status"] == "working"
    assert outputs["latest_reply"] == "Step 498 of the work."
    assert outputs["next_after"] == "row-499"
    assert outputs["message_count"] == 500
    assert outputs["has_more"] is True
    assert serialized_size(outputs) < _DIGEST_THRESHOLD


@pytest.mark.asyncio
async def test_get_session_returns_raw_rows_on_request():
    rows = long_turn(3)
    outputs = await collect(
        get_session_block(rows),
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "session_id": "s1",
            "include_messages": True,
        },
    )

    assert outputs["messages"] == rows
    assert outputs["message_count"] == 3


@pytest.mark.asyncio
async def test_wait_next_after_hands_off_to_get_session():
    """A compact wait result is enough to keep polling: its next_after,
    passed as Get Session's after, pages the turn from the prompt row with no
    receipt lookup and surfaces the turn's answer as latest_reply."""
    turn = claude_turn(RECEIPT, 2, "Done")
    rows = [
        user_row("row-old", OLD_TURN, 0),
        agent_row("row-old-reply", OLD_TURN, 1, claude_text("Earlier answer")),
        user_row("row-prompt", RECEIPT, 2),
        *turn,
    ]
    transcript = FakeTranscript(rows)
    sender = ConductorSendMessageBlock()
    mock_block(sender, {"_send": lambda *a, **k: SENT})
    with (
        mock.patch(
            "backend.blocks.conductor.send_message.ConductorClient",
            return_value=wait_client(transcript, [{"status": "idle"}]),
        ),
        wait_clock(),
    ):
        waited = await collect(
            sender,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "message": "go",
            },
        )

    assert "messages" not in waited
    assert waited["reply"].endswith("Done")
    assert waited["next_after"] == "row-prompt"

    transcript.calls.clear()
    session_client = mock.Mock()
    session_client.get_session = mock.AsyncMock(return_value={"id": "s1"})
    session_client.session_status = mock.AsyncMock(return_value={"status": "idle"})
    session_client.list_messages = transcript.list_messages
    with mock.patch(
        "backend.blocks.conductor.get_session.ConductorClient",
        return_value=session_client,
    ):
        polled = await collect(
            ConductorGetSessionBlock(),
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "session_id": "s1",
                "after": waited["next_after"],
                "message_limit": 20,
            },
        )

    assert polled["latest_reply"] == "Done"
    assert polled["message_count"] == len(turn)
    assert polled["next_after"] == turn[-1]["id"]
    assert transcript.calls == [{"after": "row-prompt", "limit": 20, "offset": None}]
