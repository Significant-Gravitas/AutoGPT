import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.chat.session_attention import (
    session_attention,
    turn_attention,
)
from backend.copilot.constants import COPILOT_ERROR_PREFIX, STREAM_ERROR_MARKER


def _tool(response_type: str) -> tuple[str, str]:
    return ("tool", json.dumps({"type": response_type, "message": "..."}))


@pytest.mark.parametrize(
    "rows,expected",
    [
        ([("assistant", f"{COPILOT_ERROR_PREFIX} The model is down.")], "failed"),
        ([_tool("agent_output"), ("assistant", STREAM_ERROR_MARKER)], "failed"),
        (
            [
                _tool("expert_change_proposed"),
                _tool("expert_change_proposed"),
                ("assistant", "Approve the two hires above."),
            ],
            "needs_you",
        ),
        ([_tool("approval_required")], "needs_you"),
        (
            [_tool("agent_builder_clarification_needed"), ("assistant", "?")],
            "needs_you",
        ),
        ([_tool("agent_output"), ("assistant", "Here is your summary.")], None),
        ([("tool", "plain text result")], None),
        ([("assistant", None)], None),
        ([], None),
    ],
)
def test_turn_attention(rows, expected):
    assert turn_attention(rows) == expected


async def test_a_waiting_task_and_the_last_turn_both_count():
    async def fake_query(sql, *args):
        if '"Task"' in sql:
            return [{"workSessionId": "chat-task"}]
        return [
            {
                "sessionId": "chat-error",
                "role": "assistant",
                "head": STREAM_ERROR_MARKER,
            },
            {
                "sessionId": "chat-hire",
                **dict(zip(("role", "head"), _tool("expert_change_proposed"))),
            },
            {"sessionId": "chat-hire", "role": "assistant", "head": "Approve them."},
            {"sessionId": "chat-done", "role": "assistant", "head": "All set."},
        ]

    with patch(
        "backend.api.features.chat.session_attention.query_raw_with_schema",
        AsyncMock(side_effect=fake_query),
    ):
        found = await session_attention(
            "user-1", ["chat-task", "chat-error", "chat-hire", "chat-done"]
        )

    assert found == {
        "chat-task": "needs_you",
        "chat-error": "failed",
        "chat-hire": "needs_you",
    }


async def test_no_sessions_skips_the_queries():
    with patch(
        "backend.api.features.chat.session_attention.query_raw_with_schema",
        AsyncMock(),
    ) as query:
        assert await session_attention("user-1", []) == {}
    query.assert_not_awaited()
