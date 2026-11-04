"""Which chats are waiting on the user, or stopped on an error.

Both are read from the chat's last turn, the rows after the user's last
message. It waits on the user when that turn left a question, an approval
card or a proposed hire, or when the task the chat works is blocked or in
review. It failed when the turn ended on an error marker.
"""

import re
from collections.abc import Iterable
from typing import Literal

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.tools.models import ResponseType
from backend.data.db import query_raw_with_schema

SessionAttention = Literal["needs_you", "failed"]

_WAITING_RESPONSES = frozenset(
    {
        ResponseType.AGENT_BUILDER_CLARIFICATION_NEEDED.value,
        ResponseType.APPROVAL_REQUIRED.value,
        ResponseType.EXPERT_CHANGE_PROPOSED.value,
    }
)
_ERROR_STARTS = (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
)
_WAITING_TASK_STATUSES = ["blocked", "in_review"]
# A tool response is serialized with ``type`` as its first key.
_RESPONSE_TYPE = re.compile(r'^\{\s*"type"\s*:\s*"([a-z_]+)"')
_HEAD_CHARS = 200

_LAST_TURN_SQL = f"""
SELECT m."sessionId", m.role, LEFT(m.content, {_HEAD_CHARS}) AS head
FROM {{schema_prefix}}"ChatMessage" m
WHERE m."sessionId" = ANY($1::text[])
  AND m.sequence > COALESCE((
    SELECT MAX(u.sequence) FROM {{schema_prefix}}"ChatMessage" u
    WHERE u."sessionId" = m."sessionId" AND u.role = 'user'
  ), -1)
ORDER BY m."sessionId", m.sequence
"""

_WAITING_TASK_SQL = """
SELECT DISTINCT "workSessionId"
FROM {schema_prefix}"Task"
WHERE "userId" = $1
  AND "workSessionId" = ANY($2::text[])
  AND status::text = ANY($3::text[])
"""


async def session_attention(
    user_id: str, session_ids: list[str]
) -> dict[str, SessionAttention]:
    if not session_ids:
        return {}
    turn_rows = await query_raw_with_schema(_LAST_TURN_SQL, session_ids)
    waiting_rows = await query_raw_with_schema(
        _WAITING_TASK_SQL, user_id, session_ids, _WAITING_TASK_STATUSES
    )
    turns: dict[str, list[tuple[str, str | None]]] = {}
    for row in turn_rows:
        turns.setdefault(row["sessionId"], []).append((row["role"], row["head"]))
    found: dict[str, SessionAttention] = {
        row["workSessionId"]: "needs_you" for row in waiting_rows
    }
    for session_id, rows in turns.items():
        attention = turn_attention(rows)
        if attention is not None:
            found[session_id] = attention
    return found


def turn_attention(
    rows: Iterable[tuple[str, str | None]],
) -> SessionAttention | None:
    """*rows* are the ``(role, content)`` of one turn, oldest first."""
    rows = list(rows)
    if not rows:
        return None
    role, content = rows[-1]
    if role == "assistant" and content and content.startswith(_ERROR_STARTS):
        return "failed"
    if any(role == "tool" and _waits(content) for role, content in rows):
        return "needs_you"
    return None


def _waits(content: str | None) -> bool:
    match = _RESPONSE_TYPE.match(content or "")
    return match is not None and match.group(1) in _WAITING_RESPONSES
