"""The checkpoint an engine publishes once its turn's rows so far are persisted."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from backend.copilot.model import ChatMessage
from backend.copilot.response_model import StreamCheckpoint


def turn_checkpoint(
    messages: Sequence[ChatMessage], turn_start: int
) -> StreamCheckpoint | None:
    """The checkpoint for ``messages[turn_start:]``, or None unless all of them landed.

    A row without a sequence failed to persist, a row still flagged
    ``tool_calls_pending_save`` holds tool calls its back-fill has not written,
    and a gap in the sequences means the DB rows from the first one are not
    these rows.
    """
    rows = messages[turn_start:]
    sequences = [row.sequence for row in rows if row.sequence is not None]
    if (
        not rows
        or len(sequences) != len(rows)
        or any(row.tool_calls_pending_save for row in rows)
    ):
        return None
    first = sequences[0]
    if sequences != list(range(first, first + len(rows))):
        return None
    return StreamCheckpoint(rows=len(rows), sequence=first, digest=rows_digest(rows))


def rows_digest(rows: Sequence[ChatMessage]) -> str:
    """SHA-256 of the rows' compact JSON, which ``JSON.stringify`` reproduces."""
    return canonical_digest(canonical_rows(rows))


def canonical_rows(rows: Sequence[ChatMessage]) -> list[list[Any]]:
    """Per row: role, content (a tool row's tool_call_id) and the ordered
    (id, name) tool calls; tool inputs and outputs serialise differently in JS."""
    return [
        [
            row.role,
            (row.tool_call_id if row.role == "tool" else row.content) or "",
            [
                [call.get("id", ""), call.get("function", {}).get("name", "")]
                for call in row.tool_calls or []
            ],
        ]
        for row in rows
    ]


def canonical_digest(canonical: list[list[Any]]) -> str:
    text = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
