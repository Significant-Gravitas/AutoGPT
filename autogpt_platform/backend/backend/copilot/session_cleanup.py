"""Pre-turn cleanup of transient markers left on ``session.messages`` by
prior turns (user-initiated Stop, cancelled tool calls, etc.).

Shared by both the SDK and baseline chat entry points so both code paths
start every new turn from a well-formed message list.
"""

import logging

from backend.copilot.constants import (
    STOPPED_BY_USER_MARKER,
    STREAM_ERROR_MARKER,
    STREAM_INCOMPLETE_MARKER,
)
from backend.copilot.model import ChatMessage

logger = logging.getLogger(__name__)


def prune_orphan_tool_calls(
    messages: list[ChatMessage],
    log_prefix: str | None = None,
) -> int:
    """Pop trailing orphan tool-use blocks from *messages* in place.

    A Stop mid-tool-call leaves the session ending on an assistant message
    whose ``tool_calls`` have no matching ``role="tool"`` row — the tool
    never produced output because the executor was cancelled.  Feeding that
    tail to the next ``--resume`` turn would hand the Claude CLI a
    ``tool_use`` with no paired ``tool_result`` and the SDK raises a
    generic error.

    Also strips trailing ``STOPPED_BY_USER_MARKER``,
    ``STREAM_INCOMPLETE_MARKER``, and ``STREAM_ERROR_MARKER`` assistant rows
    so the next turn's transcript starts clean — these synthetic notices
    must not leak into the next ``--resume`` turn's history.

    If *log_prefix* is given, emits an INFO log with the prefix whenever
    something was actually popped so the turn-start cleanup is visible.

    In-memory only — the DB write path is append-only via
    ``start_sequence`` so no delete is needed; the same rows are popped
    again on the next session load.
    """
    cut_index: int | None = None
    resolved_ids: set[str] = set()

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        if msg.role == "tool" and msg.tool_call_id:
            resolved_ids.add(msg.tool_call_id)
            continue

        if msg.role == "assistant" and msg.content in (
            STOPPED_BY_USER_MARKER,
            STREAM_INCOMPLETE_MARKER,
            STREAM_ERROR_MARKER,
        ):
            cut_index = i
            continue

        if msg.role == "assistant" and msg.tool_calls:
            pending_ids = {
                tc.get("id")
                for tc in msg.tool_calls
                if isinstance(tc, dict) and tc.get("id")
            }
            if pending_ids and not pending_ids.issubset(resolved_ids):
                cut_index = i
            break

        break

    if cut_index is None:
        return 0

    removed = len(messages) - cut_index
    del messages[cut_index:]
    if log_prefix:
        logger.info(
            "%s Dropped %d trailing orphan tool-use/stop row(s) "
            "before starting new turn",
            log_prefix,
            removed,
        )
    return removed
