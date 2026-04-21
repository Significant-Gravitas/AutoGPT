"""Pre-turn cleanup of transient markers left on ``session.messages`` by
prior turns (user-initiated Stop, cancelled tool calls, etc.).

Shared by both the SDK and baseline chat entry points so both code paths
start every new turn from a well-formed message list.
"""

from backend.copilot.constants import STOPPED_BY_USER_MARKER
from backend.copilot.model import ChatMessage


def prune_orphan_tool_calls(messages: list[ChatMessage]) -> int:
    """Pop trailing orphan tool-use blocks from *messages* in place.

    A Stop mid-tool-call leaves the session ending on an assistant message
    whose ``tool_calls`` have no matching ``role="tool"`` row — the tool
    never produced output because the executor was cancelled.  Feeding that
    tail to the next ``--resume`` turn would hand the Claude CLI a
    ``tool_use`` with no paired ``tool_result`` and the SDK raises a
    generic error.

    Also strips trailing ``STOPPED_BY_USER_MARKER`` assistant rows emitted
    by the same Stop path so the next turn's transcript starts clean.

    In-memory only — the DB write path is append-only via
    ``start_sequence`` so no delete is needed; the same rows are popped
    again on the next session load.
    """
    removed = 0

    # The Stop path appends exactly one STOPPED_BY_USER_MARKER row at the
    # tail, so a single check is enough.
    if (
        messages
        and messages[-1].role == "assistant"
        and messages[-1].content == STOPPED_BY_USER_MARKER
    ):
        messages.pop()
        removed += 1

    cut_index: int | None = None
    resolved_ids: set[str] = set()
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == "tool" and msg.tool_call_id:
            resolved_ids.add(msg.tool_call_id)
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

    if cut_index is not None:
        dropped = len(messages) - cut_index
        del messages[cut_index:]
        removed += dropped

    return removed
