"""Gate for MCP tools that are not in ``TOOL_REGISTRY``.

``create_copilot_mcp_server`` registers the file handlers from
``sdk/e2b_file_tools.py`` directly, so they are not ``BaseTool`` subclasses and
never reach the seam in ``BaseTool.execute``. Without this they would be the
one write path in SDK mode with no gate in front of it, and the one read path
that marks no provenance.

Returns an MCP-shaped error rather than a ``StreamToolOutputAvailable``,
because the caller here is the raw handler wrapper, not the tool layer.
"""

import json
import logging
from typing import Any

from backend.copilot.model import ChatSession

from . import check_action, note_taint_source
from .review import session_exec_id

logger = logging.getLogger(__name__)


async def gate_non_registry_tool(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
) -> dict[str, Any] | None:
    """An MCP error payload to return instead of running, or None to proceed."""
    try:
        await note_taint_source(session.session_id, tool_name)
        decision = await check_action(
            tool_name,
            args,
            user_id,
            session,
            tool_description=f"MCP file tool {tool_name}",
        )
    except Exception:
        logger.warning(f"Action gate failed for MCP tool {tool_name}", exc_info=True)
        return _error(
            tool_name,
            "This action could not be checked against your approval settings, "
            "so nothing ran.",
            None,
            session,
        )

    if decision.allowed:
        return None
    return _error(tool_name, decision.reason, decision.review_id, session)


def _error(
    tool_name: str,
    reason: str,
    review_id: str | None,
    session: ChatSession,
) -> dict[str, Any]:
    # ``graph_exec_id`` is what mounts the chat's approval card — the frontend
    # scans tool outputs for that key.
    payload = {
        "type": "approval_required",
        "tool_name": tool_name,
        "reason": reason,
        "message": (
            f"Nothing ran. {reason} Tell the user exactly what you wanted to "
            "do and why, then stop. Do not retry, do not work around it, and "
            "do not use another tool for the same effect."
        ),
        "session_id": session.session_id,
    }
    if review_id:
        payload["review_id"] = review_id
        payload["graph_exec_id"] = session_exec_id(session.session_id)
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": True}
