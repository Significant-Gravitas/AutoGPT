"""Gate for MCP tools that are not in ``TOOL_REGISTRY``.

``create_copilot_mcp_server`` registers the file handlers from
``sdk/e2b_file_tools.py`` directly, so they are not ``BaseTool`` subclasses and
never reach the seam in ``BaseTool.execute``. Without this they would be the
one write path in SDK mode with no gate in front of it.

Returns an MCP-shaped error rather than a ``StreamToolOutputAvailable``,
because the caller here is the raw handler wrapper, not the tool layer.
"""

import json
import logging
import uuid
from typing import Any

from backend.copilot.model import ChatSession

from . import check_action, refusal_message
from .content import Image
from .reads import release_held_read, screen_read

logger = logging.getLogger(__name__)


async def gate_non_registry_tool(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
) -> dict[str, Any] | None:
    """An MCP error payload to return instead of running, or None to proceed."""
    try:
        decision = await check_action(tool_name, args, user_id, session)
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


async def release_non_registry_read(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
) -> dict[str, Any] | None:
    """The held read's stored envelope or its stub, or None to run the read."""
    try:
        release = await release_held_read(tool_name, args, user_id, session)
    except Exception:
        logger.warning(f"Held-read lookup failed for {tool_name}", exc_info=True)
        return _error(
            tool_name,
            "This read could not be checked against your approvals, so nothing ran.",
            None,
            session,
        )
    if release is None:
        return None
    if release.released:
        return json.loads(release.output)
    return {"content": [{"type": "text", "text": release.output}], "isError": True}


async def screen_non_registry_read(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    result: dict[str, Any],
) -> dict[str, Any]:
    """``result`` as capped for the model, or the stub that replaces it."""
    blocks = [b for b in result.get("content") or () if isinstance(b, dict)]
    text = "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
    images = tuple(
        Image(mime_type=str(b.get("mimeType", "")), data_base64=str(b["data"]))
        for b in blocks
        if b.get("type") == "image" and b.get("data")
    )
    stub = await screen_read(
        tool_name,
        args,
        user_id,
        session,
        output=json.dumps(result),
        success=not result.get("isError"),
        text=text,
        images=images,
        # The MCP handler never sees the SDK's tool_use_id; registry tools
        # on this engine use the same stand-in.
        tool_call_id=f"sdk-{uuid.uuid4().hex[:12]}",
    )
    if stub is None:
        return result
    return {"content": [{"type": "text", "text": stub}], "isError": True}


def _error(
    tool_name: str,
    reason: str,
    review_id: str | None,
    session: ChatSession,
) -> dict[str, Any]:
    payload = {
        "type": "approval_required",
        "tool_name": tool_name,
        "reason": reason,
        "message": refusal_message(reason, review_id),
        "session_id": session.session_id,
    }
    if review_id:
        payload["review_id"] = review_id
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": True}
