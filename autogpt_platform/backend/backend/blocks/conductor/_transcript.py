"""Helpers for waiting on a Conductor agent and reading its transcript."""

import asyncio
import json
import time
from typing import Any

from ._api import ConductorClient


async def wait_for_idle(
    client: ConductorClient,
    session_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[dict[str, Any], bool]:
    """Poll session status until it leaves `working`.

    Returns the last status payload and whether the wait timed out. The first
    poll happens after one interval so a just-queued message has time to flip
    the session to `working`.
    """
    deadline = time.monotonic() + timeout_seconds
    status: dict[str, Any] = {}
    while True:
        await asyncio.sleep(poll_interval_seconds)
        status = await client.session_status(session_id)
        if status.get("status") in ("idle", "error"):
            return status, False
        if time.monotonic() >= deadline:
            return status, True


def message_text(message: dict[str, Any]) -> str:
    """Best-effort text of one transcript message.

    `content` is untyped in the OpenAPI spec: accept a string, a dict with a
    `text` field, or a list of content parts, and fall back to JSON.
    """
    content = message.get("content")
    return _content_text(content)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            if key in content:
                return _content_text(content[key])
        return json.dumps(content)
    if isinstance(content, list):
        parts = [_content_text(part) for part in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def is_agent_message(message: dict[str, Any]) -> bool:
    kind = str(message.get("type") or "").lower()
    return kind not in ("user", "human", "prompt", "system")


def reply_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate the agent-side messages of a transcript slice."""
    texts = [message_text(m) for m in messages if is_agent_message(m)]
    return "\n\n".join(t for t in texts if t)


async def wait_for_reply(
    client: ConductorClient,
    session_id: str,
    after_message_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Wait for the agent to go idle, then return the messages after the prompt."""
    status, timed_out = await wait_for_idle(
        client, session_id, timeout_seconds, poll_interval_seconds
    )
    listing = await client.list_messages(session_id, after=after_message_id)
    messages = listing.get("data") or []
    return {
        "session_status": status.get("status") or "",
        "error_message": status.get("errorMessage") or status.get("lastError") or "",
        "messages": messages,
        "reply": reply_text(messages),
        "timed_out": timed_out,
    }
