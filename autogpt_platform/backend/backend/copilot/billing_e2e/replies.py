"""What the fake model answers, read off each request the way a model would.

A user message carrying ``TOOL_MARKER`` gets one tool call; the tool's result
gets a closing answer; anything else gets a greeting.
"""

import json
from collections.abc import Iterator
from typing import Any

from .invoice import CacheTtl
from .wire import Reply

TOOL_MARKER = "[use-tool]"
GREETING = "Hello from the fake provider."
AFTER_TOOL = "Here is what the lookup found."
# The Codex gateway hands the whole chat over as one message: a preamble,
# then the conversation as a JSON array of Anthropic-shaped messages.
_GATEWAY_TRANSCRIPT_MARKER = "The transcript is JSON:\n"


def anthropic_reply(messages: list[dict[str, Any]]) -> Reply:
    # The CLI can follow the user's message with a system reminder, so the
    # last user message is the one to answer.
    content = _latest(messages, {"user"}).get("content")
    if isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    ):
        return Reply(kind="text", text=AFTER_TOOL)
    return _reply_to(json.dumps(content))


def chat_reply(messages: list[dict[str, Any]]) -> Reply:
    latest = _latest(messages, {"user", "tool"})
    if latest.get("role") == "tool":
        return Reply(kind="text", text=AFTER_TOOL)
    return _reply_to(json.dumps(latest.get("content")))


def responses_reply(items: list[dict[str, Any]]) -> Reply:
    last = items[-1] if items else {}
    if last.get("type") == "function_call_output":
        return Reply(kind="text", text=AFTER_TOOL)
    prompt = "".join(
        part.get("text", "")
        for part in last.get("content") or []
        if isinstance(part, dict)
    )
    _, found, transcript = prompt.partition(_GATEWAY_TRANSCRIPT_MARKER)
    return anthropic_reply(json.loads(transcript)) if found else _reply_to(prompt)


def cache_ttl(body: object) -> CacheTtl | None:
    """The cache lifetime the request asked for; None when it asked for none."""
    ttls = set(_cache_control_ttls(body))
    if not ttls:
        return None
    return "1h" if "1h" in ttls else "5m"


def _latest(messages: list[dict[str, Any]], roles: set[str]) -> dict[str, Any]:
    return next((m for m in reversed(messages) if m.get("role") in roles), {})


def _reply_to(latest_user_message: str) -> Reply:
    if TOOL_MARKER in latest_user_message:
        return Reply(kind="tool")
    return Reply(kind="text", text=GREETING)


def _cache_control_ttls(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        control = node.get("cache_control")
        if isinstance(control, dict):
            yield str(control.get("ttl") or "5m")
        for value in node.values():
            yield from _cache_control_ttls(value)
    elif isinstance(node, list):
        for value in node:
            yield from _cache_control_ttls(value)
