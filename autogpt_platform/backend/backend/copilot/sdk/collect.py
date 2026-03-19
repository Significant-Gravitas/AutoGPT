"""Public helpers for consuming a copilot stream as a simple request-response.

This module exposes :class:`CopilotResult` and :func:`collect_copilot_response`
so that callers (e.g. the AutoPilot block) can consume the copilot stream
without implementing their own event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions


class CopilotResult:
    """Aggregated result from consuming a copilot stream.

    Returned by :func:`collect_copilot_response` so callers don't need to
    implement their own event-loop over the raw stream events.
    """

    __slots__ = (
        "response_text",
        "tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    )

    def __init__(self) -> None:
        self.response_text: str = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0


async def collect_copilot_response(
    *,
    session_id: str,
    message: str,
    user_id: str,
    is_user_message: bool = True,
    permissions: "CopilotPermissions | None" = None,
) -> CopilotResult:
    """Consume :func:`stream_chat_completion_sdk` and return aggregated results.

    This is the recommended entry-point for callers that need a simple
    request-response interface (e.g. the AutoPilot block) rather than
    streaming individual events.  It avoids duplicating the event-collection
    logic and does NOT wrap the stream in ``asyncio.timeout`` — the SDK
    manages its own heartbeat-based timeouts internally.

    Args:
        session_id: Chat session to use.
        message: The user message / prompt.
        user_id: Authenticated user ID.
        is_user_message: Whether this is a user-initiated message.
        permissions: Optional capability filter.  When provided, restricts
            which tools and blocks the copilot may use during this execution.

    Returns:
        A :class:`CopilotResult` with the aggregated response text,
        tool calls, and token usage.

    Raises:
        RuntimeError: If the stream yields a ``StreamError`` event.
    """
    from backend.copilot.response_model import (
        StreamError,
        StreamTextDelta,
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
        StreamUsage,
    )

    from .service import stream_chat_completion_sdk

    result = CopilotResult()
    response_parts: list[str] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}

    async for event in stream_chat_completion_sdk(
        session_id=session_id,
        message=message,
        is_user_message=is_user_message,
        user_id=user_id,
        permissions=permissions,
    ):
        if isinstance(event, StreamTextDelta):
            response_parts.append(event.delta)
        elif isinstance(event, StreamToolInputAvailable):
            entry: dict[str, Any] = {
                "tool_call_id": event.toolCallId,
                "tool_name": event.toolName,
                "input": event.input,
                "output": None,
                "success": None,
            }
            result.tool_calls.append(entry)
            tool_calls_by_id[event.toolCallId] = entry
        elif isinstance(event, StreamToolOutputAvailable):
            if tc := tool_calls_by_id.get(event.toolCallId):
                tc["output"] = event.output
                tc["success"] = event.success
        elif isinstance(event, StreamUsage):
            result.prompt_tokens += event.prompt_tokens
            result.completion_tokens += event.completion_tokens
            result.total_tokens += event.total_tokens
        elif isinstance(event, StreamError):
            raise RuntimeError(f"Copilot error: {event.errorText}")

    result.response_text = "".join(response_parts)
    return result
