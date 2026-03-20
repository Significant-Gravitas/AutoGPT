"""Public helpers for consuming a copilot stream as a simple request-response.

This module exposes :class:`CopilotResult` and :func:`collect_copilot_response`
so that callers (e.g. the AutoPilot block) can consume the copilot stream
without implementing their own event loop.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Identifiers used when registering AutoPilot-originated streams in the
# stream registry.  Distinct from "chat_stream"/"chat" used by the HTTP SSE
# endpoint, making it easy to filter AutoPilot streams in logs/observability.
AUTOPILOT_TOOL_CALL_ID = "autopilot_stream"
AUTOPILOT_TOOL_NAME = "autopilot"


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
) -> CopilotResult:
    """Consume :func:`stream_chat_completion_sdk` and return aggregated results.

    This is the recommended entry-point for callers that need a simple
    request-response interface (e.g. the AutoPilot block) rather than
    streaming individual events.  It avoids duplicating the event-collection
    logic and does NOT wrap the stream in ``asyncio.timeout`` — the SDK
    manages its own heartbeat-based timeouts internally.

    Registers with the stream registry so the frontend can connect via SSE
    and receive real-time updates while the AutoPilot block is executing.

    Args:
        session_id: Chat session to use.
        message: The user message / prompt.
        user_id: Authenticated user ID.
        is_user_message: Whether this is a user-initiated message.

    Returns:
        A :class:`CopilotResult` with the aggregated response text,
        tool calls, and token usage.

    Raises:
        RuntimeError: If the stream yields a ``StreamError`` event.
    """
    from .. import stream_registry
    from ..response_model import (
        StreamError,
        StreamFinish,
        StreamTextDelta,
        StreamToolInputAvailable,
        StreamToolOutputAvailable,
        StreamUsage,
    )
    from .service import stream_chat_completion_sdk

    result = CopilotResult()
    response_parts: list[str] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    publish_failed_once = False

    # Register with the stream registry so the frontend sees active_stream
    # and can connect via the SSE reconnect endpoint for real-time updates.
    turn_id = str(uuid.uuid4())
    error_msg: str | None = None
    try:
        await stream_registry.create_session(
            session_id=session_id,
            user_id=user_id,
            tool_call_id=AUTOPILOT_TOOL_CALL_ID,
            tool_name=AUTOPILOT_TOOL_NAME,
            turn_id=turn_id,
        )
    except Exception:
        logger.warning(
            "[collect] Failed to create stream registry session for %s, "
            "frontend will not receive real-time updates",
            session_id[:12],
            exc_info=True,
        )
        # Proceed without stream registry — AutoPilot still works,
        # just without real-time frontend updates.
        turn_id = ""

    try:
        async for event in stream_chat_completion_sdk(
            session_id=session_id,
            message=message,
            is_user_message=is_user_message,
            user_id=user_id,
        ):
            # Publish to stream registry for frontend SSE consumption.
            # Skip StreamFinish — mark_session_completed publishes it.
            if turn_id and not isinstance(event, StreamFinish):
                try:
                    await stream_registry.publish_chunk(turn_id, event)
                except Exception:
                    # Log first failure at WARNING for visibility; subsequent
                    # failures at DEBUG to avoid log spam during Redis outages.
                    if not publish_failed_once:
                        publish_failed_once = True
                        logger.warning(
                            "[collect] Failed to publish chunk %s for %s "
                            "(further failures logged at DEBUG)",
                            type(event).__name__,
                            session_id[:12],
                            exc_info=True,
                        )
                    else:
                        logger.debug(
                            "[collect] Failed to publish chunk %s",
                            type(event).__name__,
                            exc_info=True,
                        )

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
                error_msg = event.errorText
                raise RuntimeError(f"Copilot error: {event.errorText}")
    except Exception:
        if error_msg is None:
            error_msg = "AutoPilot execution failed"
        raise
    finally:
        # Mark session completed in the stream registry so the frontend
        # knows the stream has ended and stops reconnecting.
        if turn_id:
            try:
                await stream_registry.mark_session_completed(
                    session_id, error_message=error_msg
                )
            except Exception:
                logger.warning(
                    "[collect] Failed to mark stream completed for %s",
                    session_id[:12],
                    exc_info=True,
                )

    result.response_text = "".join(response_parts)
    return result
