"""Public helpers for consuming a copilot stream as a simple request-response.

This module exposes :class:`CopilotResult` and :func:`collect_copilot_response`
so that callers (e.g. the AutoPilot block) can consume the copilot stream
without implementing their own event loop.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from .. import stream_registry
from ..response_model import (
    StreamError,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamUsage,
)
from .service import stream_chat_completion_sdk

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


class _RegistryHandle(BaseModel):
    """Tracks stream registry session state for cleanup."""

    publish_turn_id: str = ""
    error_msg: str | None = None
    error_already_published: bool = False


@asynccontextmanager
async def _registry_session(
    session_id: str, user_id: str, turn_id: str
) -> AsyncIterator[_RegistryHandle]:
    """Create a stream registry session and ensure it is finalized."""
    handle = _RegistryHandle(publish_turn_id=turn_id)
    try:
        await stream_registry.create_session(
            session_id=session_id,
            user_id=user_id,
            tool_call_id=AUTOPILOT_TOOL_CALL_ID,
            tool_name=AUTOPILOT_TOOL_NAME,
            turn_id=turn_id,
        )
    except (RedisError, ConnectionError, OSError):
        logger.warning(
            "[collect] Failed to create stream registry session for %s, "
            "frontend will not receive real-time updates",
            session_id[:12],
            exc_info=True,
        )
        # Disable chunk publishing but keep finalization enabled so
        # mark_session_completed can clean up any partial registry state.
        handle.publish_turn_id = ""

    try:
        yield handle
    finally:
        try:
            await stream_registry.mark_session_completed(
                session_id,
                error_message=handle.error_msg,
                skip_error_publish=handle.error_already_published,
            )
        except (RedisError, ConnectionError, OSError):
            logger.warning(
                "[collect] Failed to mark stream completed for %s",
                session_id[:12],
                exc_info=True,
            )


class _ToolCallEntry(BaseModel):
    """A single tool call observed during stream consumption."""

    tool_call_id: str
    tool_name: str
    input: Any
    output: Any = None
    success: bool | None = None


class _EventAccumulator(BaseModel):
    """Mutable accumulator for stream events."""

    response_parts: list[str] = Field(default_factory=list)
    tool_calls: list[_ToolCallEntry] = Field(default_factory=list)
    tool_calls_by_id: dict[str, _ToolCallEntry] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _process_event(event: object, acc: _EventAccumulator) -> str | None:
    """Process a single stream event and return error_msg if StreamError.

    Uses structural pattern matching for dispatch per project guidelines.
    """
    match event:
        case StreamTextDelta(delta=delta):
            acc.response_parts.append(delta)
        case StreamToolInputAvailable() as e:
            entry = _ToolCallEntry(
                tool_call_id=e.toolCallId,
                tool_name=e.toolName,
                input=e.input,
            )
            acc.tool_calls.append(entry)
            acc.tool_calls_by_id[e.toolCallId] = entry
        case StreamToolOutputAvailable() as e:
            if tc := acc.tool_calls_by_id.get(e.toolCallId):
                tc.output = e.output
                tc.success = e.success
            else:
                logger.debug(
                    "Received tool output for unknown tool_call_id: %s",
                    e.toolCallId,
                )
        case StreamUsage() as e:
            acc.prompt_tokens += e.prompt_tokens
            acc.completion_tokens += e.completion_tokens
            acc.total_tokens += e.total_tokens
        case StreamError(errorText=err):
            return err
    return None


async def collect_copilot_response(
    *,
    session_id: str,
    message: str,
    user_id: str,
    is_user_message: bool = True,
    permissions: "CopilotPermissions | None" = None,
) -> CopilotResult:
    """Consume :func:`stream_chat_completion_sdk` and return aggregated results.

    Registers with the stream registry so the frontend can connect via SSE
    and receive real-time updates while the AutoPilot block is executing.

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
    turn_id = str(uuid.uuid4())
    async with _registry_session(session_id, user_id, turn_id) as handle:
        try:
            raw_stream = stream_chat_completion_sdk(
                session_id=session_id,
                message=message,
                is_user_message=is_user_message,
                user_id=user_id,
                permissions=permissions,
            )
            published_stream = stream_registry.stream_and_publish(
                session_id=session_id,
                turn_id=handle.publish_turn_id,
                stream=raw_stream,
            )

            acc = _EventAccumulator()
            async for event in published_stream:
                if err := _process_event(event, acc):
                    handle.error_msg = err
                    # stream_and_publish skips StreamError events, so
                    # mark_session_completed must publish the error to Redis.
                    handle.error_already_published = False
                    raise RuntimeError(f"Copilot error: {err}")
        except Exception:
            if handle.error_msg is None:
                handle.error_msg = "AutoPilot execution failed"
            raise

    result = CopilotResult()
    result.response_text = "".join(acc.response_parts)
    result.tool_calls = [tc.model_dump() for tc in acc.tool_calls]
    result.prompt_tokens = acc.prompt_tokens
    result.completion_tokens = acc.completion_tokens
    result.total_tokens = acc.total_tokens
    return result
