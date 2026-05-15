"""
Response models for Vercel AI SDK UI Stream Protocol.

This module implements the AI SDK UI Stream Protocol (v1) for streaming chat responses.
See: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

import json
import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.util.json import dumps as json_dumps
from backend.util.truncate import truncate

logger = logging.getLogger(__name__)


class ResponseType(str, Enum):
    """Types of streaming responses following AI SDK protocol."""

    # Message lifecycle
    START = "start"
    FINISH = "finish"

    # Step lifecycle (one LLM API call within a message)
    START_STEP = "start-step"
    FINISH_STEP = "finish-step"

    # Text streaming
    TEXT_START = "text-start"
    TEXT_DELTA = "text-delta"
    TEXT_END = "text-end"

    # Reasoning streaming (extended_thinking content blocks).  Matches
    # the Vercel AI SDK v5 wire names so the client's ``useChat``
    # transport accumulates these into a ``type: 'reasoning'`` UIMessage
    # part that the ``ReasoningCollapse`` component renders collapsed by
    # default.
    REASONING_START = "reasoning-start"
    REASONING_DELTA = "reasoning-delta"
    REASONING_END = "reasoning-end"

    # Tool interaction
    TOOL_INPUT_START = "tool-input-start"
    TOOL_INPUT_AVAILABLE = "tool-input-available"
    TOOL_OUTPUT_AVAILABLE = "tool-output-available"

    # Other
    ERROR = "error"
    USAGE = "usage"
    HEARTBEAT = "heartbeat"
    STATUS = "data-status"
    CURSOR = "data-cursor"


class StreamBaseResponse(BaseModel):
    """Base response model for all streaming responses."""

    type: ResponseType

    def to_sse(self) -> str:
        """Convert to SSE format."""
        json_str = self.model_dump_json(exclude_none=True)
        return f"data: {json_str}\n\n"


# ========== Message Lifecycle ==========


class StreamStart(StreamBaseResponse):
    """Start of a new message."""

    type: ResponseType = ResponseType.START
    messageId: str = Field(..., description="Unique message ID")
    sessionId: str | None = Field(
        default=None,
        description="Session ID for SSE reconnection.",
    )

    def to_sse(self) -> str:
        """Convert to SSE format, excluding non-protocol fields like sessionId."""
        data: dict[str, Any] = {
            "type": self.type.value,
            "messageId": self.messageId,
        }
        return f"data: {json.dumps(data)}\n\n"


class StreamFinish(StreamBaseResponse):
    """End of message/stream."""

    type: ResponseType = ResponseType.FINISH


class StreamStartStep(StreamBaseResponse):
    """Start of a step (one LLM API call within a message).

    The AI SDK uses this to add a step-start boundary to message.parts,
    enabling visual separation between multiple LLM calls in a single message.
    """

    type: ResponseType = ResponseType.START_STEP


class StreamFinishStep(StreamBaseResponse):
    """End of a step (one LLM API call within a message).

    The AI SDK uses this to reset activeTextParts and activeReasoningParts,
    so the next LLM call in a tool-call continuation starts with clean state.
    """

    type: ResponseType = ResponseType.FINISH_STEP


# ========== Text Streaming ==========


class StreamTextStart(StreamBaseResponse):
    """Start of a text block."""

    type: ResponseType = ResponseType.TEXT_START
    id: str = Field(..., description="Text block ID")


class StreamTextDelta(StreamBaseResponse):
    """Streaming text content delta."""

    type: ResponseType = ResponseType.TEXT_DELTA
    id: str = Field(..., description="Text block ID")
    delta: str = Field(..., description="Text content delta")


class StreamTextEnd(StreamBaseResponse):
    """End of a text block."""

    type: ResponseType = ResponseType.TEXT_END
    id: str = Field(..., description="Text block ID")


# ========== Reasoning Streaming ==========


class StreamReasoningStart(StreamBaseResponse):
    """Start of a reasoning block (extended_thinking content)."""

    type: ResponseType = ResponseType.REASONING_START
    id: str = Field(..., description="Reasoning block ID")


class StreamReasoningDelta(StreamBaseResponse):
    """Streaming reasoning content delta."""

    type: ResponseType = ResponseType.REASONING_DELTA
    id: str = Field(..., description="Reasoning block ID")
    delta: str = Field(..., description="Reasoning content delta")


class StreamReasoningEnd(StreamBaseResponse):
    """End of a reasoning block."""

    type: ResponseType = ResponseType.REASONING_END
    id: str = Field(..., description="Reasoning block ID")


# ========== Tool Interaction ==========


class StreamToolInputStart(StreamBaseResponse):
    """Tool call started notification."""

    type: ResponseType = ResponseType.TOOL_INPUT_START
    toolCallId: str = Field(..., description="Unique tool call ID")
    toolName: str = Field(..., description="Name of the tool being called")


class StreamToolInputAvailable(StreamBaseResponse):
    """Tool input is ready for execution."""

    type: ResponseType = ResponseType.TOOL_INPUT_AVAILABLE
    toolCallId: str = Field(..., description="Unique tool call ID")
    toolName: str = Field(..., description="Name of the tool being called")
    input: dict[str, Any] = Field(
        default_factory=dict, description="Tool input arguments"
    )


_MAX_TOOL_OUTPUT_SIZE = 100_000  # ~100 KB; truncate to avoid bloating SSE/DB


class StreamToolOutputAvailable(StreamBaseResponse):
    """Tool execution result."""

    type: ResponseType = ResponseType.TOOL_OUTPUT_AVAILABLE
    toolCallId: str = Field(..., description="Tool call ID this responds to")
    output: str | dict[str, Any] = Field(..., description="Tool execution output")
    # Keep these for internal backend use
    toolName: str | None = Field(
        default=None, description="Name of the tool that was executed"
    )
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )

    def model_post_init(self, __context: Any) -> None:
        """Truncate oversized outputs after construction."""
        self.output = truncate(self.output, _MAX_TOOL_OUTPUT_SIZE)

    def to_sse(self) -> str:
        """Convert to SSE format, excluding non-spec fields."""
        data = {
            "type": self.type.value,
            "toolCallId": self.toolCallId,
            "output": self.output,
        }
        return f"data: {json.dumps(data)}\n\n"


# ========== Other ==========


class StreamUsage(StreamBaseResponse):
    """Token usage statistics.

    Emitted as an SSE comment so the Vercel AI SDK parser ignores it
    (it uses z.strictObject() and rejects unknown event types).
    Usage data is recorded server-side (session DB + Redis counters).
    """

    type: ResponseType = ResponseType.USAGE
    prompt_tokens: int = Field(
        ...,
        serialization_alias="promptTokens",
        description="Number of uncached prompt tokens",
    )
    completion_tokens: int = Field(
        ...,
        serialization_alias="completionTokens",
        description="Number of completion tokens",
    )
    total_tokens: int = Field(
        ...,
        serialization_alias="totalTokens",
        description="Total number of tokens (raw, not weighted)",
    )
    cache_read_tokens: int = Field(
        default=0,
        serialization_alias="cacheReadTokens",
        description="Prompt tokens served from cache (10% cost)",
    )
    cache_creation_tokens: int = Field(
        default=0,
        serialization_alias="cacheCreationTokens",
        description="Prompt tokens written to cache (25% cost)",
    )

    def to_sse(self) -> str:
        """Emit as SSE comment so the AI SDK parser ignores it."""
        return f": usage {self.model_dump_json(exclude_none=True, by_alias=True)}\n\n"


class StreamError(StreamBaseResponse):
    """Error response."""

    type: ResponseType = ResponseType.ERROR
    errorText: str = Field(..., description="Error message text")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )

    def to_sse(self) -> str:
        """Convert to SSE format, only emitting fields required by AI SDK protocol.

        The AI SDK uses z.strictObject({type, errorText}) which rejects
        any extra fields like `code` or `details`.

        When ``code`` is set we prefix ``errorText`` with ``[code:<id>]`` so
        the frontend can still parse a machine-readable code out of the
        otherwise opaque text. Idempotent: if the caller already embedded
        the prefix, we don't double it.
        """
        text = self.errorText
        if self.code and not text.lstrip().startswith(f"[code:{self.code}]"):
            text = f"[code:{self.code}] {text}"
        data = {
            "type": self.type.value,
            "errorText": text,
        }
        return f"data: {json_dumps(data)}\n\n"


class StreamHeartbeat(StreamBaseResponse):
    """Heartbeat to keep SSE connection alive during long-running operations.

    Uses SSE comment format (: comment) which is ignored by clients but keeps
    the connection alive through proxies and load balancers.
    """

    type: ResponseType = ResponseType.HEARTBEAT
    toolCallId: str | None = Field(
        default=None, description="Tool call ID if heartbeat is for a specific tool"
    )

    def to_sse(self) -> str:
        """Convert to SSE comment format to keep connection alive."""
        return ": heartbeat\n\n"


class StreamCursor(StreamBaseResponse):
    """Deprecated Redis-stream cursor data part.

    Kept so older stored chunks or tests can still be reconstructed, but new
    stream subscriptions no longer emit it. AI SDK resume needs a full replay
    from ``0-0`` so every ``*-delta`` has its matching ``*-start`` event.
    """

    type: ResponseType = ResponseType.CURSOR
    chunkId: str = Field(..., description="Redis Stream message ID (XADD)")

    def to_sse(self) -> str:
        """Emit as an AI SDK v5 data part."""
        data = {
            "type": self.type.value,
            "data": {"chunkId": self.chunkId},
        }
        return f"data: {json.dumps(data)}\n\n"


class StreamStatus(StreamBaseResponse):
    """Transient status notification shown to the user during long operations.

    Emitted when the backend is about to enter a phase that would otherwise
    leave the user staring at a silent "Thinking…" bubble — e.g. the first
    LLM call, the continuation after a tool result, compacting conversation
    context on retry, or activating a fallback model. The frontend reads
    the latest `data-status` part on the current assistant message and uses
    its `message` in place of the generic "Thinking…" copy.
    """

    type: ResponseType = ResponseType.STATUS
    message: str = Field(..., description="Human-readable status message")

    def to_sse(self) -> str:
        """Emit as an AI SDK v5 data part so the client surfaces it as
        `type="data-status"` on `message.parts` instead of dropping it as
        an unknown chunk type."""
        data = {
            "type": self.type.value,
            "data": {"message": self.message},
        }
        return f"data: {json.dumps(data)}\n\n"
