"""
Response models for Vercel AI SDK UI Stream Protocol.

This module implements the AI SDK UI Stream Protocol (v1) for streaming chat responses.
See: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Types of streaming responses following AI SDK protocol."""

    # Message lifecycle
    START = "start"
    FINISH = "finish"

    # Text streaming
    TEXT_START = "text-start"
    TEXT_DELTA = "text-delta"
    TEXT_END = "text-end"

    # Tool interaction
    TOOL_INPUT_START = "tool-input-start"
    TOOL_INPUT_AVAILABLE = "tool-input-available"
    TOOL_OUTPUT_AVAILABLE = "tool-output-available"

    # Other
    ERROR = "error"
    USAGE = "usage"
    HEARTBEAT = "heartbeat"


class StreamBaseResponse(BaseModel):
    """Base response model for all streaming responses."""

    type: ResponseType

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


# ========== Message Lifecycle ==========


class StreamStart(StreamBaseResponse):
    """Start of a new message."""

    type: ResponseType = ResponseType.START
    messageId: str = Field(..., description="Unique message ID")
    taskId: str | None = Field(
        default=None,
        description="Task ID for SSE reconnection. Clients can reconnect using GET /tasks/{taskId}/stream",
    )


class StreamFinish(StreamBaseResponse):
    """End of message/stream."""

    type: ResponseType = ResponseType.FINISH


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


class StreamToolOutputAvailable(StreamBaseResponse):
    """Tool execution result."""

    type: ResponseType = ResponseType.TOOL_OUTPUT_AVAILABLE
    toolCallId: str = Field(..., description="Tool call ID this responds to")
    output: str | dict[str, Any] = Field(..., description="Tool execution output")
    # Additional fields for internal use (not part of AI SDK spec but useful)
    toolName: str | None = Field(
        default=None, description="Name of the tool that was executed"
    )
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )


# ========== Other ==========


class StreamUsage(StreamBaseResponse):
    """Token usage statistics."""

    type: ResponseType = ResponseType.USAGE
    promptTokens: int = Field(..., description="Number of prompt tokens")
    completionTokens: int = Field(..., description="Number of completion tokens")
    totalTokens: int = Field(..., description="Total number of tokens")


class StreamError(StreamBaseResponse):
    """Error response."""

    type: ResponseType = ResponseType.ERROR
    errorText: str = Field(..., description="Error message text")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )


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
