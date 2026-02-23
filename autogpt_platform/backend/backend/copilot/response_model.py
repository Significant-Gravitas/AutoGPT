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
    # Keep these for internal backend use
    toolName: str | None = Field(
        default=None, description="Name of the tool that was executed"
    )
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )

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

    def to_sse(self) -> str:
        """Convert to SSE format, only emitting fields required by AI SDK protocol.

        The AI SDK uses z.strictObject({type, errorText}) which rejects
        any extra fields like `code` or `details`.
        """
        data = {
            "type": self.type.value,
            "errorText": self.errorText,
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
