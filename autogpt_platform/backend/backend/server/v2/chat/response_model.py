from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Types of streaming responses."""

    TEXT_CHUNK = "text_chunk"
    TEXT_ENDED = "text_ended"
    TOOL_CALL = "tool_call"
    TOOL_CALL_START = "tool_call_start"
    TOOL_RESPONSE = "tool_response"
    ERROR = "error"
    USAGE = "usage"
    STREAM_END = "stream_end"


class StreamBaseResponse(BaseModel):
    """Base response model for all streaming responses."""

    type: ResponseType
    timestamp: str | None = None

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class StreamTextChunk(StreamBaseResponse):
    """Streaming text content from the assistant."""

    type: ResponseType = ResponseType.TEXT_CHUNK
    content: str = Field(..., description="Text content chunk")


class StreamToolCallStart(StreamBaseResponse):
    """Tool call started notification."""

    type: ResponseType = ResponseType.TOOL_CALL_START
    tool_id: str = Field(..., description="Unique tool call ID")


class StreamToolCall(StreamBaseResponse):
    """Tool invocation notification."""

    type: ResponseType = ResponseType.TOOL_CALL
    tool_id: str = Field(..., description="Unique tool call ID")
    tool_name: str = Field(..., description="Name of the tool being called")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )


class StreamToolExecutionResult(StreamBaseResponse):
    """Tool execution result."""

    type: ResponseType = ResponseType.TOOL_RESPONSE
    tool_id: str = Field(..., description="Tool call ID this responds to")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: str | dict[str, Any] = Field(..., description="Tool execution result")
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )


class StreamUsage(StreamBaseResponse):
    """Token usage statistics."""

    type: ResponseType = ResponseType.USAGE
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StreamError(StreamBaseResponse):
    """Error response."""

    type: ResponseType = ResponseType.ERROR
    message: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )


class StreamTextEnded(StreamBaseResponse):
    """Text streaming completed marker."""

    type: ResponseType = ResponseType.TEXT_ENDED


class StreamEnd(StreamBaseResponse):
    """End of stream marker."""

    type: ResponseType = ResponseType.STREAM_END
    summary: dict[str, Any] | None = Field(
        default=None, description="Stream summary statistics"
    )
