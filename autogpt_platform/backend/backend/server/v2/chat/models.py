from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Types of streaming responses."""

    TEXT_CHUNK = "text_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"
    LOGIN_NEEDED = "login_needed"
    ERROR = "error"
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


class StreamToolCall(StreamBaseResponse):
    """Tool invocation notification."""

    type: ResponseType = ResponseType.TOOL_CALL
    tool_id: str = Field(..., description="Unique tool call ID")
    tool_name: str = Field(..., description="Name of the tool being called")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )


class StreamToolResponse(StreamBaseResponse):
    """Tool execution result."""

    type: ResponseType = ResponseType.TOOL_RESPONSE
    tool_id: str = Field(..., description="Tool call ID this responds to")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: str | dict[str, Any] = Field(..., description="Tool execution result")
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )


class StreamLoginNeeded(StreamBaseResponse):
    """Authentication required notification."""

    type: ResponseType = ResponseType.LOGIN_NEEDED
    message: str = Field(..., description="Message explaining why login is needed")
    session_id: str = Field(..., description="Current session ID to preserve")
    agent_info: dict[str, Any] | None = Field(
        default=None, description="Agent context if applicable"
    )
    required_action: str = Field(
        default="login", description="Required action (login/signup)"
    )


class StreamError(StreamBaseResponse):
    """Error response."""

    type: ResponseType = ResponseType.ERROR
    message: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )

class StreamEnd(StreamBaseResponse):
    """End of stream marker."""

    type: ResponseType = ResponseType.STREAM_END
    summary: dict[str, Any] | None = Field(
        default=None, description="Stream summary statistics"
    )