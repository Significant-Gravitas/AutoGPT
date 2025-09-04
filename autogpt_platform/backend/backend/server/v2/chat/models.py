"""Response models for chat streaming."""

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


class BaseResponse(BaseModel):
    """Base response model for all streaming responses."""

    type: ResponseType
    timestamp: str | None = None


class TextChunk(BaseResponse):
    """Streaming text content from the assistant."""

    type: ResponseType = ResponseType.TEXT_CHUNK
    content: str = Field(..., description="Text content chunk")

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class ToolCall(BaseResponse):
    """Tool invocation notification."""

    type: ResponseType = ResponseType.TOOL_CALL
    tool_id: str = Field(..., description="Unique tool call ID")
    tool_name: str = Field(..., description="Name of the tool being called")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class ToolResponse(BaseResponse):
    """Tool execution result."""

    type: ResponseType = ResponseType.TOOL_RESPONSE
    tool_id: str = Field(..., description="Tool call ID this responds to")
    tool_name: str = Field(..., description="Name of the tool that was executed")
    result: str | dict[str, Any] = Field(..., description="Tool execution result")
    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class LoginNeeded(BaseResponse):
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

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class Error(BaseResponse):
    """Error response."""

    type: ResponseType = ResponseType.ERROR
    message: str = Field(..., description="Error message")
    code: str | None = Field(default=None, description="Error code")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


class StreamEnd(BaseResponse):
    """End of stream marker."""

    type: ResponseType = ResponseType.STREAM_END
    summary: dict[str, Any] | None = Field(
        default=None, description="Stream summary statistics"
    )

    def to_sse(self) -> str:
        """Convert to SSE format."""
        return f"data: {self.model_dump_json()}\n\n"


# Additional model for agent carousel data
class AgentCarouselData(BaseModel):
    """Data structure for agent carousel display."""

    type: str = "agent_carousel"
    query: str
    count: int
    agents: list[dict[str, Any]]
