"""Base classes and shared utilities for chat tools."""

import asyncio
import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.response_model import StreamToolOutputAvailable

from .models import ErrorResponse, NeedLoginResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Default per-tool execution timeout (seconds).
# Long-running tools (is_long_running=True) get a longer default.
DEFAULT_TOOL_TIMEOUT = 120
DEFAULT_LONG_RUNNING_TOOL_TIMEOUT = 600


class BaseTool:
    """Base class for all chat tools."""

    @property
    def name(self) -> str:
        """Tool name for OpenAI function calling."""
        raise NotImplementedError

    @property
    def description(self) -> str:
        """Tool description for OpenAI."""
        raise NotImplementedError

    @property
    def parameters(self) -> dict[str, Any]:
        """Tool parameters schema for OpenAI."""
        raise NotImplementedError

    @property
    def requires_auth(self) -> bool:
        """Whether this tool requires authentication."""
        return False

    @property
    def is_long_running(self) -> bool:
        """Whether this tool is long-running and should execute in background.

        Long-running tools (like agent generation) are executed via background
        tasks to survive SSE disconnections. The result is persisted to chat
        history and visible when the user refreshes.
        """
        return False

    @property
    def timeout_seconds(self) -> int:
        """Maximum execution time in seconds before the tool is cancelled."""
        return (
            DEFAULT_LONG_RUNNING_TOOL_TIMEOUT
            if self.is_long_running
            else DEFAULT_TOOL_TIMEOUT
        )

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Convert to OpenAI tool format."""
        return ChatCompletionToolParam(
            type="function",
            function={
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        )

    async def execute(
        self,
        user_id: str | None,
        session: ChatSession,
        tool_call_id: str,
        **kwargs,
    ) -> StreamToolOutputAvailable:
        """Execute the tool with authentication check.

        Args:
            user_id: User ID (may be anonymous like "anon_123")
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        if self.requires_auth and not user_id:
            logger.error(
                f"Attempted tool call for {self.name} but user not authenticated"
            )
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=NeedLoginResponse(
                    message=f"Please sign in to use {self.name}",
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

        try:
            result = await asyncio.wait_for(
                self._execute(user_id, session, **kwargs),
                timeout=self.timeout_seconds,
            )
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=result.model_dump_json(),
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Tool {self.name} timed out after {self.timeout_seconds}s"
            )
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=ErrorResponse(
                    message=f"{self.name} timed out after {self.timeout_seconds}s",
                    error="timeout",
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=ErrorResponse(
                    message=f"An error occurred while executing {self.name}",
                    error=str(e),
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Internal execution logic to be implemented by subclasses.

        Args:
            user_id: User ID (authenticated or anonymous)
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        raise NotImplementedError
