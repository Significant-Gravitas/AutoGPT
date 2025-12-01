"""Base classes and shared utilities for chat tools."""

import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.response_model import StreamToolExecutionResult

from .models import ErrorResponse, NeedLoginResponse, ToolResponseBase

logger = logging.getLogger(__name__)


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
    ) -> StreamToolExecutionResult:
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
            return StreamToolExecutionResult(
                tool_id=tool_call_id,
                tool_name=self.name,
                result=NeedLoginResponse(
                    message=f"Please sign in to use {self.name}",
                    session_id=session.session_id,
                ).model_dump_json(),
                success=False,
            )

        try:
            result = await self._execute(user_id, session, **kwargs)
            return StreamToolExecutionResult(
                tool_id=tool_call_id,
                tool_name=self.name,
                result=result.model_dump_json(),
            )
        except Exception as e:
            logger.error(f"Error in {self.name}: {e}", exc_info=True)
            return StreamToolExecutionResult(
                tool_id=tool_call_id,
                tool_name=self.name,
                result=ErrorResponse(
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
