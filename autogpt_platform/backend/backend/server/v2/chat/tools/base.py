"""Base classes and shared utilities for chat tools."""

from typing import Any

from openai.types.chat import ChatCompletionToolParam

from .models import ErrorResponse, NeedLoginResponse, ToolResponseBase


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
        session_id: str,
        **kwargs,
    ) -> ToolResponseBase:
        """Execute the tool with authentication check.

        Args:
            user_id: User ID (may be anonymous like "anon_123")
            session_id: Chat session ID
            **kwargs: Tool-specific parameters

        Returns:
            Pydantic response object

        """
        # Check authentication if required
        if self.requires_auth and (not user_id or user_id.startswith("anon_")):
            return NeedLoginResponse(
                message=f"Please sign in to use {self.name}",
                session_id=session_id,
            )

        try:
            return await self._execute(user_id, session_id, **kwargs)
        except Exception as e:
            # Log the error internally but return a safe message
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error in {self.name}: {e}", exc_info=True)

            return ErrorResponse(
                message=f"An error occurred while executing {self.name}",
                error=str(e),
                session_id=session_id,
            )

    async def _execute(
        self,
        user_id: str | None,
        session_id: str,
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
