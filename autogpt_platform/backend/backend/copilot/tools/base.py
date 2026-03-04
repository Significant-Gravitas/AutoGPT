"""Base classes and shared utilities for chat tools."""

import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.data.db_accessors import workspace_db
from backend.util.workspace import WorkspaceManager

from .models import ErrorResponse, NeedLoginResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Persist full tool output to workspace when it exceeds this threshold.
# Must be below _MAX_TOOL_OUTPUT_SIZE (100K) in response_model.py so we
# capture the data before model_post_init middle-out truncation discards it.
_LARGE_OUTPUT_THRESHOLD = 80_000

# Number of head characters included in the preview that replaces the output.
_PREVIEW_HEAD_CHARS = 50_000


async def _persist_and_summarize(
    raw_output: str,
    user_id: str,
    session_id: str,
    tool_call_id: str,
) -> str:
    """Persist full output to workspace and return a head preview with retrieval instructions.

    On failure, returns the original ``raw_output`` unchanged so that the
    existing ``model_post_init`` middle-out truncation handles it as before.
    """
    file_path = f"tool-outputs/{tool_call_id}.json"
    try:
        workspace = await workspace_db().get_or_create_workspace(user_id)
        manager = WorkspaceManager(user_id, workspace.id, session_id)
        await manager.write_file(
            content=raw_output.encode("utf-8"),
            filename=f"{tool_call_id}.json",
            path=file_path,
            mime_type="application/json",
            overwrite=True,
        )
    except Exception:
        logger.warning(
            "Failed to persist large tool output for %s",
            tool_call_id,
            exc_info=True,
        )
        return raw_output  # fall back to normal truncation

    total = len(raw_output)
    preview = raw_output[:_PREVIEW_HEAD_CHARS]
    remaining = total - len(preview)
    return (
        f'<tool-output-truncated total_chars={total} path="{file_path}">\n'
        f"{preview}\n"
        f"... [{remaining:,} more characters]\n\n"
        f"Full output saved to workspace. Use read_workspace_file("
        f'path="{file_path}", offset={len(preview)}, length=50000) '
        f"to read the next section.\n"
        f"</tool-output-truncated>"
    )


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
    def is_available(self) -> bool:
        """Whether this tool is available in the current environment.

        Override to check required env vars, binaries, or other dependencies.
        Unavailable tools are excluded from the LLM tool list so the model is
        never offered an option that will immediately fail.
        """
        return True

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
            result = await self._execute(user_id, session, **kwargs)
            raw_output = result.model_dump_json()

            if (
                len(raw_output) > _LARGE_OUTPUT_THRESHOLD
                and user_id
                and session.session_id
            ):
                raw_output = await _persist_and_summarize(
                    raw_output, user_id, session.session_id, tool_call_id
                )

            return StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=self.name,
                output=raw_output,
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
