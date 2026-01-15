import logging
from typing import Any

from backend.api.features.chat.tools.base  import BaseTool, ToolResponseBase
from backend.api.features.chat.model import ChatSession 
logger = logging.getLogger(__name__)


class FindBlockTool(BaseTool):
    """Tool for searching available blocks."""

    @property
    def name(self) -> str:
        return "find_block"

    @property
    def description(self) -> str:
        return (
            "Search for available blocks by name or description. "
            "Blocks are reusable components that perform specific tasks like "
            "sending emails, making API calls, processing text, etc. "
            "Use this to find blocks that can be executed directly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query to find blocks by name or description. "
                        "Use keywords like 'email', 'http', 'text', 'ai', etc."
                    ),
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for blocks matching the query.

        Args:
            user_id: User ID (required)
            session: Chat session
            query: Search query

        Returns:
            BlockListResponse: List of matching blocks
            NoResultsResponse: No blocks found
            ErrorResponse: Error message
        """
