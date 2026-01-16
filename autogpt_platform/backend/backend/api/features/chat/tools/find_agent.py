"""Tool for discovering agents from marketplace."""

from typing import Any

from backend.api.features.chat.model import ChatSession

from .agent_search import search_agents
from .base import BaseTool
from .models import ToolResponseBase


class FindAgentTool(BaseTool):
    """Tool for discovering agents from the marketplace."""

    @property
    def name(self) -> str:
        return "find_agent"

    @property
    def description(self) -> str:
        return (
            "Discover agents from the marketplace based on capabilities and user needs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what the user wants to accomplish. Use single keywords for best results.",
                },
            },
            "required": ["query"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        return await search_agents(
            query=kwargs.get("query", "").strip(),
            source="marketplace",
            session_id=session.session_id,
            user_id=user_id,
        )
