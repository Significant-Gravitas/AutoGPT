"""Tool for discovering agents from marketplace."""

from typing import Any

from backend.copilot.model import ChatSession

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
        return "Search marketplace agents by capability, or look up by slug ('username/agent-name')."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords, or 'username/agent-name' for direct slug lookup.",
                },
            },
            "required": ["query"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        """Search marketplace for agents matching the query."""
        return await search_agents(
            query=query.strip(),
            source="marketplace",
            session_id=session.session_id,
            user_id=user_id,
        )
