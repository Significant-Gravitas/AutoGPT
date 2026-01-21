"""Tool for searching agents in the user's library."""

from typing import Any

from langfuse import observe

from backend.api.features.chat.model import ChatSession

from .agent_search import search_agents
from .base import BaseTool
from .models import ToolResponseBase


class FindLibraryAgentTool(BaseTool):
    """Tool for searching agents in the user's library."""

    @property
    def name(self) -> str:
        return "find_library_agent"

    @property
    def description(self) -> str:
        return (
            "Search for agents in the user's library. Use this to find agents "
            "the user has already added to their library, including agents they "
            "created or added from the marketplace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find agents by name or description.",
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @observe(as_type="tool", name="find_library_agent")
    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs
    ) -> ToolResponseBase:
        return await search_agents(
            query=kwargs.get("query", "").strip(),
            source="library",
            session_id=session.session_id,
            user_id=user_id,
        )
