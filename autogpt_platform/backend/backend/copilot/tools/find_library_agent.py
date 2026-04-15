"""Tool for searching agents in the user's library."""

from typing import Any

from backend.copilot.model import ChatSession

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
            "Search user's library agents. Returns graph_id, schemas for sub-agent composition. "
            "Omit query to list all. Set include_graph=true to also fetch the full "
            "graph structure (nodes + links) for debugging or editing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search by name/description. Omit to list all.",
                },
                "include_graph": {
                    "type": "boolean",
                    "description": (
                        "When true, includes the full graph structure "
                        "(nodes + links) for each found agent. "
                        "Use when you need to inspect, debug, or edit an agent."
                    ),
                    "default": False,
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        include_graph: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        return await search_agents(
            query=query.strip(),
            source="library",
            session_id=session.session_id,
            user_id=user_id,
            include_graph=include_graph,
        )
