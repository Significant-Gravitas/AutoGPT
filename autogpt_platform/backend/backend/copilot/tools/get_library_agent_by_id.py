"""Tool for fetching a single library agent by its exact id."""

from typing import Any

from backend.copilot.model import ChatSession

from .agent_search import get_library_agent_by_id
from .base import BaseTool
from .models import ToolResponseBase


class GetLibraryAgentByIdTool(BaseTool):
    """Fetch one library agent by exact id — a direct lookup, not a search."""

    @property
    def name(self) -> str:
        return "get_library_agent_by_id"

    @property
    def description(self) -> str:
        return (
            "Fetch ONE library agent by exact id (library_agent_id or "
            "graph_id) — direct lookup, NOT a search. Use when you know the "
            "id (e.g. an id given in the prompt). include_graph=true adds "
            "nodes+links. For name discovery use find_library_agent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Exact library_agent_id or graph_id.",
                },
                "include_graph": {
                    "type": "boolean",
                    "description": "Include the full graph (nodes + links).",
                    "default": False,
                },
            },
            "required": ["agent_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_id: str = "",
        include_graph: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        return await get_library_agent_by_id(
            agent_id=agent_id,
            session_id=session.session_id,
            user_id=user_id,
            include_graph=include_graph,
        )
