"""Tool for discovering agents from marketplace."""

from typing import Any

from backend.copilot.model import ChatSession

from .agent_search import search_agents
from .base import BaseTool
from .models import AgentsFoundResponse, ToolResponseBase


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
        result = await search_agents(
            query=query.strip(),
            source="marketplace",
            session_id=session.session_id,
            user_id=user_id,
        )
        if session.expert_id is None or not isinstance(result, AgentsFoundResponse):
            return result
        return result.model_copy(
            update={
                "message": (
                    f"{result.message} This is an expert chat: a marketplace agent "
                    "must be installed with install_expert_workflow "
                    "(username_agent_slug='creator/slug', the agent id below) "
                    "before run_agent can use it."
                )
            }
        )
