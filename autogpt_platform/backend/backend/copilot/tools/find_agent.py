"""Tool for discovering agents from marketplace."""

from typing import Any

from pydantic import BaseModel, field_validator

from backend.copilot.model import ChatSession

from .agent_search import search_agents
from .base import BaseTool
from .models import ToolResponseBase


class FindAgentInput(BaseModel):
    """Input parameters for the find_agent tool."""

    query: str = ""

    @field_validator("query", mode="before")
    @classmethod
    def strip_string(cls, v: Any) -> str:
        """Strip whitespace from query."""
        return v.strip() if isinstance(v, str) else (v if v is not None else "")


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
        self, user_id: str | None, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        """Search marketplace for agents matching the query."""
        params = FindAgentInput(**kwargs)
        return await search_agents(
            query=params.query,
            source="marketplace",
            session_id=session.session_id,
            user_id=user_id,
        )
