"""Tool for searching agents in the user's library."""

from typing import Any

from pydantic import BaseModel, field_validator

from backend.copilot.model import ChatSession

from .agent_search import (
    lookup_library_agent_by_id,
    search_agents,
    search_library_for_creation,
)
from .base import BaseTool
from .models import ToolResponseBase


class FindLibraryAgentInput(BaseModel):
    query: str = ""
    agent_id: str = ""
    include_graph: bool = False
    for_creation: bool = False
    goal_summary: str = ""

    @field_validator("query", "agent_id", "goal_summary", mode="before")
    @classmethod
    def strip_strings(cls, v: Any) -> str:
        return v.strip() if isinstance(v, str) else (v if v is not None else "")


class FindLibraryAgentTool(BaseTool):
    """Tool for searching agents in the user's library."""

    @property
    def name(self) -> str:
        return "find_library_agent"

    @property
    def description(self) -> str:
        return (
            "Search library agents by name/description, or pass agent_id "
            "(library_agent_id/graph_id) for a direct by-id lookup. "
            "for_creation=true+goal_summary runs the create_agent similarity "
            "check. Omit query to list all; include_graph=true for nodes+links."
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
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Exact library_agent_id/graph_id for a direct lookup "
                        "(no fuzzy fallback). Use when you know the id."
                    ),
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
                "for_creation": {
                    "type": "boolean",
                    "description": "Pre-create similarity check.",
                    "default": False,
                },
                "goal_summary": {
                    "type": "string",
                    "description": "Required when for_creation.",
                },
            },
            # goal_summary is enforced inside the for_creation branch via
            # a NoResultsResponse soft-fail, not as a JSON-schema required.
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        params = FindLibraryAgentInput(**kwargs)
        if params.for_creation:
            # No ``or query`` fallback: the gate only accepts non-empty
            # goal_summary, so falling back to ``query`` would loop the LLM.
            return await search_library_for_creation(
                goal_summary=params.goal_summary,
                session_id=session.session_id,
                user_id=user_id,
            )
        if agent_id := params.agent_id:
            return await lookup_library_agent_by_id(
                agent_id=agent_id,
                session_id=session.session_id,
                user_id=user_id,
                include_graph=params.include_graph,
            )
        return await search_agents(
            query=params.query,
            source="library",
            session_id=session.session_id,
            user_id=user_id,
            include_graph=params.include_graph,
        )
