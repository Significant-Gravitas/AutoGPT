"""Tool for searching agents in the user's library."""

from typing import Any

from backend.copilot.model import ChatSession

from .agent_search import search_agents, search_library_for_creation
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
            "Search user's library agents. for_creation=true+goal_summary "
            "runs the similarity check required by create_agent. Omit query "
            "to list all; include_graph=true for nodes+links."
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
            # ``goal_summary`` is deliberately NOT listed as a required
            # property even though it is required in for_creation mode:
            # ``search_library_for_creation`` returns a ``NoResultsResponse``
            # with a recovery message instead of a hard validation error
            # when it's missing, so the LLM can retry without a tool-call
            # error surfacing in chat.
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
        for_creation: bool = False,
        goal_summary: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if for_creation:
            # No ``or query`` fallback: the create_agent gate's validator
            # (_has_for_creation_args in helpers.py) only accepts a non-empty
            # ``goal_summary``. Letting ``query`` satisfy the search but not
            # the gate would force the LLM into a retry loop. Empty
            # goal_summary still soft-fails to NoResultsResponse with a
            # retry hint, so this is a no-op for well-formed callers.
            return await search_library_for_creation(
                goal_summary=goal_summary,
                session_id=session.session_id,
                user_id=user_id,
            )
        return await search_agents(
            query=query.strip(),
            source="library",
            session_id=session.session_id,
            user_id=user_id,
            include_graph=include_graph,
        )
