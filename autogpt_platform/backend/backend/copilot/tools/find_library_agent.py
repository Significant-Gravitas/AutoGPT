"""Tool for searching agents in the user's library."""

from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator import get_agent_as_json
from .agent_json_input import write_agent_json_to_workspace
from .agent_search import (
    lookup_library_agent_by_id,
    search_agents,
    search_library_for_creation,
)
from .base import BaseTool
from .models import AgentsFoundResponse, ErrorResponse, ToolResponseBase


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
                "write_graph_to": {
                    "type": "string",
                    "description": (
                        "Workspace filename (no directories) to write the "
                        "agent's full graph JSON to (pretty-printed, "
                        "overwrites) instead of returning it inline. Requires "
                        "agent_id. The response includes an @@agptfile ref to "
                        "pass to edit_agent — avoids pulling a large graph "
                        "through context when editing an existing agent."
                    ),
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
        query: str = "",
        agent_id: str = "",
        include_graph: bool = False,
        write_graph_to: str = "",
        for_creation: bool = False,
        goal_summary: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        if for_creation:
            # No ``or query`` fallback: the gate only accepts non-empty
            # goal_summary, so falling back to ``query`` would loop the LLM.
            return await search_library_for_creation(
                goal_summary=goal_summary,
                session_id=session.session_id,
                user_id=user_id,
            )
        write_graph_to = write_graph_to.strip()
        if write_graph_to and not agent_id.strip():
            return ErrorResponse(
                message=(
                    "write_graph_to requires agent_id — pass the library agent "
                    "or graph id whose graph should be written to the file."
                ),
                error="missing_agent_id",
                session_id=session.session_id,
            )
        if agent_id := agent_id.strip():
            result = await lookup_library_agent_by_id(
                agent_id=agent_id,
                session_id=session.session_id,
                user_id=user_id,
                include_graph=include_graph and not write_graph_to,
            )
            if write_graph_to and isinstance(result, AgentsFoundResponse):
                note = await _write_graph_note(
                    agent_id, write_graph_to, user_id, session.session_id
                )
                result.message = f"{result.message}\n\n{note.strip()}"
            return result
        return await search_agents(
            query=query.strip(),
            source="library",
            session_id=session.session_id,
            user_id=user_id,
            include_graph=include_graph,
        )


async def _write_graph_note(
    agent_id: str, write_to: str, user_id: str | None, session_id: str | None
) -> str:
    """Write the agent's graph to a workspace file; return the message note.

    The note either carries the @@agptfile ref to pass to edit_agent, or
    explains why the write failed and what to do instead. Never raises —
    the agent lookup already succeeded, so a graph-write hiccup must degrade
    to a note on that result, not replace it with a generic tool error.
    """
    try:
        agent_json = await get_agent_as_json(agent_id, user_id)
    except Exception:
        agent_json = None
    if agent_json is None:
        return (
            "NOTE: could not load the agent's graph to write it to a file; "
            "retry with include_graph=true to inspect it inline."
        )
    _ref, note = await write_agent_json_to_workspace(
        agent_json,
        write_to,
        user_id,
        session_id,
        label="Agent graph",
        pass_to="edit_agent / validate_agent_graph",
        fallback_note="retry with include_graph=true to inspect the graph inline.",
    )
    return note
