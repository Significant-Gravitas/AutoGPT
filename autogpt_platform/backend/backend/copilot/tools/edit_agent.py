"""EditAgentTool - Edits existing agents using pre-built JSON."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator import get_agent_as_json
from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
from .helpers import require_guide_read
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class EditAgentTool(BaseTool):
    """Tool for editing existing agents using pre-built JSON."""

    @property
    def name(self) -> str:
        return "edit_agent"

    @property
    def description(self) -> str:
        return (
            "Edit an existing agent. Validates, auto-fixes, and saves. "
            "Requires get_agent_building_guide first (refuses otherwise)."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "string",
                    "description": "Graph ID or library agent ID to edit.",
                },
                "agent_json": {
                    "type": "object",
                    "description": "Updated agent JSON with nodes and links.",
                },
                "library_agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Library agent IDs as building blocks.",
                },
                "save": {
                    "type": "boolean",
                    "description": "Save changes (default: true). False for preview.",
                    "default": True,
                },
            },
            "required": ["agent_id", "agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_id: str = "",
        agent_json: dict[str, Any] | None = None,
        save: bool = True,
        library_agent_ids: list[str] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        agent_id = agent_id.strip()
        if library_agent_ids is None:
            library_agent_ids = []
        session_id = session.session_id if session else None

        # Builder-bound sessions are locked to a specific graph: default
        # missing agent_id to the bound graph, and reject any other id so
        # the assistant cannot accidentally mutate a different agent.
        builder_graph_id = session.metadata.builder_graph_id if session else None
        if builder_graph_id:
            if not agent_id:
                agent_id = builder_graph_id
            elif agent_id != builder_graph_id:
                return ErrorResponse(
                    message=(
                        "This chat is bound to the builder's current agent. "
                        "Editing a different agent is not allowed here — "
                        "open that agent in the builder instead."
                    ),
                    error="builder_session_graph_mismatch",
                    session_id=session_id,
                )

        guide_gate = require_guide_read(session, "edit_agent")
        if guide_gate is not None:
            return guide_gate

        if not agent_id:
            return ErrorResponse(
                message="Please provide the agent ID to edit.",
                error="missing_agent_id",
                session_id=session_id,
            )

        if not agent_json:
            return ErrorResponse(
                message=(
                    "Please provide agent_json with the complete updated agent graph."
                ),
                error="missing_agent_json",
                session_id=session_id,
            )

        nodes = agent_json.get("nodes", [])
        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes.",
                error="empty_agent",
                session_id=session_id,
            )

        # Preserve original agent's ID
        current_agent = await get_agent_as_json(agent_id, user_id)
        if current_agent is None:
            return ErrorResponse(
                message=f"Could not find agent with ID '{agent_id}' in your library.",
                error="agent_not_found",
                session_id=session_id,
            )

        agent_json["id"] = current_agent.get("id", agent_id)
        agent_json["version"] = current_agent.get("version", 1)
        agent_json.setdefault("is_active", True)

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=True,
            default_name="Updated Agent",
            library_agents=library_agents,
        )
