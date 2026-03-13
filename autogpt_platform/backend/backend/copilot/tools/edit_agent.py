"""EditAgentTool - Edits existing agents using pre-built JSON."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator import get_agent_as_json
from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
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
            "Edit an existing agent. Pass `agent_json` with the complete "
            "updated agent JSON you generated. The tool validates, auto-fixes, "
            "and saves.\n\n"
            "IMPORTANT: Before calling this tool, if the changes involve adding new "
            "functionality, search for relevant existing agents using find_library_agent "
            "that could be used as building blocks."
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
                    "description": (
                        "The ID of the agent to edit. "
                        "Can be a graph ID or library agent ID."
                    ),
                },
                "agent_json": {
                    "type": "object",
                    "description": (
                        "Complete updated agent JSON to validate and save. "
                        "Must contain 'nodes' and 'links'. "
                        "Include 'name' and/or 'description' if they need "
                        "to be updated."
                    ),
                },
                "library_agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of library agent IDs to use as building blocks for the changes."
                    ),
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "Whether to save the changes. "
                        "Default is true. Set to false for preview only."
                    ),
                    "default": True,
                },
            },
            "required": ["agent_id", "agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        agent_id = kwargs.get("agent_id", "").strip()
        agent_json: dict[str, Any] | None = kwargs.get("agent_json")
        session_id = session.session_id if session else None

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

        save = kwargs.get("save", True)
        library_agent_ids = kwargs.get("library_agent_ids", [])

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
