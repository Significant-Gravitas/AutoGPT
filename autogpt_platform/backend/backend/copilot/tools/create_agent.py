"""CreateAgentTool - Creates agents from pre-built JSON."""

import logging
import uuid
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class CreateAgentTool(BaseTool):
    """Tool for creating agents from pre-built JSON."""

    @property
    def name(self) -> str:
        return "create_agent"

    @property
    def description(self) -> str:
        return (
            "Create a new agent workflow. Pass `agent_json` with the complete "
            "agent graph JSON you generated using block schemas from find_block. "
            "The tool validates, auto-fixes, and saves.\n\n"
            "IMPORTANT: Before calling this tool, search for relevant existing agents "
            "using find_library_agent that could be used as building blocks. "
            "Pass their IDs in the library_agent_ids parameter."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_json": {
                    "type": "object",
                    "description": (
                        "The agent JSON to validate and save. "
                        "Must contain 'nodes' and 'links' arrays, and optionally "
                        "'name' and 'description'."
                    ),
                },
                "library_agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of library agent IDs to use as building blocks."
                    ),
                },
                "save": {
                    "type": "boolean",
                    "description": (
                        "Whether to save the agent. Default is true. "
                        "Set to false for preview only."
                    ),
                    "default": True,
                },
                "folder_id": {
                    "type": "string",
                    "description": (
                        "Optional folder ID to save the agent into. "
                        "If not provided, the agent is saved at root level. "
                        "Use list_folders to find available folders."
                    ),
                },
            },
            "required": ["agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        agent_json: dict[str, Any] | None = kwargs.get("agent_json")
        session_id = session.session_id if session else None

        if not agent_json:
            return ErrorResponse(
                message=(
                    "Please provide agent_json with the complete agent graph. "
                    "Use find_block to discover blocks, then generate the JSON."
                ),
                error="missing_agent_json",
                session_id=session_id,
            )

        save = kwargs.get("save", True)
        library_agent_ids = kwargs.get("library_agent_ids", [])
        folder_id: str | None = kwargs.get("folder_id")

        nodes = agent_json.get("nodes", [])
        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes. An agent needs at least one block.",
                error="empty_agent",
                session_id=session_id,
            )

        # Ensure top-level fields
        if "id" not in agent_json:
            agent_json["id"] = str(uuid.uuid4())
        if "version" not in agent_json:
            agent_json["version"] = 1
        if "is_active" not in agent_json:
            agent_json["is_active"] = True

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=False,
            default_name="Generated Agent",
            library_agents=library_agents,
            folder_id=folder_id,
        )
