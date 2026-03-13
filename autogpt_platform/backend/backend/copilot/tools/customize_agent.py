"""CustomizeAgentTool - Customizes marketplace/template agents."""

import logging
import uuid
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class CustomizeAgentTool(BaseTool):
    """Tool for customizing marketplace/template agents."""

    @property
    def name(self) -> str:
        return "customize_agent"

    @property
    def description(self) -> str:
        return (
            "Customize a marketplace/template agent. Validates, auto-fixes, and saves."
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
                    "description": "Customized agent JSON with nodes and links.",
                },
                "library_agent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Library agent IDs as building blocks.",
                },
                "save": {
                    "type": "boolean",
                    "description": "Save the agent (default: true). False for preview.",
                    "default": True,
                },
                "folder_id": {
                    "type": "string",
                    "description": "Folder ID to save into (default: root).",
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
                    "Please provide agent_json with the complete customized agent graph."
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
                message="The agent JSON has no nodes.",
                error="empty_agent",
                session_id=session_id,
            )

        # Ensure top-level fields before the fixer pipeline
        if "id" not in agent_json:
            agent_json["id"] = str(uuid.uuid4())
        agent_json.setdefault("version", 1)
        agent_json.setdefault("is_active", True)

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            agent_json,
            user_id=user_id,
            session_id=session_id,
            save=save,
            is_update=False,
            default_name="Customized Agent",
            library_agents=library_agents,
            folder_id=folder_id,
        )
