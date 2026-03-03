"""CustomizeAgentTool - Customizes marketplace/template agents."""

import logging
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
            "Customize a marketplace or template agent. Pass `agent_json` "
            "with the complete customized agent JSON. The tool validates, "
            "auto-fixes, and saves.\n\n"
            "Use agent_id in format 'creator/slug' to specify the marketplace agent."
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
                        "The marketplace agent ID in format 'creator/slug' "
                        "(e.g., 'autogpt/newsletter-writer')."
                    ),
                },
                "agent_json": {
                    "type": "object",
                    "description": (
                        "Complete customized agent JSON to validate and save."
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
                        "Whether to save the customized agent. Default is true."
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
                message="Please provide the marketplace agent ID (e.g., 'creator/agent-name').",
                error="missing_agent_id",
                session_id=session_id,
            )

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

        nodes = agent_json.get("nodes", [])
        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes.",
                error="empty_agent",
                session_id=session_id,
            )

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
        )
