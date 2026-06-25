"""CreateAgentTool - Creates agents from pre-built JSON."""

import logging
import uuid
from typing import Any

from pydantic import BaseModel, field_validator

from backend.copilot.model import ChatSession
from backend.copilot.tracking import track_library_check_outcome

from .agent_generator.pipeline import fetch_library_agents, fix_validate_and_save
from .base import BaseTool
from .helpers import require_guide_read, require_library_check
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class CreateAgentInput(BaseModel):
    agent_json: dict[str, Any] | None = None
    save: bool = True
    library_agent_ids: list[str] | None = None
    folder_id: str | None = None
    is_hidden: bool = False
    library_check_ack: bool = False

    @field_validator("folder_id", mode="before")
    @classmethod
    def strip_optional_string(cls, v: Any) -> str | None:
        return v.strip() if isinstance(v, str) else v


class CreateAgentTool(BaseTool):
    """Tool for creating agents from pre-built JSON."""

    @property
    def name(self) -> str:
        return "create_agent"

    @property
    def description(self) -> str:
        return (
            "Create a new agent from JSON (nodes + links). Validates, "
            "auto-fixes, and saves. Requires get_agent_building_guide and "
            "find_library_agent(for_creation=true) first."
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
                    "description": "Agent graph with 'nodes' and 'links' arrays.",
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
                "is_hidden": {
                    "type": "boolean",
                    "description": (
                        "Hide from the user's library listing. "
                        "Use for trigger agents — they appear under "
                        "the parent agent's triggers (auto-derived "
                        "from AgentExecutorBlock usage in the graph)."
                    ),
                    "default": False,
                },
                "library_check_ack": {
                    "type": "boolean",
                    "description": "Bypass library-similarity gate after user declined.",
                    "default": False,
                },
            },
            "required": ["agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        params = CreateAgentInput(**kwargs)
        session_id = session.session_id if session else None

        guide_gate = require_guide_read(session, "create_agent")
        if guide_gate is not None:
            return guide_gate

        if not params.library_check_ack:
            library_gate = require_library_check(session, "create_agent")
            if library_gate is not None:
                return library_gate
        elif user_id and not (session and session.metadata.builder_graph_id):
            # Track the LLM-driven gate bypass so we can measure how often
            # users were shown matches but chose to build new anyway.
            track_library_check_outcome(
                user_id=user_id, session_id=session_id, outcome="bypassed_ack"
            )

        if not params.agent_json:
            return ErrorResponse(
                message=(
                    "Please provide agent_json with the complete agent graph. "
                    "Use find_block to discover blocks, then generate the JSON."
                ),
                error="missing_agent_json",
                session_id=session_id,
            )

        library_agent_ids = params.library_agent_ids or []

        nodes = params.agent_json.get("nodes", [])
        if not nodes:
            return ErrorResponse(
                message=(
                    "The agent JSON has no nodes. " "An agent needs at least one block."
                ),
                error="empty_agent",
                session_id=session_id,
            )

        # Ensure top-level fields
        if "id" not in params.agent_json:
            params.agent_json["id"] = str(uuid.uuid4())
        if "version" not in params.agent_json:
            params.agent_json["version"] = 1
        if "is_active" not in params.agent_json:
            params.agent_json["is_active"] = True

        # Fetch library agents for AgentExecutorBlock validation
        library_agents = await fetch_library_agents(user_id, library_agent_ids)

        return await fix_validate_and_save(
            params.agent_json,
            user_id=user_id,
            session_id=session_id,
            save=params.save,
            is_update=False,
            default_name="Generated Agent",
            library_agents=library_agents,
            folder_id=params.folder_id,
            is_hidden=params.is_hidden,
        )
