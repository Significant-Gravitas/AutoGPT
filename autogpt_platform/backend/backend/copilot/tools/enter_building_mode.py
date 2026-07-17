"""EnterAgentBuildingModeTool - switches the session into building mode.

Building mode moves the agent-building guide into the (prompt-cached) system
prompt, where it survives context compaction — instead of living in the
conversation as a ~9K-token tool result that every compaction evicts and the
model then re-fetches. On the SDK path, calling this tool triggers an
in-turn restart with the upgraded system prompt (see
``_BuildingModeRestart`` in ``backend.copilot.sdk.service``); the tool call
itself, persisted in message history, is the durable mode signal for all
later turns.
"""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class BuildingModeResponse(ToolResponseBase):
    """Response for enter_agent_building_mode."""

    type: ResponseType = ResponseType.AGENT_BUILDER_GUIDE
    content: str = ""


class EnterAgentBuildingModeTool(BaseTool):
    """Switches the session into agent-building mode."""

    @property
    def name(self) -> str:
        return "enter_agent_building_mode"

    @property
    def description(self) -> str:
        return (
            "Enter agent-building mode: loads the building guide into "
            "your system prompt, compaction-proof. Call BEFORE "
            "designing or building an agent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if session is not None and session.guide_in_system_prompt:
            return BuildingModeResponse(
                message="Building mode is already active.",
                content=(
                    "Building mode is already active — the agent-building "
                    "guide is in your system prompt (see <building_guide>)."
                ),
                session_id=session_id,
            )
        if session is not None:
            session.building_mode_requested = True
        return BuildingModeResponse(
            message="Entering agent building mode…",
            content=(
                "Building mode requested. Your context is being upgraded — "
                "the complete agent-building guide will appear in your "
                "system prompt as <building_guide> momentarily. Continue "
                "with the user's request once it arrives; do NOT call "
                "get_agent_building_guide."
            ),
            session_id=session_id,
        )
