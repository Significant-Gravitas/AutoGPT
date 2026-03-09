"""GetAgentBuildingGuideTool - Returns the complete agent building guide."""

import logging
from pathlib import Path
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

_GUIDE_CACHE: str | None = None


def _load_guide() -> str:
    global _GUIDE_CACHE
    if _GUIDE_CACHE is None:
        guide_path = Path(__file__).parent.parent / "sdk" / "agent_generation_guide.md"
        _GUIDE_CACHE = guide_path.read_text(encoding="utf-8")
    return _GUIDE_CACHE


class AgentBuildingGuideResponse(ToolResponseBase):
    """Response containing the agent building guide."""

    type: ResponseType = ResponseType.AGENT_BUILDER_GUIDE
    content: str


class GetAgentBuildingGuideTool(BaseTool):
    """Returns the complete guide for building agent JSON graphs.

    Covers block IDs, link structure, AgentInputBlock, AgentOutputBlock,
    AgentExecutorBlock (sub-agent composition), and MCPToolBlock usage.
    """

    @property
    def name(self) -> str:
        return "get_agent_building_guide"

    @property
    def description(self) -> str:
        return (
            "Returns the complete guide for building agent JSON graphs, including "
            "block IDs, link structure, AgentInputBlock, AgentOutputBlock, "
            "AgentExecutorBlock (for sub-agent composition), and MCPToolBlock usage. "
            "Call this before generating agent JSON to ensure correct structure."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

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
        try:
            content = _load_guide()
            return AgentBuildingGuideResponse(
                message="Agent building guide loaded.",
                content=content,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Failed to load agent building guide: %s", e)
            return ErrorResponse(
                message="Failed to load agent building guide.",
                error=str(e),
                session_id=session_id,
            )
