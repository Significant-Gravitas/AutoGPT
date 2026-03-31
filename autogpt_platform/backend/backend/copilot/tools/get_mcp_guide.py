"""GetMCPGuideTool - Returns the MCP tool usage guide."""

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
        guide_path = Path(__file__).parent.parent / "sdk" / "mcp_tool_guide.md"
        _GUIDE_CACHE = guide_path.read_text(encoding="utf-8")
    return _GUIDE_CACHE


class MCPGuideResponse(ToolResponseBase):
    """Response containing the MCP tool guide."""

    type: ResponseType = ResponseType.MCP_GUIDE
    content: str


class GetMCPGuideTool(BaseTool):
    """Returns the MCP tool usage guide with known server URLs and auth details."""

    @property
    def name(self) -> str:
        return "get_mcp_guide"

    @property
    def description(self) -> str:
        return "Get MCP server URLs and auth guide. Call before run_mcp_tool if you need a server URL or auth info."

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
            return MCPGuideResponse(
                message="MCP guide loaded.",
                content=content,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Failed to load MCP guide: %s", e)
            return ErrorResponse(
                message="Failed to load MCP guide.",
                error=str(e),
                session_id=session_id,
            )
