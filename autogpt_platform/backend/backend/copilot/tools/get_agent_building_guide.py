"""GetAgentBuildingGuideTool - Returns the complete agent building guide."""

import logging
from pathlib import Path
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)

# Heading title (without the ``### ``) used to drop the trigger-agents
# section when the generic-trigger-agents feature flag is off. Keep
# this string in sync with the heading in agent_generation_guide.md —
# the guide-gating test locks both branches.
_TRIGGER_AGENTS_HEADING = "Building Trigger Agents"

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
            "Agent JSON building guide (nodes, links, AgentExecutorBlock, "
            "MCPToolBlock, iterative create->dry-run->fix flow). REQUIRED "
            "before create_agent / edit_agent / validate_agent_graph / "
            "fix_agent_graph — they refuse until called once per session."
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
        **kwargs,  # no tool-specific params; accepts kwargs for forward-compat
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        try:
            content = _load_guide()
            triggers_enabled = (
                await is_feature_enabled(
                    Flag.GENERIC_TRIGGER_AGENTS, user_id, default=False
                )
                if user_id
                else False
            )
            if not triggers_enabled:
                content = _strip_h3_section(content, _TRIGGER_AGENTS_HEADING)
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


def _strip_h3_section(guide: str, heading: str) -> str:
    """Remove a single ``### {heading}`` section from the guide.

    The section runs from its ``### `` heading up to (but not
    including) the next H2/H3 heading, or end-of-file if it's the
    last section. Sections after the stripped one are preserved —
    future additions to the guide aren't silently dropped.
    """
    lines = guide.split("\n")
    target = f"### {heading}"
    out: list[str] = []
    skipping = False
    for line in lines:
        if line == target:
            skipping = True
            continue
        if skipping and line.startswith(("# ", "## ", "### ")):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)
