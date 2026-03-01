"""GetBlocksForGoalTool - Recommends relevant blocks for agent generation."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.blocks import recommend_blocks_for_goal
from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class GetBlocksForGoalTool(BaseTool):
    """Tool for discovering relevant blocks given a natural-language goal."""

    @property
    def name(self) -> str:
        return "get_blocks_for_goal"

    @property
    def description(self) -> str:
        return (
            "Discover relevant blocks (building blocks) for constructing an agent. "
            "Given a natural-language goal, returns the most relevant blocks with "
            "their full input/output schemas. Always call this FIRST when generating "
            "or editing an agent so you know which blocks are available.\n\n"
            "Each returned block includes:\n"
            "- id: UUID to use as block_id in agent nodes\n"
            "- name: Human-readable block name\n"
            "- description: What the block does\n"
            "- inputSchema: JSON schema for the block's inputs\n"
            "- outputSchema: JSON schema for the block's outputs\n"
            "- categories: Block categories (e.g. BASIC, AI, SOCIAL)\n"
            "- staticOutput: Whether outputs are known at design time\n\n"
            "Blocks in the BASIC category (utility blocks) are always included."
        )

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what the agent should do. "
                        "Example: 'monitor Twitter for mentions and send email alerts'"
                    ),
                },
                "max_blocks": {
                    "type": "integer",
                    "description": (
                        "Maximum number of blocks to return. Default is 25."
                    ),
                    "default": 25,
                },
            },
            "required": ["goal"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        goal = kwargs.get("goal", "").strip()
        max_blocks = kwargs.get("max_blocks", 25)
        session_id = session.session_id if session else None

        if not goal:
            return ErrorResponse(
                message="Please provide a goal describing what the agent should do.",
                error="Missing goal parameter",
                session_id=session_id,
            )

        try:
            blocks = recommend_blocks_for_goal(goal, max_blocks=max_blocks)
        except Exception as e:
            logger.error(f"Failed to recommend blocks: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to load block information. Please try again.",
                error=str(e),
                session_id=session_id,
            )

        # Format blocks for the response — include full schemas for generation
        block_infos = []
        for block in blocks:
            block_infos.append(
                {
                    "id": block["id"],
                    "name": block["name"],
                    "description": block.get("description", ""),
                    "inputSchema": block.get("inputSchema", {}),
                    "outputSchema": block.get("outputSchema", {}),
                    "categories": [
                        cat.get("category", "") for cat in block.get("categories", [])
                    ],
                    "staticOutput": block.get("staticOutput", False),
                    "relevance_score": block.get("relevance_score", 0),
                }
            )

        from .models import BlocksForGoalResponse

        return BlocksForGoalResponse(
            message=f"Found {len(block_infos)} relevant blocks for your goal.",
            blocks=block_infos,
            count=len(block_infos),
            goal=goal,
            session_id=session_id,
        )
