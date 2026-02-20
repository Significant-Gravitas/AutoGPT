import logging
from typing import Any

from prisma.enums import ContentType

from backend.blocks import get_block
from backend.blocks._base import BlockType
from backend.copilot.model import ChatSession
from backend.data.db_accessors import search

from .base import BaseTool, ToolResponseBase
from .models import (
    BlockInfoSummary,
    BlockListResponse,
    ErrorResponse,
    NoResultsResponse,
)

logger = logging.getLogger(__name__)

_TARGET_RESULTS = 10
# Over-fetch to compensate for post-hoc filtering of graph-only blocks.
# 40 is 2x current removed; speed of query 10 vs 40 is minimial
_OVERFETCH_PAGE_SIZE = 40

# Block types that only work within graphs and cannot run standalone in CoPilot.
COPILOT_EXCLUDED_BLOCK_TYPES = {
    BlockType.INPUT,  # Graph interface definition - data enters via chat, not graph inputs
    BlockType.OUTPUT,  # Graph interface definition - data exits via chat, not graph outputs
    BlockType.WEBHOOK,  # Wait for external events - would hang forever in CoPilot
    BlockType.WEBHOOK_MANUAL,  # Same as WEBHOOK
    BlockType.NOTE,  # Visual annotation only - no runtime behavior
    BlockType.HUMAN_IN_THE_LOOP,  # Pauses for human approval - CoPilot IS human-in-the-loop
    BlockType.AGENT,  # AgentExecutorBlock requires execution_context - use run_agent tool
}

# Specific block IDs excluded from CoPilot (STANDARD type but still require graph context)
COPILOT_EXCLUDED_BLOCK_IDS = {
    # SmartDecisionMakerBlock - dynamically discovers downstream blocks via graph topology
    "3b191d9f-356f-482d-8238-ba04b6d18381",
}


class FindBlockTool(BaseTool):
    """Tool for searching available blocks."""

    @property
    def name(self) -> str:
        return "find_block"

    @property
    def description(self) -> str:
        return (
            "Search for available blocks by name or description. "
            "Blocks are reusable components that perform specific tasks like "
            "sending emails, making API calls, processing text, etc. "
            "IMPORTANT: Use this tool FIRST to get the block's 'id' before calling run_block. "
            "The response includes each block's id, name, and description. "
            "Call run_block with the block's id **with no inputs** to see detailed inputs/outputs and execute it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query to find blocks by name or description. "
                        "Use keywords like 'email', 'http', 'text', 'ai', etc."
                    ),
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for blocks matching the query.

        Args:
            user_id: User ID (required)
            session: Chat session
            query: Search query

        Returns:
            BlockListResponse: List of matching blocks
            NoResultsResponse: No blocks found
            ErrorResponse: Error message
        """
        query = kwargs.get("query", "").strip()
        session_id = session.session_id

        if not query:
            return ErrorResponse(
                message="Please provide a search query",
                session_id=session_id,
            )

        try:
            # Search for blocks using hybrid search
            results, total = await search().unified_hybrid_search(
                query=query,
                content_types=[ContentType.BLOCK],
                page=1,
                page_size=_OVERFETCH_PAGE_SIZE,
            )

            if not results:
                return NoResultsResponse(
                    message=f"No blocks found for '{query}'",
                    suggestions=[
                        "Try broader keywords like 'email', 'http', 'text', 'ai'",
                        "Check spelling of technical terms",
                    ],
                    session_id=session_id,
                )

            # Enrich results with block information
            blocks: list[BlockInfoSummary] = []
            for result in results:
                block_id = result["content_id"]
                block = get_block(block_id)

                # Skip disabled blocks
                if not block or block.disabled:
                    continue

                # Skip blocks excluded from CoPilot (graph-only blocks)
                if (
                    block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
                    or block.id in COPILOT_EXCLUDED_BLOCK_IDS
                ):
                    continue

                blocks.append(
                    BlockInfoSummary(
                        id=block_id,
                        name=block.name,
                        description=block.description or "",
                        categories=[c.value for c in block.categories],
                    )
                )

                if len(blocks) >= _TARGET_RESULTS:
                    break

            if blocks and len(blocks) < _TARGET_RESULTS:
                logger.debug(
                    "find_block returned %d/%d results for query '%s' "
                    "(filtered %d excluded/disabled blocks)",
                    len(blocks),
                    _TARGET_RESULTS,
                    query,
                    len(results) - len(blocks),
                )

            if not blocks:
                return NoResultsResponse(
                    message=f"No blocks found for '{query}'",
                    suggestions=[
                        "Try broader keywords like 'email', 'http', 'text', 'ai'",
                    ],
                    session_id=session_id,
                )

            return BlockListResponse(
                message=(
                    f"Found {len(blocks)} block(s) matching '{query}'. "
                    "To see a block's inputs/outputs and execute it, use run_block with the block's 'id' - providing no inputs."
                ),
                blocks=blocks,
                count=len(blocks),
                query=query,
                session_id=session_id,
            )

        except Exception as e:
            logger.error(f"Error searching blocks: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search blocks",
                error=str(e),
                session_id=session_id,
            )
