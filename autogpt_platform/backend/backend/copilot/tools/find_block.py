import logging
from typing import Any

from prisma.enums import ContentType

from backend.blocks import get_block
from backend.blocks._base import BlockType
from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession
from backend.data.db_accessors import search

from .base import BaseTool, ToolResponseBase
from .models import (
    BlockInfoSummary,
    BlockListResponse,
    ErrorResponse,
    NoResultsResponse,
)
from .utils import is_uuid

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
    BlockType.MCP_TOOL,  # Has dedicated run_mcp_tool tool with discovery + auth flow
}

# Specific block IDs excluded from CoPilot (STANDARD type but still require graph context)
COPILOT_EXCLUDED_BLOCK_IDS = {
    # OrchestratorBlock - dynamically discovers downstream blocks via graph topology;
    # usable in agent graphs (guide hardcodes its ID) but cannot run standalone.
    "3b191d9f-356f-482d-8238-ba04b6d18381",
    # AutoPilotBlock - has dedicated run_sub_session tool with async start +
    # poll lifecycle. Calling it via run_block would block the parent stream
    # for the sub-AutoPilot's entire runtime (15-45+ min typical).
    "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6",
}


class FindBlockTool(BaseTool):
    """Tool for searching available blocks."""

    @property
    def name(self) -> str:
        return "find_block"

    @property
    def description(self) -> str:
        return (
            "Search blocks by name or description. Returns block IDs for run_block. "
            "Always call this FIRST to get block IDs before using run_block. "
            "Then call run_block with the block's id and empty input_data to see its detailed schema."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords (e.g. 'email', 'http', 'ai').",
                },
                "include_schemas": {
                    "type": "boolean",
                    "description": "Include full input/output schemas (for agent JSON generation).",
                    "default": False,
                },
                "for_agent_generation": {
                    "type": "boolean",
                    "description": (
                        "Set to true when searching for blocks to use inside an agent graph "
                        "(e.g. AgentInputBlock, AgentOutputBlock, OrchestratorBlock). "
                        "Bypasses the CoPilot-only filter so graph-only blocks are visible."
                    ),
                    "default": False,
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
        query: str = "",
        include_schemas: bool = False,
        for_agent_generation: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        """Search for blocks matching the query.

        Args:
            user_id: User ID (required)
            session: Chat session
            query: Search query
            include_schemas: Whether to include block schemas in results
            for_agent_generation: When True, bypasses the CoPilot exclusion filter
                so graph-only blocks (INPUT, OUTPUT, ORCHESTRATOR, etc.) are visible.

        Returns:
            BlockListResponse: List of matching blocks
            NoResultsResponse: No blocks found
            ErrorResponse: Error message
        """
        query = (query or "").strip()
        session_id = session.session_id

        if not query:
            return ErrorResponse(
                message="Please provide a search query or block ID",
                session_id=session_id,
            )

        try:
            # Direct ID lookup if query looks like a UUID
            if is_uuid(query):
                block = get_block(query.lower())
                if block:
                    if block.disabled:
                        return NoResultsResponse(
                            message=f"Block '{block.name}' (ID: {block.id}) is disabled and cannot be used.",
                            suggestions=["Search for an alternative block by name"],
                            session_id=session_id,
                        )
                    is_excluded = (
                        block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
                        or block.id in COPILOT_EXCLUDED_BLOCK_IDS
                    )
                    if is_excluded:
                        # Graph-only blocks (INPUT, OUTPUT, MCP_TOOL, AGENT, etc.) are
                        # exposed when building an agent graph so the LLM can inspect
                        # their schemas and wire them as nodes.  In CoPilot direct use
                        # they are not executable — guide the LLM to the right tool.
                        if not for_agent_generation:
                            if block.block_type == BlockType.MCP_TOOL:
                                message = (
                                    f"Block '{block.name}' (ID: {block.id}) cannot be "
                                    "run directly in CoPilot. Use run_mcp_tool for "
                                    "interactive MCP execution, or call find_block with "
                                    "for_agent_generation=true to embed it in an agent graph."
                                )
                            else:
                                message = (
                                    f"Block '{block.name}' (ID: {block.id}) is not available "
                                    "in CoPilot. It can only be used within agent graphs."
                                )
                            return NoResultsResponse(
                                message=message,
                                suggestions=[
                                    "Search for an alternative block by name",
                                    "Use this block in an agent graph instead",
                                ],
                                session_id=session_id,
                            )

                    # Check block-level permissions — hide denied blocks entirely
                    perms = get_current_permissions()
                    if perms is not None and not perms.is_block_allowed(
                        block.id, block.name
                    ):
                        return NoResultsResponse(
                            message=f"No blocks found for '{query}'",
                            suggestions=[
                                "Search for an alternative block by name",
                            ],
                            session_id=session_id,
                        )

                    summary = BlockInfoSummary(
                        id=block.id,
                        name=block.name,
                        description=(
                            block.optimized_description or block.description or ""
                        ),
                        categories=[c.value for c in block.categories],
                    )
                    if include_schemas:
                        info = block.get_info()
                        summary.input_schema = info.inputSchema
                        summary.output_schema = info.outputSchema
                        summary.static_output = info.staticOutput

                    return BlockListResponse(
                        message=(
                            f"Found block '{block.name}' by ID. "
                            "To see inputs/outputs and execute it, use "
                            "run_block with the block's 'id' - providing "
                            "no inputs."
                        ),
                        blocks=[summary],
                        count=1,
                        query=query,
                        session_id=session_id,
                    )

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
            perms = get_current_permissions()
            blocks: list[BlockInfoSummary] = []
            for result in results:
                block_id = result["content_id"]
                block = get_block(block_id)

                # Skip disabled blocks
                if not block or block.disabled:
                    continue

                # Graph-only blocks (INPUT, OUTPUT, MCP_TOOL, AGENT, etc.) are
                # skipped in CoPilot direct use but surfaced for agent graph building.
                if not for_agent_generation and (
                    block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
                    or block.id in COPILOT_EXCLUDED_BLOCK_IDS
                ):
                    continue

                # Skip blocks denied by execution permissions
                if perms is not None and not perms.is_block_allowed(
                    block.id, block.name
                ):
                    continue

                summary = BlockInfoSummary(
                    id=block_id,
                    name=block.name,
                    description=block.optimized_description or block.description or "",
                    categories=[c.value for c in block.categories],
                )

                if include_schemas:
                    info = block.get_info()
                    summary.input_schema = info.inputSchema
                    summary.output_schema = info.outputSchema
                    summary.static_output = info.staticOutput

                blocks.append(summary)

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
