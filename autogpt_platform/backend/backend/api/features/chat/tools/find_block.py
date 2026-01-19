import logging
from typing import Any

from langfuse import observe
from prisma.enums import ContentType

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools.base import BaseTool, ToolResponseBase
from backend.api.features.chat.tools.models import (
    BlockInfoSummary,
    BlockInputFieldInfo,
    BlockListResponse,
    ErrorResponse,
    NoResultsResponse,
)
from backend.api.features.store.hybrid_search import unified_hybrid_search
from backend.data.block import get_block

logger = logging.getLogger(__name__)


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
            "The response includes each block's id, required_inputs, and input_schema."
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

    @observe(as_type="tool", name="find_block")
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
            results, total = await unified_hybrid_search(
                query=query,
                content_types=[ContentType.BLOCK],
                page=1,
                page_size=10,
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

            # Enrich results with full block information
            blocks: list[BlockInfoSummary] = []
            for result in results:
                block_id = result["content_id"]
                block = get_block(block_id)

                if block:
                    # Get input/output schemas
                    input_schema = {}
                    output_schema = {}
                    try:
                        input_schema = block.input_schema.jsonschema()
                    except Exception:
                        pass
                    try:
                        output_schema = block.output_schema.jsonschema()
                    except Exception:
                        pass

                    # Get categories from block instance
                    categories = []
                    if hasattr(block, "categories") and block.categories:
                        categories = [cat.value for cat in block.categories]

                    # Extract required inputs for easier use
                    required_inputs: list[BlockInputFieldInfo] = []
                    if input_schema:
                        properties = input_schema.get("properties", {})
                        required_fields = set(input_schema.get("required", []))
                        # Get credential field names to exclude from required inputs
                        credentials_fields = set(
                            block.input_schema.get_credentials_fields().keys()
                        )

                        for field_name, field_schema in properties.items():
                            # Skip credential fields - they're handled separately
                            if field_name in credentials_fields:
                                continue

                            required_inputs.append(
                                BlockInputFieldInfo(
                                    name=field_name,
                                    type=field_schema.get("type", "string"),
                                    description=field_schema.get("description", ""),
                                    required=field_name in required_fields,
                                    default=field_schema.get("default"),
                                )
                            )

                    blocks.append(
                        BlockInfoSummary(
                            id=block_id,
                            name=block.name,
                            description=block.description or "",
                            categories=categories,
                            input_schema=input_schema,
                            output_schema=output_schema,
                            required_inputs=required_inputs,
                        )
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
                    "To execute a block, use run_block with the block's 'id' field "
                    "and provide 'input_data' matching the block's input_schema."
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
