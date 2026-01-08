"""Tool for searching available blocks using hybrid search."""

import logging
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.blocks import load_all_blocks

from .base import BaseTool
from .models import (
    BlockInfoSummary,
    BlockListResponse,
    ErrorResponse,
    NoResultsResponse,
    ToolResponseBase,
)
from .search_blocks import get_block_search_index

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
            "Use this to find blocks that can be executed directly."
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

    def _matches_query(self, block, query: str) -> tuple[int, bool]:
        """
        Check if a block matches the query and return a priority score.

        Returns (priority, matches) where:
        - priority 0: exact name match
        - priority 1: name contains query
        - priority 2: description contains query
        - priority 3: category contains query
        """
        query_lower = query.lower()
        name_lower = block.name.lower()
        desc_lower = block.description.lower()

        # Exact name match
        if query_lower == name_lower:
            return 0, True

        # Name contains query
        if query_lower in name_lower:
            return 1, True

        # Description contains query
        if query_lower in desc_lower:
            return 2, True

        # Category contains query
        for category in block.categories:
            if query_lower in category.name.lower():
                return 3, True

        return 4, False

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
            # Try hybrid search first
            search_results = self._hybrid_search(query)

            if search_results is not None:
                # Hybrid search succeeded
                if not search_results:
                    return NoResultsResponse(
                        message=f"No blocks found matching '{query}'",
                        session_id=session_id,
                        suggestions=[
                            "Try more general terms",
                            "Search by category: ai, text, social, search, etc.",
                            "Check block names like 'SendEmail', 'HttpRequest', etc.",
                        ],
                    )

                # Get full block info for each result
                all_blocks = load_all_blocks()
                blocks = []
                for result in search_results:
                    block_cls = all_blocks.get(result.block_id)
                    if block_cls:
                        block = block_cls()
                        blocks.append(
                            BlockInfoSummary(
                                id=block.id,
                                name=block.name,
                                description=block.description,
                                categories=[cat.name for cat in block.categories],
                                input_schema=block.input_schema.jsonschema(),
                                output_schema=block.output_schema.jsonschema(),
                            )
                        )

                return BlockListResponse(
                    message=(
                        f"Found {len(blocks)} block{'s' if len(blocks) != 1 else ''} "
                        f"matching '{query}'. Use run_block to execute a block with "
                        "the required inputs."
                    ),
                    blocks=blocks,
                    count=len(blocks),
                    query=query,
                    session_id=session_id,
                )

            # Fallback to simple search if hybrid search failed
            return self._simple_search(query, session_id)

        except Exception as e:
            logger.error(f"Error searching blocks: {e}", exc_info=True)
            return ErrorResponse(
                message="Failed to search blocks. Please try again.",
                error=str(e),
                session_id=session_id,
            )

    def _hybrid_search(self, query: str) -> list | None:
        """
        Perform hybrid search using embeddings and BM25.

        Returns:
            List of BlockSearchResult if successful, None if index not available
        """
        try:
            index = get_block_search_index()
            if not index.load():
                logger.info(
                    "Block search index not available, falling back to simple search"
                )
                return None

            results = index.search(query, top_k=10)
            logger.info(f"Hybrid search found {len(results)} blocks for: {query}")
            return results

        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to simple: {e}")
            return None

    def _simple_search(self, query: str, session_id: str) -> ToolResponseBase:
        """Fallback simple search using substring matching."""
        all_blocks = load_all_blocks()
        logger.info(f"Simple searching {len(all_blocks)} blocks for: {query}")

        # Find matching blocks with priority scores
        matches: list[tuple[int, Any]] = []
        for block_id, block_cls in all_blocks.items():
            block = block_cls()
            priority, is_match = self._matches_query(block, query)
            if is_match:
                matches.append((priority, block))

        # Sort by priority (lower is better)
        matches.sort(key=lambda x: x[0])

        # Take top 10 results
        top_matches = [block for _, block in matches[:10]]

        if not top_matches:
            return NoResultsResponse(
                message=f"No blocks found matching '{query}'",
                session_id=session_id,
                suggestions=[
                    "Try more general terms",
                    "Search by category: ai, text, social, search, etc.",
                    "Check block names like 'SendEmail', 'HttpRequest', etc.",
                ],
            )

        # Build response
        blocks = []
        for block in top_matches:
            blocks.append(
                BlockInfoSummary(
                    id=block.id,
                    name=block.name,
                    description=block.description,
                    categories=[cat.name for cat in block.categories],
                    input_schema=block.input_schema.jsonschema(),
                    output_schema=block.output_schema.jsonschema(),
                )
            )

        return BlockListResponse(
            message=(
                f"Found {len(blocks)} block{'s' if len(blocks) != 1 else ''} "
                f"matching '{query}'. Use run_block to execute a block with "
                "the required inputs."
            ),
            blocks=blocks,
            count=len(blocks),
            query=query,
            session_id=session_id,
        )
