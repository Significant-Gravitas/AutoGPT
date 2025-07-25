import logging
from typing import List, Tuple

from backend.data.block import BlockInput, BlockType, get_block
from backend.data.graph import GraphModel
from backend.server.v2.iffy.models import BlockContentForModeration
from backend.server.v2.iffy.service import IffyService
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()


def moderate_block_content(
    graph_id: str,
    graph_exec_id: str,
    node_id: str,
    block_id: str,
    input_data: BlockInput,
    user_id: str,
) -> None:
    """
    Moderate the content of a single block before execution.

    Args:
        graph_id: The ID of the graph
        graph_exec_id: The ID of the graph execution
        node_id: The ID of the node being executed
        block_id: The ID of the block being executed
        input_data: Input data for the block
        user_id: The ID of the user running the block
    """
    if settings.config.behave_as == BehaveAs.LOCAL:
        return

    try:
        block = get_block(block_id)
        if not block or block.block_type == BlockType.NOTE:
            return

        block_content = BlockContentForModeration(
            graph_id=graph_id,
            graph_exec_id=graph_exec_id,
            node_id=node_id,
            block_id=block.id,
            block_name=block.name,
            block_type=block.block_type.value,
            input_data=input_data,
        )

        # Send to Iffy for moderation
        result = IffyService.moderate_content(user_id, block_content)

        # CRITICAL: Ensure we never proceed if moderation fails
        if not result.is_safe:
            logger.error(
                f"Content moderation failed for {block.name}: {result.reason}"
            )
            raise ValueError(f"Content moderation failed for {block.name}: {result.reason}")

    except Exception as e:
        logger.error(f"Error during content moderation: {str(e)}")
        raise ValueError("Content moderation system error")