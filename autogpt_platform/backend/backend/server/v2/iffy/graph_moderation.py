import logging
from typing import List, Tuple

from backend.data.block import BlockInput, BlockType, get_block
from backend.data.graph import GraphModel
from backend.server.v2.iffy.service import IffyService
from backend.util.settings import Settings, BehaveAs

logger = logging.getLogger(__name__)
settings = Settings()

def moderate_graph_content(
    graph: GraphModel,
    graph_id: str,
    graph_exec_id: str,
    nodes_input: List[Tuple[str, BlockInput]],
    user_id: str
) -> None:
    """
    Moderate the content of a graph before execution.
    
    Args:
        graph: The graph model to moderate
        graph_id: The ID of the graph
        graph_exec_id: The ID of the graph execution
        nodes_input: Input data for starting nodes
        user_id: The ID of the user running the graph
    """
    if settings.config.behave_as == BehaveAs.LOCAL:
        return
        
    try:
        for node in graph.nodes:
            block = get_block(node.block_id)
            if not block or block.block_type == BlockType.NOTE:
                continue

            # For starting nodes, use their input data
            if node.id in dict(nodes_input):
                input_data = dict(nodes_input)[node.id]
            else:
                # For non-starting nodes, collect their default inputs and static values
                input_data = node.input_default.copy()
                # Add any static input from connected nodes
                for link in node.input_links:
                    if link.is_static:
                        source_node = next((n for n in graph.nodes if n.id == link.source_id), None)
                        if source_node:
                            source_block = get_block(source_node.block_id)
                            if source_block:
                                input_data[link.sink_name] = source_node.input_default.get(link.source_name)
            
            block_content = {
                "graph_id": graph_id,
                "graph_exec_id": graph_exec_id,
                "node_id": node.id,
                "block_id": block.id,
                "block_name": block.name,
                "block_type": block.block_type.value,
                "input_data": input_data
            }

            # Send to Iffy for moderation
            result = IffyService.moderate_content(user_id, block_content)
            
            # CRITICAL: Ensure we never proceed if moderation fails
            if not result.is_safe:
                logger.error(f"Content moderation failed for {block.name}: {result.reason}")
                raise ValueError(f"Content moderation failed for {block.name}")

    except ValueError as e:
        logger.error(f"Moderation error: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Error during content moderation: {str(e)}")
        raise ValueError(f"Content moderation system error")