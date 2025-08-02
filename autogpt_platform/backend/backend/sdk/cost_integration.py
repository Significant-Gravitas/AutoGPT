"""
Integration between SDK provider costs and the execution cost system.

This module provides the glue between provider-defined base costs and the 
BLOCK_COSTS configuration used by the execution system.
"""

import logging
from typing import List, Type

from backend.data.block import Block
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.cost import BlockCost
from backend.sdk.registry import AutoRegistry

logger = logging.getLogger(__name__)


def register_provider_costs_for_block(block_class: Type[Block]) -> None:
    """
    Register provider base costs for a specific block in BLOCK_COSTS.

    This function checks if the block uses credentials from a provider that has
    base costs defined, and automatically registers those costs for the block.

    Args:
        block_class: The block class to register costs for
    """
    # Skip if block already has custom costs defined
    if block_class in BLOCK_COSTS:
        logger.debug(
            f"Block {block_class.__name__} already has costs defined, skipping provider costs"
        )
        return

    # Get the block's input schema
    # We need to instantiate the block to get its input schema
    try:
        block_instance = block_class()
        input_schema = block_instance.input_schema
    except Exception as e:
        logger.debug(f"Block {block_class.__name__} cannot be instantiated: {e}")
        return

    # Look for credentials fields
    # The cost system works of filtering on credentials fields,
    # without credentials fields, we can not apply costs
    # TODO: Improve cost system to allow for costs witout a provider
    credentials_fields = input_schema.get_credentials_fields()
    if not credentials_fields:
        logger.debug(f"Block {block_class.__name__} has no credentials fields")
        return

    # Get provider information from credentials fields
    for field_name, field_info in credentials_fields.items():
        # Get the field schema to extract provider information
        field_schema = input_schema.get_field_schema(field_name)

        # Extract provider names from json_schema_extra
        providers = field_schema.get("credentials_provider", [])
        if not providers:
            continue

        # For each provider, check if it has base costs
        block_costs: List[BlockCost] = []
        for provider_name in providers:
            provider = AutoRegistry.get_provider(provider_name)
            if not provider:
                logger.debug(f"Provider {provider_name} not found in registry")
                continue

            # Add provider's base costs to the block
            if provider.base_costs:
                logger.debug(
                    f"Registering {len(provider.base_costs)} base costs from provider {provider_name} for block {block_class.__name__}"
                )
                block_costs.extend(provider.base_costs)

        # Register costs if any were found
        if block_costs:
            BLOCK_COSTS[block_class] = block_costs
            logger.debug(
                f"Registered {len(block_costs)} total costs for block {block_class.__name__}"
            )


def sync_all_provider_costs() -> None:
    """
    Sync all provider base costs to blocks that use them.

    This should be called after all providers and blocks are registered,
    typically during application startup.
    """
    from backend.blocks import load_all_blocks

    logger.info("Syncing provider costs to blocks...")

    blocks_with_costs = 0
    total_costs = 0

    for block_id, block_class in load_all_blocks().items():
        initial_count = len(BLOCK_COSTS.get(block_class, []))
        register_provider_costs_for_block(block_class)
        final_count = len(BLOCK_COSTS.get(block_class, []))

        if final_count > initial_count:
            blocks_with_costs += 1
            total_costs += final_count - initial_count

    logger.info(f"Synced {total_costs} costs to {blocks_with_costs} blocks")


def get_block_costs(block_class: Type[Block]) -> List[BlockCost]:
    """
    Get all costs for a block, including both explicit and provider costs.

    Args:
        block_class: The block class to get costs for

    Returns:
        List of BlockCost objects for the block
    """
    # First ensure provider costs are registered
    register_provider_costs_for_block(block_class)

    # Return all costs for the block
    return BLOCK_COSTS.get(block_class, [])


def cost(*costs: BlockCost):
    """
    Decorator to set custom costs for a block.

    This decorator allows blocks to define their own costs, which will override
    any provider base costs. Multiple costs can be specified with different
    filters for different pricing tiers (e.g., different models).

    Example:
        @cost(
            BlockCost(cost_type=BlockCostType.RUN, cost_amount=10),
            BlockCost(
                cost_type=BlockCostType.RUN,
                cost_amount=20,
                cost_filter={"model": "premium"}
            )
        )
        class MyBlock(Block):
            ...

    Args:
        *costs: Variable number of BlockCost objects
    """

    def decorator(block_class: Type[Block]) -> Type[Block]:
        # Register the costs for this block
        if costs:
            BLOCK_COSTS[block_class] = list(costs)
            logger.info(
                f"Registered {len(costs)} custom costs for block {block_class.__name__}"
            )
        return block_class

    return decorator
