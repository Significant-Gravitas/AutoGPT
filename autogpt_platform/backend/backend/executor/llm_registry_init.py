"""
Helper functions for LLM registry initialization in executor context.

These functions handle refreshing the LLM registry when the executor starts
and subscribing to real-time updates via Redis pub/sub.
"""

import logging

from backend.blocks._base import BlockSchema
from backend.data import db, llm_registry
from backend.data.block import initialize_blocks
from backend.data.block_cost_config import refresh_llm_costs
from backend.data.llm_registry import subscribe_to_registry_refresh

logger = logging.getLogger(__name__)


async def initialize_registry_for_executor() -> None:
    """
    Initialize blocks and refresh LLM registry in the executor context.

    This must run in the executor's event loop to have access to the database.
    """
    try:
        # Connect to database if not already connected
        if not db.is_connected():
            await db.connect()
            logger.info("[GraphExecutor] Connected to database for registry refresh")

        # Initialize blocks (internally refreshes LLM registry and costs)
        await initialize_blocks()
        logger.info("[GraphExecutor] Blocks initialized")
    except Exception as exc:
        logger.warning(
            "[GraphExecutor] Failed to refresh LLM registry on startup: %s",
            exc,
            exc_info=True,
        )


async def refresh_registry_on_notification() -> None:
    """Refresh LLM registry when notified via Redis pub/sub."""
    try:
        # Ensure DB is connected
        if not db.is_connected():
            await db.connect()

        # Refresh registry and costs
        await llm_registry.refresh_llm_registry()
        await refresh_llm_costs()

        # Clear block schema caches so they regenerate with new model options
        BlockSchema.clear_all_schema_caches()

        logger.info("[GraphExecutor] LLM registry refreshed from notification")
    except Exception as exc:
        logger.error(
            "[GraphExecutor] Failed to refresh LLM registry from notification: %s",
            exc,
            exc_info=True,
        )


async def subscribe_to_registry_updates() -> None:
    """Subscribe to Redis pub/sub for LLM registry refresh notifications."""
    await subscribe_to_registry_refresh(refresh_registry_on_notification)
