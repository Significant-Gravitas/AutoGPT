"""
Redis pub/sub notifications for LLM registry updates.

When models are added/updated/removed via the admin UI, this module
publishes notifications to Redis that all executor services subscribe to,
ensuring they refresh their registry cache in real-time.
"""

import asyncio
import logging
from typing import Any

from backend.data.redis_client import connect_async

logger = logging.getLogger(__name__)

# Redis channel name for LLM registry refresh notifications
REGISTRY_REFRESH_CHANNEL = "llm_registry:refresh"


async def publish_registry_refresh_notification() -> None:
    """
    Publish a notification to Redis that the LLM registry has been updated.
    All executor services subscribed to this channel will refresh their registry.
    """
    try:
        redis = await connect_async()
        await redis.publish(REGISTRY_REFRESH_CHANNEL, "refresh")
        logger.info("Published LLM registry refresh notification to Redis")
    except Exception as exc:
        logger.warning(
            "Failed to publish LLM registry refresh notification: %s",
            exc,
            exc_info=True,
        )


async def subscribe_to_registry_refresh(
    on_refresh: Any,  # Async callable that takes no args
) -> None:
    """
    Subscribe to Redis notifications for LLM registry updates.
    This runs in a loop and processes messages as they arrive.

    Args:
        on_refresh: Async callable to execute when a refresh notification is received
    """
    try:
        redis = await connect_async()
        pubsub = redis.pubsub()
        await pubsub.subscribe(REGISTRY_REFRESH_CHANNEL)
        logger.info(
            "Subscribed to LLM registry refresh notifications on channel: %s",
            REGISTRY_REFRESH_CHANNEL,
        )

        # Process messages in a loop
        while True:
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if (
                    message
                    and message["type"] == "message"
                    and message["channel"] == REGISTRY_REFRESH_CHANNEL
                ):
                    logger.info("Received LLM registry refresh notification")
                    try:
                        await on_refresh()
                    except Exception as exc:
                        logger.error(
                            "Error refreshing LLM registry from notification: %s",
                            exc,
                            exc_info=True,
                        )
            except Exception as exc:
                logger.warning(
                    "Error processing registry refresh message: %s", exc, exc_info=True
                )
                # Continue listening even if one message fails
                await asyncio.sleep(1)
    except Exception as exc:
        logger.error(
            "Failed to subscribe to LLM registry refresh notifications: %s",
            exc,
            exc_info=True,
        )
        raise
