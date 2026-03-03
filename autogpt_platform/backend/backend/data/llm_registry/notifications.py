"""
Redis pub/sub notifications for LLM registry updates.

When models are added/updated/removed via the admin UI, this module
publishes notifications to Redis that all executor services subscribe to,
ensuring they refresh their registry cache in real-time.
"""

import asyncio
import json
import logging
from typing import Any

from backend.data.redis_client import connect_async

logger = logging.getLogger(__name__)

# Redis channel name for LLM registry refresh notifications
REGISTRY_REFRESH_CHANNEL = "llm_registry:refresh"


async def publish_registry_refresh_notification(
    models_data: list[dict[str, Any]] | None = None
) -> None:
    """
    Publish a notification to Redis that the LLM registry has been updated.

    Args:
        models_data: Optional full registry data to include in notification
    """
    try:
        redis = await connect_async()

        # Prepare payload
        if models_data is not None:
            payload = json.dumps({"action": "refresh", "data": models_data})
        else:
            payload = "refresh"  # Backwards compatible

        await redis.publish(REGISTRY_REFRESH_CHANNEL, payload)

        if models_data:
            logger.info(
                "Published LLM registry refresh notification with %d models",
                len(models_data),
            )
        else:
            logger.info("Published LLM registry refresh notification")

    except Exception as exc:
        logger.warning(
            "Failed to publish LLM registry refresh notification: %s",
            exc,
            exc_info=True,
        )


async def subscribe_to_registry_refresh(
    on_refresh: Any,  # Async callable that takes optional models_data
) -> None:
    """
    Subscribe to Redis notifications for LLM registry updates.

    Args:
        on_refresh: Async callable(models_data: list[dict] | None) -> None
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

                    # Extract models_data if present
                    models_data = None
                    try:
                        payload = message["data"]
                        if isinstance(payload, bytes):
                            payload = payload.decode("utf-8")

                        # Try to parse as JSON
                        if payload != "refresh":
                            parsed = json.loads(payload)
                            models_data = parsed.get("data")
                            logger.debug(
                                "Notification includes %d models",
                                len(models_data) if models_data else 0,
                            )
                    except (json.JSONDecodeError, AttributeError):
                        # Backwards compatible: simple "refresh" string
                        pass

                    try:
                        await on_refresh(models_data)
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
