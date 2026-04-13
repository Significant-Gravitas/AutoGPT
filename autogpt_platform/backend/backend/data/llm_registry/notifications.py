"""Pub/sub notifications for LLM registry cross-process synchronisation."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from backend.data.redis_client import HOST, PASSWORD, PORT
from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)

REGISTRY_REFRESH_CHANNEL = "llm_registry:refresh"


async def publish_registry_refresh_notification() -> None:
    """Publish a refresh signal so all other workers reload their in-process cache."""
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        await redis.publish(REGISTRY_REFRESH_CHANNEL, "refresh")
        logger.debug("Published LLM registry refresh notification")
    except Exception as e:
        logger.warning("Failed to publish registry refresh notification: %s", e)


async def subscribe_to_registry_refresh(
    on_refresh: Callable[[], Awaitable[None]],
) -> None:
    """Listen for registry refresh signals and call on_refresh each time one arrives.

    Designed to run as a long-lived background asyncio.Task.  Automatically
    reconnects if the Redis connection drops.

    Args:
        on_refresh: Async callable invoked on each refresh signal.
                    Typically ``llm_registry.refresh_llm_registry``.
    """
    while True:
        try:
            # Dedicated connection — pub/sub must not share a connection used
            # for regular commands.
            redis_sub = AsyncRedis(
                host=HOST, port=PORT, password=PASSWORD, decode_responses=True
            )
            pubsub = redis_sub.pubsub()
            await pubsub.subscribe(REGISTRY_REFRESH_CHANNEL)
            logger.info("Subscribed to LLM registry refresh channel")

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
                        logger.debug("LLM registry refresh signal received")
                        try:
                            await on_refresh()
                        except Exception as e:
                            logger.error(
                                "Error in registry on_refresh callback: %s", e
                            )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(
                        "Error processing registry refresh message: %s", e
                    )
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("LLM registry subscription task cancelled")
            break
        except Exception as e:
            logger.warning(
                "LLM registry subscription error: %s. Retrying in 5s...", e
            )
            await asyncio.sleep(5)
