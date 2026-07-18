"""Pub/sub notifications for LLM registry cross-process synchronisation.

Uses sharded pub/sub (SPUBLISH/SSUBSCRIBE) — prod Redis runs in cluster mode,
where classic pub/sub broadcasts to every node. Mirrors the pattern in
``backend.data.event_bus.AsyncRedisEventBus``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from backend.data.redis_client import connect_sharded_pubsub_async, get_redis_async

logger = logging.getLogger(__name__)

REGISTRY_REFRESH_CHANNEL = "llm_registry:refresh"


async def publish_registry_refresh_notification() -> None:
    """Publish a refresh signal so all other workers reload their in-process cache."""
    try:
        cluster = await get_redis_async()
        # redis-py 6.x async cluster has no spublish(); execute_command handles MOVED.
        await cluster.execute_command("SPUBLISH", REGISTRY_REFRESH_CHANNEL, "refresh")
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
        client = None
        pubsub = None
        try:
            # Sharded pub/sub only delivers on the keyslot-owning shard, so pin
            # a plain AsyncRedis to that node (see event_bus.listen_events).
            client = await connect_sharded_pubsub_async(REGISTRY_REFRESH_CHANNEL)
            pubsub = client.pubsub()
            await pubsub.execute_command("SSUBSCRIBE", REGISTRY_REFRESH_CHANNEL)
            # redis-py 6.x async PubSub.listen() exits when ``channels`` is
            # empty; raw SSUBSCRIBE doesn't populate it, so do it ourselves.
            pubsub.channels[REGISTRY_REFRESH_CHANNEL] = None  # type: ignore[index]
            logger.info("Subscribed to LLM registry refresh channel")

            async for message in pubsub.listen():
                if message and message.get("type") in ("message", "smessage"):
                    logger.debug("LLM registry refresh signal received")
                    try:
                        await on_refresh()
                    except Exception as e:
                        logger.error("Error in registry on_refresh callback: %s", e)

        except asyncio.CancelledError:
            logger.info("LLM registry subscription task cancelled")
            break
        except Exception as e:
            logger.warning("LLM registry subscription error: %s. Retrying in 5s...", e)
            await asyncio.sleep(5)
        finally:
            if pubsub is not None:
                try:
                    await pubsub.aclose()
                except Exception:
                    logger.warning("Failed to close PubSub connection", exc_info=True)
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    logger.warning(
                        "Failed to close shard-pinned Redis connection", exc_info=True
                    )
