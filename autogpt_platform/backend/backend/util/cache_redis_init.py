"""
Initialize Redis clients for the cache system.

This module bridges the gap between backend's Redis client and autogpt_libs cache system.
"""

import logging

from autogpt_libs.utils.cache import set_redis_client_provider

logger = logging.getLogger(__name__)


def initialize_cache_redis():
    """
    Initialize the cache system with backend's Redis clients.

    This function sets up the cache system to use the backend's
    existing Redis connection instead of creating its own.
    """
    try:
        from backend.data.redis_client import HOST, PASSWORD, PORT

        # Create provider functions that create new binary-mode clients
        def get_sync_redis_for_cache():
            """Get sync Redis client configured for cache (binary mode)."""
            try:
                from redis import Redis

                client = Redis(
                    host=HOST,
                    port=PORT,
                    password=PASSWORD,
                    decode_responses=False,  # Binary mode for pickle
                )
                # Test the connection
                client.ping()
                return client
            except Exception as e:
                logger.warning(f"Failed to get Redis client for cache: {e}")
            return None

        async def get_async_redis_for_cache():
            """Get async Redis client configured for cache (binary mode)."""
            try:
                from redis.asyncio import Redis as AsyncRedis

                client = AsyncRedis(
                    host=HOST,
                    port=PORT,
                    password=PASSWORD,
                    decode_responses=False,  # Binary mode for pickle
                )
                # Test the connection
                await client.ping()
                return client
            except Exception as e:
                logger.warning(f"Failed to get async Redis client for cache: {e}")
            return None

        # Set the providers in the cache system
        set_redis_client_provider(
            sync_provider=get_sync_redis_for_cache,
            async_provider=get_async_redis_for_cache,
        )

        logger.info("Cache system initialized with backend Redis clients")

    except ImportError as e:
        logger.warning(f"Could not import Redis clients, cache will use fallback: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize cache Redis: {e}")


# Auto-initialize when module is imported
initialize_cache_redis()
