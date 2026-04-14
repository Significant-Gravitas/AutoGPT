"""Graphiti client management with per-group_id isolation and LRU caching."""

import asyncio
import logging
import re

from cachetools import TTLCache

from .config import graphiti_config

logger = logging.getLogger(__name__)

_GROUP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_GROUP_ID_LEN = 128

_client_cache: TTLCache | None = None
_cache_lock = asyncio.Lock()


def derive_group_id(user_id: str) -> str:
    """Derive a deterministic, injection-safe group_id from a user_id.

    Strips to ``[a-zA-Z0-9_-]``, enforces max length, and prefixes with
    ``user_``.  Raises if sanitization changed the input.
    """
    if not user_id:
        raise ValueError("user_id must be non-empty to derive group_id")

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", user_id)[:_MAX_GROUP_ID_LEN]
    if not safe_id:
        raise ValueError(
            f"user_id '{user_id[:32]}...' yields empty group_id after sanitization"
        )

    if safe_id != user_id:
        raise ValueError(
            f"user_id contains invalid characters for group_id derivation "
            f"(original length={len(user_id)}, sanitized='{safe_id[:32]}'). "
            f"Only [a-zA-Z0-9_-] are allowed."
        )

    group_id = f"user_{safe_id}"
    if not _GROUP_ID_PATTERN.match(group_id):
        raise ValueError(f"Generated group_id '{group_id}' fails validation")

    return group_id


def _close_client_driver(client) -> None:
    """Best-effort close of a Graphiti client's graph driver.

    Called on cache eviction (TTL expiry or manual pop) to prevent
    leaked FalkorDB connections.  Runs the async ``driver.close()``
    in a fire-and-forget task if an event loop is running, otherwise
    logs and moves on.
    """
    driver = getattr(client, "graph_driver", None) or getattr(client, "driver", None)
    if driver is None or not hasattr(driver, "close"):
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(driver.close())
    except RuntimeError:
        logger.debug("No running event loop — skipping driver.close() on eviction")


class _EvictingTTLCache(TTLCache):
    """TTLCache that closes Graphiti drivers on TTL expiry and capacity eviction.

    Overrides ``expire()`` (not ``__delitem__``) per cachetools maintainer
    guidance — ``expire()`` is the only hook that fires for TTL-expired items
    since the internal expiry path uses ``Cache.__delitem__`` directly,
    bypassing subclass overrides.  ``popitem()`` handles capacity eviction.
    See https://github.com/tkem/cachetools/issues/205.
    """

    def expire(self, time=None):
        expired = super().expire(time)
        for _key, client in expired:
            _close_client_driver(client)
        return expired

    def popitem(self):
        key, client = super().popitem()
        _close_client_driver(client)
        return key, client


def _get_cache() -> TTLCache:
    global _client_cache
    if _client_cache is None:
        _client_cache = _EvictingTTLCache(
            maxsize=graphiti_config.client_cache_maxsize,
            ttl=graphiti_config.client_cache_ttl,
        )
    return _client_cache


async def get_graphiti_client(group_id: str):
    """Return a Graphiti client scoped to the given group_id.

    Each group_id gets its own ``Graphiti`` instance to prevent the
    ``self.driver`` mutation race condition when different groups are
    accessed concurrently.  Instances are cached with a TTL to bound
    memory usage.

    Returns a ``graphiti_core.Graphiti`` instance.
    """
    from graphiti_core import Graphiti
    from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client import LLMConfig, OpenAIClient

    from .falkordb_driver import AutoGPTFalkorDriver

    cache = _get_cache()

    async with _cache_lock:
        if group_id in cache:
            return cache[group_id]

        llm_config = LLMConfig(
            api_key=graphiti_config.resolve_llm_api_key(),
            model=graphiti_config.llm_model,
            small_model=graphiti_config.llm_model,  # avoid gpt-4.1-nano dedup hallucination (#760)
            base_url=graphiti_config.resolve_llm_base_url(),
        )
        llm_client = OpenAIClient(config=llm_config)

        embedder_config = OpenAIEmbedderConfig(
            api_key=graphiti_config.resolve_embedder_api_key(),
            embedding_model=graphiti_config.embedder_model,
            base_url=graphiti_config.resolve_embedder_base_url(),
        )
        embedder = OpenAIEmbedder(config=embedder_config)

        graph_driver = AutoGPTFalkorDriver(
            host=graphiti_config.falkordb_host,
            port=graphiti_config.falkordb_port,
            password=graphiti_config.falkordb_password or None,
            database=group_id,
        )
        client = Graphiti(
            llm_client=llm_client,
            embedder=embedder,
            graph_driver=graph_driver,
            max_coroutines=graphiti_config.semaphore_limit,
        )

        cache[group_id] = client
        return client


async def evict_client(group_id: str) -> None:
    """Remove a cached client and close its driver connection."""
    cache = _get_cache()
    # pop() may return None for expired or missing keys.
    # _EvictingTTLCache.expire() handles TTL-expired cleanup separately.
    client = cache.pop(group_id, None)
    if client is not None:
        driver = getattr(client, "graph_driver", None) or getattr(
            client, "driver", None
        )
        if driver and hasattr(driver, "close"):
            try:
                await driver.close()
            except Exception:
                logger.debug("Failed to close driver for %s", group_id, exc_info=True)
