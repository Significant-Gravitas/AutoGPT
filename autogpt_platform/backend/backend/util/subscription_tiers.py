import asyncio
import hashlib
import logging
import uuid
from collections.abc import Callable
from typing import Any, cast

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS = 3600
_GENERATION_KEY_PREFIX = "subscription-tier-cache:generation:"


class SubscriptionTierCacheInvalidationError(RuntimeError):
    """Raised after a committed tier write when a security cache was not fenced."""

    def __init__(self, failed_caches: tuple[str, ...]):
        self.failed_caches = failed_caches
        super().__init__(
            "Subscription tier was updated, but cache invalidation failed for: "
            + ", ".join(failed_caches)
        )


def _generation_key(user_id: str) -> str:
    # Normalize the key length and avoid plaintext identifiers in Redis key scans.
    # This digest is namespacing, not a confidentiality boundary.
    digest = hashlib.sha256(user_id.encode()).hexdigest()
    return f"{_GENERATION_KEY_PREFIX}{digest}"


async def _rotate_subscription_tier_generation(user_id: str) -> None:
    redis = await get_redis_async()
    # The redis client protocol omits keyword overloads supported at runtime.
    written = await cast(Any, redis.set)(
        _generation_key(user_id),
        uuid.uuid4().hex,
        ex=SUBSCRIPTION_TIER_GENERATION_TTL_SECONDS,
    )
    if not written:
        raise RuntimeError("Redis did not acknowledge the generation update")


async def _run_dual_cache_invalidation(
    user_id: str,
    legacy_cache_delete: Callable[[str], object],
) -> None:
    failures: list[str] = []

    try:
        await _rotate_subscription_tier_generation(user_id)
    except Exception:
        failures.append("generation")
        logger.exception(
            "Failed to rotate subscription-tier generation for user %s",
            user_id[:8],
        )

    try:
        legacy_cache_delete(user_id)
    except Exception:
        failures.append("legacy")
        logger.exception(
            "Failed to evict legacy subscription-tier cache for user %s",
            user_id[:8],
        )

    if failures:
        raise SubscriptionTierCacheInvalidationError(tuple(failures))


def invalidate_subscription_tier_auxiliary_caches(user_id: str) -> None:
    """Best-effort eviction for non-authoritative subscription-tier views."""
    # Local imports keep subscription-tier policy below its billing consumers.
    from backend.data.credit import get_pending_subscription_change, get_user_by_id

    for cache_delete in (
        get_user_by_id.cache_delete,  # type: ignore[attr-defined]
        get_pending_subscription_change.cache_delete,  # type: ignore[attr-defined]
    ):
        try:
            cache_delete(user_id)
        except Exception:
            logger.exception(
                "Subscription tier updated for user %s, but an auxiliary cache "
                "could not be cleared",
                user_id[:8],
            )


async def invalidate_subscription_tier_caches(
    user_id: str,
    legacy_cache_delete: Callable[[str], object],
) -> None:
    """Fence future tier readers and evict the cache used by deployed readers.

    Both invalidations finish even when the caller is cancelled after its DB
    commit. The caller's cancellation is restored once the cache fence ends.
    """
    invalidation = asyncio.create_task(
        _run_dual_cache_invalidation(user_id, legacy_cache_delete)
    )
    cancellation: asyncio.CancelledError | None = None
    while not invalidation.done():
        try:
            await asyncio.shield(invalidation)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except Exception:
            # ``invalidation.result()`` below preserves and raises this error.
            break

    try:
        invalidation.result()
    except Exception:
        if cancellation is not None:
            logger.exception(
                "Subscription-tier cache invalidation failed while caller was "
                "cancelled for user %s",
                user_id[:8],
            )
            raise cancellation
        raise

    if cancellation is not None:
        raise cancellation
