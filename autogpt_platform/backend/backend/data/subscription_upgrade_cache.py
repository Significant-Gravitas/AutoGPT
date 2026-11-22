"""User-scoped storage keys for reviewed subscription upgrades."""

from backend.data.redis_client import get_redis_async


async def discard_subscription_upgrade_preview(user_id: str, preview_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(subscription_upgrade_preview_key(user_id, preview_id))


def subscription_upgrade_preview_key(user_id: str, preview_id: str) -> str:
    return f"subscription-upgrade-preview:{user_id}:{preview_id}"
