"""Thread subscription tracking.

When the bot creates a thread in response to an @mention, we record the
thread ID so subsequent messages in it don't require another mention.
Subscriptions live in Redis with a 7-day TTL — stale threads age out
automatically.
"""

from backend.data.redis_client import get_redis_async

THREAD_SUBSCRIPTION_TTL = 7 * 86400  # 7 days


def _key(platform: str, thread_id: str) -> str:
    return f"copilot-bot:thread:{platform}:{thread_id}"


async def is_subscribed(platform: str, thread_id: str) -> bool:
    redis = await get_redis_async()
    return bool(await redis.get(_key(platform, thread_id)))


async def subscribe(platform: str, thread_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(_key(platform, thread_id), "1", ex=THREAD_SUBSCRIPTION_TTL)
