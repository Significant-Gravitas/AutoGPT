"""Remove the account's cached skill metadata, including deleted experts."""

from uuid import UUID

from backend.data.redis_client import AsyncRedisClient, get_redis_async


async def delete_skills_cache(user_id: str) -> int:
    owner = str(UUID(user_id))
    redis = await get_redis_async()
    patterns = (
        (f"copilot:skills_index:*:{owner}", 3),
        (f"copilot:skills_index:*:{owner}:expert:*", 3),
        (f"copilot:skills_heal_backoff:{owner}:expert:*", 2),
        (f"copilot:skills_update_backoff:{owner}:*", 2),
    )
    deleted = 0
    for pattern, owner_field in patterns:
        async for key in redis.scan_iter(
            match=pattern, count=100, target_nodes=AsyncRedisClient.PRIMARIES
        ):
            # A Redis glob can cross colons and match another owner's expert ID.
            if key.split(":")[owner_field] != owner:
                continue
            # Index entries also hold rebuild claims; deleting them prevents
            # an existing rebuild from publishing its old snapshot afterwards.
            # Delete one key at a time because owners span Redis Cluster slots.
            deleted += await redis.delete(key)
    return deleted
