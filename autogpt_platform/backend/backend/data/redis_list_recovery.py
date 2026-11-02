"""Atomic restoration of previously admitted JSON list entries."""

from typing import Any, cast

from backend.data.redis_client import AsyncRedisClient

_PREPEND_JSON_ITEMS_LUA = """
local ttl = math.max(redis.call('PTTL', KEYS[1]), tonumber(ARGV[1]) * 1000)
local restored = {}
local ids = {}
for i = 2, #ARGV do
    local entry = cjson.decode(ARGV[i])
    if not ids[entry.id] then
        ids[entry.id] = true
        table.insert(restored, ARGV[i])
    end
end
for _, value in ipairs(redis.call('LRANGE', KEYS[1], 0, -1)) do
    local ok, entry = pcall(cjson.decode, value)
    if ok and type(entry) == 'table' and ids[entry.id] then
        redis.call('LREM', KEYS[1], 0, value)
    end
end
for i = #restored, 1, -1 do
    redis.call('LPUSH', KEYS[1], restored[i])
end
redis.call('PEXPIRE', KEYS[1], ttl)
return redis.call('LLEN', KEYS[1])
"""


async def prepend_json_items_by_id(
    redis: AsyncRedisClient, key: str, values: list[str], *, ttl_seconds: int
) -> int:
    """Restore JSON objects with string IDs at the front, once and in order.

    Recovery bypasses admission capacity because every item was already accepted.
    Existing copies are replaced atomically and the list's TTL never shrinks.
    """
    return int(
        await cast(
            Any, redis.eval(_PREPEND_JSON_ITEMS_LUA, 1, key, ttl_seconds, *values)
        )
    )
