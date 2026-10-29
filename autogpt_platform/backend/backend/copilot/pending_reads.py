"""Taking messages off the pending queue and recording that they were read.

Every way a message leaves the queue (a turn's drain, or a resend taking
one by id) records its id in the session's read set in the same Lua script,
so a message is never gone from the queue without being marked read: a
resend of it then answers "already read" rather than running it again. If
the script fails, nothing was popped. A resend that is refused puts its
message back the same way, unread again in the same step.
"""

from typing import Any, cast

from backend.data.redis_client import AsyncRedisClient

# Longer than the queue lives (``PENDING_TTL_SECONDS``): a client can show a
# chip for a message well after a turn read it, and "read" must not expire
# into "unknown" first, which a resend would send again.
READ_TTL_SECONDS = 24 * 3600

# KEYS: queue, read set, held marker.
# ARGV: count to pop, read-set TTL, then the inspected prefix to pop if it
# still matches; with no prefix, pops up to count from the head.
_POP_RECORDING_READS_LUA = """
local popped
if #ARGV > 2 then
    local n = #ARGV - 2
    popped = redis.call('LRANGE', KEYS[1], 0, n - 1)
    if #popped ~= n then return {} end
    for i = 1, n do
        if popped[i] ~= ARGV[i + 2] then return {} end
    end
    redis.call('LTRIM', KEYS[1], n, -1)
else
    popped = redis.call('LPOP', KEYS[1], tonumber(ARGV[1]))
    if not popped then return {} end
end
local ids = {}
for _, item in ipairs(popped) do
    local ok, entry = pcall(cjson.decode, item)
    if ok and type(entry) == 'table' and type(entry['id']) == 'string' then
        ids[#ids + 1] = entry['id']
    end
end
if #ids > 0 then
    redis.call('SADD', KEYS[2], unpack(ids))
    redis.call('EXPIRE', KEYS[2], tonumber(ARGV[2]))
end
redis.call('DEL', KEYS[3])
return popped
"""

# KEYS: queue, read set. ARGV: the stored payload, its id, read-set TTL.
_TAKE_RECORDING_READ_LUA = """
if redis.call('LREM', KEYS[1], 1, ARGV[1]) == 0 then return 0 end
redis.call('SADD', KEYS[2], ARGV[2])
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[3]))
return 1
"""


# KEYS: queue, read set. ARGV: the stored payload, its id, the queue's TTL,
# its enqueue time. It goes back before the first message queued after it,
# so turns still read the queue in the order it was typed. No trim to the
# cap: it was in the queue moments ago, and a push made meanwhile into a
# full queue must not cost it, or any other message, its place.
_PUT_BACK_UNREAD_LUA = """
local placed = false
for _, item in ipairs(redis.call('LRANGE', KEYS[1], 0, -1)) do
    local ok, entry = pcall(cjson.decode, item)
    local at = ok and type(entry) == 'table' and tonumber(entry['enqueued_at'])
    if at and at > tonumber(ARGV[4]) then
        redis.call('LINSERT', KEYS[1], 'BEFORE', item, ARGV[1])
        placed = true
        break
    end
end
if not placed then redis.call('RPUSH', KEYS[1], ARGV[1]) end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
redis.call('SREM', KEYS[2], ARGV[2])
return redis.call('LLEN', KEYS[1])
"""


async def pop_recording_reads(
    redis: AsyncRedisClient,
    *,
    queue_key: str,
    read_key: str,
    held_key: str,
    count: int,
    prefix: list[str] | None = None,
) -> list[str | bytes]:
    """Pop up to *count* entries from the head of the queue, or exactly
    *prefix* if it is still the queue's head, record their ids as read and
    end the hold, all at once. Returns what was popped."""
    if prefix is not None and not prefix:
        return []
    result = await cast(
        "Any",
        redis.eval(
            _POP_RECORDING_READS_LUA,
            3,
            queue_key,
            read_key,
            held_key,
            count,
            READ_TTL_SECONDS,
            *(prefix or []),
        ),
    )
    return list(result or [])


async def take_recording_read(
    redis: AsyncRedisClient,
    *,
    queue_key: str,
    read_key: str,
    payload: str,
    pending_id: str,
) -> bool:
    """Remove the stored *payload* and record *pending_id* as read, at once.
    False when it was no longer in the queue."""
    result = await cast(
        "Any",
        redis.eval(
            _TAKE_RECORDING_READ_LUA,
            2,
            queue_key,
            read_key,
            payload,
            pending_id,
            READ_TTL_SECONDS,
        ),
    )
    return int(result) == 1


async def put_back_unread(
    redis: AsyncRedisClient,
    *,
    queue_key: str,
    read_key: str,
    payload: str,
    pending_id: str,
    enqueued_at: float,
    ttl_seconds: int,
) -> int:
    """Return *payload* to its place in the queue (by *enqueued_at*) and
    forget that *pending_id* was read, at once, so it reads as waiting and
    never as read and gone. Returns the queue's length."""
    result = await cast(
        "Any",
        redis.eval(
            _PUT_BACK_UNREAD_LUA,
            2,
            queue_key,
            read_key,
            payload,
            pending_id,
            ttl_seconds,
            enqueued_at,
        ),
    )
    return int(result)
