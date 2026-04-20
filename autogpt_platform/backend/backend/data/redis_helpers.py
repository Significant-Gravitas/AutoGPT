"""Shared Redis helpers for patterns that need client-side atomicity.

Redis is a wonderful key-value store but has ergonomic gaps that every
app ends up papering over the same way — usually as ad-hoc Lua EVALs or
raw pipelines scattered across the codebase.  This module collects the
patterns we actually use into a single place:

- :func:`incr_with_ttl` — atomic INCR + set TTL (Redis has no native
  "increment and set TTL on first bump" command).  Implemented with
  ``pipeline(transaction=True)`` (MULTI/EXEC) — no Lua needed.
- :func:`capped_rpush` — push to a bounded list (RPUSH + LTRIM + EXPIRE +
  LLEN) atomically.  Pipeline-based.
- :func:`hash_compare_and_set` — set a hash field only if its current
  value matches an expected one.  Genuinely needs Lua because the
  condition depends on the current value (pipeline can't branch).

Everything sharable lives here.  If a new Lua script is tempting in
application code, add a helper here first — callers should not touch
``redis.eval`` / ``pipeline(transaction=True)`` directly for anything
this module can cover.
"""

from typing import Any, cast

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

# ---------------------------------------------------------------------------
# Lua scripts — registered centrally so there is exactly ONE authoritative
# copy per pattern and ``SCRIPT LOAD`` can be amortised in future if needed.
# ---------------------------------------------------------------------------

# Compare-and-set on a hash field.  Returns 1 if swapped, 0 if the current
# value didn't match.  Needs Lua because the SET is conditional on a GET
# result (MULTI/EXEC cannot branch on intermediate replies).
#
#   KEYS[1]  hash key
#   ARGV[1]  hash field
#   ARGV[2]  expected current value
#   ARGV[3]  new value
_HASH_CAS_LUA = """
local current = redis.call('HGET', KEYS[1], ARGV[1])
if current == ARGV[2] then
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
    return 1
end
return 0
"""


async def incr_with_ttl(
    redis: AsyncRedis,
    key: str,
    ttl_seconds: int,
    *,
    reset_ttl_on_bump: bool = False,
) -> int:
    """Atomically increment *key* and set its TTL.

    Returns the new counter value.

    Args:
        redis: AsyncRedis client.
        key: Counter key.
        ttl_seconds: TTL to apply to the key.
        reset_ttl_on_bump: When ``False`` (default, fixed-window), the TTL is
            only set on the first bump in a window — subsequent bumps leave
            the existing TTL alone so the window genuinely expires
            ``ttl_seconds`` after the first push.  When ``True``
            (sliding-window), every bump refreshes the TTL.

    Atomicity: uses MULTI/EXEC so the counter can never end up without a
    TTL (the classic "process dies between INCR and EXPIRE" orphan).
    """
    pipe = redis.pipeline(transaction=True)
    pipe.incr(key)
    # EXPIRE ... NX = "only set TTL if none exists" (Redis 7+).  In
    # reset_ttl_on_bump mode, unconditional EXPIRE refreshes every bump.
    if reset_ttl_on_bump:
        pipe.expire(key, ttl_seconds)
    else:
        pipe.expire(key, ttl_seconds, nx=True)
    results = await pipe.execute()
    return int(results[0])


def incr_with_ttl_sync(
    redis: Redis,
    key: str,
    ttl_seconds: int,
    *,
    reset_ttl_on_bump: bool = False,
) -> int:
    """Sync variant of :func:`incr_with_ttl` — same semantics."""
    pipe = redis.pipeline(transaction=True)
    pipe.incr(key)
    if reset_ttl_on_bump:
        pipe.expire(key, ttl_seconds)
    else:
        pipe.expire(key, ttl_seconds, nx=True)
    results = pipe.execute()
    return int(results[0])


async def capped_rpush(
    redis: AsyncRedis,
    key: str,
    value: str,
    *,
    max_len: int,
    ttl_seconds: int,
) -> int:
    """Atomically RPUSH *value*, trim to *max_len*, set TTL, and return LLEN.

    Returns the list length after the push+trim.

    Atomicity: MULTI/EXEC so a concurrent LPOP can never observe the
    list transiently over ``max_len``.

    Use this for bounded producer/consumer buffers where the newest
    entries matter most (LTRIM from the left, keeping the tail).
    """
    pipe = redis.pipeline(transaction=True)
    pipe.rpush(key, value)
    pipe.ltrim(key, -max_len, -1)
    pipe.expire(key, ttl_seconds)
    pipe.llen(key)
    results = cast("list[Any]", await pipe.execute())
    return int(results[-1])


async def hash_compare_and_set(
    redis: AsyncRedis,
    key: str,
    field: str,
    *,
    expected: str,
    new: str,
) -> bool:
    """Atomically set ``HSET key field new`` iff current value == *expected*.

    Returns ``True`` if the swap happened, ``False`` otherwise.

    Use this for idempotent state transitions (e.g. mark a task as
    ``completed`` only when it is still ``running``, so a late retry
    cannot clobber an earlier terminal state).  Genuinely needs Lua
    because the write is conditional on the read result — MULTI/EXEC
    cannot branch on intermediate replies.
    """
    result = await cast(
        "Any",
        redis.eval(_HASH_CAS_LUA, 1, key, field, expected, new),
    )
    return int(result) == 1
