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

from enum import IntEnum
from typing import Any, cast

from backend.data.redis_client import AsyncRedisClient, RedisClient

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

# Push to a capped list only when a hash field currently matches the expected
# value. Returns the new list length, or -1 when the guard fails.
#
#   KEYS[1]  hash key
#   KEYS[2]  list key
#   ARGV[1]  hash field
#   ARGV[2]  expected current value
#   ARGV[3]  list value
#   ARGV[4]  max list length
#   ARGV[5]  list TTL seconds
_GATED_CAPPED_RPUSH_LUA = """
local current = redis.call('HGET', KEYS[1], ARGV[1])
if current ~= ARGV[2] then
    return -1
end
redis.call('RPUSH', KEYS[2], ARGV[3])
redis.call('LTRIM', KEYS[2], -tonumber(ARGV[4]), -1)
redis.call('EXPIRE', KEYS[2], tonumber(ARGV[5]))
return redis.call('LLEN', KEYS[2])
"""

# Atomically: sweep stale slots, refresh an existing slot's claim or add
# a new slot iff the pool is under capacity, then set the pool key's TTL.
# Returns 1 when a NEW slot was admitted, 2 when an EXISTING slot's claim
# was refreshed (no change to slot count), 0 when reservation was refused
# (pool full and slot wasn't already held).
#
# The new-vs-refreshed distinction lets callers tell apart a "fresh
# admission" (caller now owns the slot's release) from a "re-entrant
# touch" (someone else owns the release, caller is just bumping the
# heartbeat). Without it, a refresh path would `release_turn_slot` on
# context-manager exit and prematurely free a slot held by another
# concurrent caller for the same session_id.
#
# Used for distributed per-actor concurrency caps. A "pool" is a Redis
# sorted set whose members are active slot ids and whose scores are the
# slot's reservation timestamp. Stale slots (owner crashed without
# release) are reclaimed by the sweep so a one-time leak cannot
# permanently consume capacity.
#
#   KEYS[1]  pool key (sorted set)
#   ARGV[1]  slot id (member)
#   ARGV[2]  reservation score (typically now)
#   ARGV[3]  stale-cutoff score (slots with score <= this are dropped)
#   ARGV[4]  capacity (max concurrent slots before reservation is refused)
#   ARGV[5]  TTL seconds applied to the pool key on every successful reserve
_TRY_ACQUIRE_CONCURRENCY_SLOT_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[3])
local existing = redis.call('ZSCORE', KEYS[1], ARGV[1])
if existing then
    redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
    redis.call('EXPIRE', KEYS[1], ARGV[5])
    return 2
end
local count = redis.call('ZCARD', KEYS[1])
if count >= tonumber(ARGV[4]) then
    return 0
end
redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
"""


async def incr_with_ttl(
    redis: AsyncRedisClient,
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
    redis: RedisClient,
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
    redis: AsyncRedisClient,
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


async def capped_rpush_if_hash_field(
    redis: AsyncRedisClient,
    *,
    hash_key: str,
    hash_field: str,
    expected: str,
    list_key: str,
    value: str,
    max_len: int,
    ttl_seconds: int,
) -> int | None:
    """Atomically RPUSH to a bounded list iff a hash field matches.

    Returns the new list length when the push happens, or ``None`` when the
    hash field does not currently match ``expected``.
    """
    result = await cast(
        "Any",
        redis.eval(
            _GATED_CAPPED_RPUSH_LUA,
            2,
            hash_key,
            list_key,
            hash_field,
            expected,
            value,
            str(max_len),
            str(ttl_seconds),
        ),
    )
    length = int(result)
    return None if length < 0 else length


class SlotAdmission(IntEnum):
    """Result of :func:`try_acquire_concurrency_slot`.

    Distinguishes a fresh admission from a re-entrant refresh so callers
    that automatically release on exit (e.g. context managers) only do
    so for slots they actually admitted, not slots they just touched.
    """

    REJECTED = 0  # pool was full and slot wasn't already held
    ADMITTED = 1  # slot newly added to the pool
    REFRESHED = 2  # slot was already in the pool; score bumped only


async def try_acquire_concurrency_slot(
    redis: AsyncRedisClient,
    *,
    pool_key: str,
    slot_id: str,
    score: float,
    capacity: int,
    stale_before_score: float,
    ttl_seconds: int,
) -> SlotAdmission:
    """Atomically reserve one of *capacity* slots in a Redis-backed pool.

    Use this whenever you need a distributed concurrency cap — "this
    actor may have at most N of X in flight at the same time". The pool
    is a sorted set whose members are active slot ids and whose scores
    are reservation timestamps; on every call we:

    1. Sweep slots with ``score <= stale_before_score`` — slots whose
       holder crashed without releasing don't permanently consume capacity.
    2. If ``slot_id`` is already in the pool, refresh its score
       (re-reservation is idempotent — same holder, same logical slot)
       and return :attr:`SlotAdmission.REFRESHED`.
    3. Otherwise, reserve only if the post-sweep slot count is below
       ``capacity`` and return :attr:`SlotAdmission.ADMITTED`.
    4. On a successful reserve, set ``ttl_seconds`` on the pool key as a
       belt-and-braces TTL for the case where the sweep ever stops.

    Returns the :class:`SlotAdmission` outcome — ``ADMITTED`` (newly
    added), ``REFRESHED`` (already held, score bumped), or ``REJECTED``
    (pool full).

    Why a Lua script: the reservation is conditional on the post-sweep
    count, and ``MULTI/EXEC`` cannot branch on intermediate replies.
    Without atomicity, two concurrent callers both read ``count =
    capacity - 1`` and both add, ending up over capacity.

    Redis Cluster: only ``KEYS[1]`` is touched, so callers are free to
    hash-tag *pool_key* (e.g. ``foo:{user_id}``) to colocate the pool
    on one shard without CROSSSLOT issues.
    """
    result = await cast(
        "Any",
        redis.eval(
            _TRY_ACQUIRE_CONCURRENCY_SLOT_LUA,
            1,
            pool_key,
            slot_id,
            str(score),
            str(stale_before_score),
            str(capacity),
            str(ttl_seconds),
        ),
    )
    return SlotAdmission(int(result))


async def hash_compare_and_set(
    redis: AsyncRedisClient,
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
