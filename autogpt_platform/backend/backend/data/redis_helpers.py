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
this module can cover.  Owner-checked token-lock operations live in
:mod:`backend.data.redis_scripts`.
"""

from enum import IntEnum
from typing import Any, cast

from redis_lua_py import Key, redis, script

from backend.data.redis_client import AsyncRedisClient, RedisClient

# ---------------------------------------------------------------------------
# Lua scripts — written as Python and compiled to Lua by redis-lua-py, which
# sends them with EVALSHA.  Exactly ONE authoritative copy per pattern.
# ---------------------------------------------------------------------------


# Compare-and-set on a hash field.  Returns 1 if swapped, 0 if the current
# value didn't match.  Needs Lua because the SET is conditional on a GET
# result (MULTI/EXEC cannot branch on intermediate replies).
@script
def _hash_cas(key: Key, field: str, expected: str, new: str) -> int:
    if redis.hget(key, field) == expected:
        redis.hset(key, field, new)
        return 1
    return 0


# Push to a capped list only when a hash field currently matches the expected
# value. Returns the new list length, or -1 when the guard fails.
@script
def _gated_capped_rpush(
    hash_key: Key,
    list_key: Key,
    hash_field: str,
    expected: str,
    value: str,
    max_len: int,
    ttl_seconds: int,
) -> int:
    if redis.hget(hash_key, hash_field) != expected:
        return -1
    redis.rpush(list_key, value)
    redis.ltrim(list_key, -max_len, -1)
    redis.expire(list_key, ttl_seconds)
    return redis.llen(list_key)


# Exactly-once batch-dispatch claim. Used by the BatchExecutor's walk
# to transition a finished provider batch from ``pending → dispatched``
# in one indivisible step. Without this, the dispatch path is racy:
#
#   await _dispatch(entry, rows)       # side effects fire here
#   await remove_pending(batch_id)     # ← any crash / slow write here
#                                      #    leaves the batch in pending,
#                                      #    next walk re-dispatches.
#
# The atomic claim SETs a per-batch tombstone STRING key BEFORE the
# walker calls _dispatch, so a re-entry will be refused even when the
# pending hash hasn't been HDEL'd yet. One tombstone key per batch_id
# (``SET ... NX EX ttl``) means each tombstone self-expires its own
# TTL after its own dispatch. The previous design — one shared SET
# whose TTL was re-applied on every claim — never aged members out:
# as long as one batch dispatched within the window, the whole set
# lived forever and grew unbounded.
#
# Cluster safety: this script touches two keys (pending hash +
# tombstone), so both MUST share a Redis Cluster hash tag (e.g. both
# prefixed with ``{llm:batch}:``) to land on the same slot — required
# for multi-key Lua under cluster mode. The pending hash is a
# single-slot structure anyway, so colocating the small, self-expiring
# tombstones on its slot costs nothing and buys full claim+HDEL
# atomicity (no claimed-but-still-pending window for walkers to skip).
#
# Returns 1 when this caller won the claim (proceed with dispatch),
# 0 when another walker already dispatched (skip silently).
@script
def _claim_batch_dispatch(
    pending_key: Key, tombstone_key: Key, batch_id: str, ttl_seconds: int
) -> int:
    if redis.set(tombstone_key, "1", "NX", "EX", ttl_seconds):
        redis.hdel(pending_key, batch_id)
        return 1
    return 0


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
# Scores are passed as ``str`` so a timestamp is written exactly as given,
# not rounded through a Lua number.
@script
def _try_acquire_concurrency_slot(
    pool_key: Key,
    slot_id: str,
    score: str,
    stale_before_score: str,
    capacity: int,
    ttl_seconds: int,
) -> int:
    redis.zremrangebyscore(pool_key, "-inf", stale_before_score)
    if redis.zscore(pool_key, slot_id) is not None:
        redis.zadd(pool_key, score, slot_id)
        redis.expire(pool_key, ttl_seconds)
        return 2
    if redis.zcard(pool_key) >= capacity:
        return 0
    redis.zadd(pool_key, score, slot_id)
    redis.expire(pool_key, ttl_seconds)
    return 1


def as_str(value: bytes | str | None) -> str | None:
    """Coerce a value read back from Redis to ``str``.

    Our clients decode responses, but redis-py types every read as
    ``bytes | str | None``; this keeps that knowledge in one place instead of
    an ``isinstance`` at every call site.
    """
    if value is None:
        return None
    return value if isinstance(value, str) else value.decode()


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
    length = await _gated_capped_rpush(
        redis,
        hash_key=hash_key,
        list_key=list_key,
        hash_field=hash_field,
        expected=expected,
        value=value,
        max_len=max_len,
        ttl_seconds=ttl_seconds,
    )
    return None if length < 0 else length


async def claim_batch_dispatch_atomic(
    redis: AsyncRedisClient,
    *,
    pending_key: str,
    dispatched_key_prefix: str,
    batch_id: str,
    ttl_seconds: int,
) -> bool:
    """Atomically claim the right to dispatch results for ``batch_id``.

    Used by the BatchExecutor walk loop. The race we're closing:

        await _dispatch(entry, rows)        # side effects fire here
        await remove_pending(batch_id)      # gap → re-dispatch on
                                            #     crash / contention

    The tombstone is one STRING key per batch —
    ``f"{dispatched_key_prefix}:{batch_id}"`` — set with ``NX EX`` so
    each tombstone expires individually ``ttl_seconds`` after its own
    dispatch (a shared set with one refreshed TTL would never age
    members out and grow unbounded). The Lua script SETs the tombstone
    and, only when it won, HDELs the pending entry, all in one
    indivisible step. Returns ``True`` when this caller won the claim
    (proceed with dispatch), ``False`` when another walker already
    dispatched (skip silently).

    ``pending_key`` and ``dispatched_key_prefix`` MUST share a Redis
    Cluster hash tag (e.g. both prefixed with ``"{llm:batch}:"``) so
    the pending hash and every tombstone land on the same slot —
    required for multi-key Lua under cluster mode. Same-slot keys keep
    the claim + HDEL fully atomic, so a crash can never leave a
    claimed-but-still-pending entry.

    ``ttl_seconds`` should comfortably exceed the longest possible
    in-flight batch lifetime so stale tombstones cannot let a very
    late re-poll cause a re-dispatch. 7 days is fine today (Anthropic
    batch SLA is 24h, max batch lifetime cap is 24h); raise it if the
    provider window widens.
    """
    result = await _claim_batch_dispatch(
        redis,
        pending_key=pending_key,
        tombstone_key=f"{dispatched_key_prefix}:{batch_id}",
        batch_id=batch_id,
        ttl_seconds=ttl_seconds,
    )
    return bool(result)


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
    result = await _try_acquire_concurrency_slot(
        redis,
        pool_key=pool_key,
        slot_id=slot_id,
        score=str(score),
        stale_before_score=str(stale_before_score),
        capacity=capacity,
        ttl_seconds=ttl_seconds,
    )
    return SlotAdmission(result)


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
    result = await _hash_cas(redis, key=key, field=field, expected=expected, new=new)
    return result == 1
