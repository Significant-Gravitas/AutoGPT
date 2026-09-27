"""Redis SETNX advisory lock for the dream pass.

One in-flight dream per user at a time. Bails if the lock is already
held — the caller surfaces that as ``skipped=True`` in
``DreamPassResult`` so the admin UI can render "another dream is
already running" rather than failing.

The lock value is a per-acquire ownership token (uuid4); every release
is a single-key Lua compare-and-delete on that token and every TTL
extend is a single-key Lua compare-and-extend. A blind delete (or a
blind ``SET XX``) is NOT safe: if a pass outlives its TTL (slow sync
pass, or a batch callback landing inside the thin margin between the
batch lifetime and ``BATCH_LOCK_TTL_SECONDS``), the key may already
belong to a *newer* pass — deleting or overwriting it would break that
pass's ownership and let a third concurrent pass start. Prod Redis runs
in cluster mode, so everything here stays single-key: SET NX plus
single-key Lua scripts, no multi-key scripts and no cross-key
transactions.

The sync-path TTL is 1800 s (30 min), sized to the scheduler's job
timeout; the batch path extends it (see ``BATCH_LOCK_TTL_SECONDS``). A pass
renews its lock at each step, and its DreamPass row records the new expiry
as the pass's lease (``lease.py``).

Right before it applies, a pass proves the key is still its own by renewing
it (``extend``), and the sync pass reads it first (``held``): a lock that
lapsed may already be a newer pass's, and only the holder may write the
scope's graph.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, cast

from backend.copilot.graphiti.scope import MemoryScope

logger = logging.getLogger(__name__)

# Sync-path lock TTL (30 min).
DEFAULT_LOCK_TTL_SECONDS = 1800

# Batch path: one phase's batch stays in flight up to the BatchExecutor's
# MAX_BATCH_LIFETIME_SECONDS (24h). The lock is extended to this TTL at submit
# and again at each callback whose phase landed, and once more before apply,
# so it covers the phase batch in flight (not the whole chain of three) with
# 10 minutes to spare, and apply, which runs in the last callback, is still
# covered by "one dream per user". The callbacks release it when the pass
# ends; the TTL is the backstop for a pass that died.
BATCH_LOCK_TTL_SECONDS = 24 * 60 * 60 + 600

# How long the ownership read right before apply may take.
LOCK_CHECK_TIMEOUT_SECONDS = 2.0

# Compare-and-delete: only the holder whose token still matches the stored
# value may delete the key. Single-key Lua routes on Redis Cluster.
_UNLOCK_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)

# Compare-and-extend: only the holder whose token still matches the stored
# value may stretch the TTL. Same single-key Lua pattern as
# ``_UNLOCK_SCRIPT`` — a blind ``SET XX`` would overwrite a *newer* pass's
# token (and TTL) when our lock expired and was re-acquired mid-pass.
_EXTEND_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("expire", KEYS[1], ARGV[2]) else return 0 end'
)


class DreamLockHeld(Exception):
    """Raised when another process is already running a dream for this user."""

    def __init__(self, user_id: str) -> None:
        super().__init__(f"Dream pass already in flight for user {user_id[:12]}")
        self.user_id = user_id


class DreamLockLostError(Exception):
    """Raised when a held dream lock could not be renewed — the key expired
    (and may already belong to a newer pass), so continuing to write to the
    user's graph risks two passes mutating it concurrently."""

    def __init__(self, user_id: str) -> None:
        super().__init__(
            f"Dream lock lost before renewal for user {user_id[:12]} — "
            "aborting before further graph writes"
        )
        self.user_id = user_id


class DreamLockHandle:
    """Yielded by ``dream_lock``. Lets the async batch path hand lock
    ownership to the callback chain: ``extend`` stretches the TTL to the
    batch lifetime and ``disown`` stops the context manager from releasing on
    exit, so the lock survives until ``release_dream_lock`` runs in the batch
    callback (or the TTL expires).

    ``token`` is the ownership value stored at the lock key. Release is a
    compare-and-delete on it, so the batch path persists the token alongside
    its per-pass state (the input bundle) for the callback to read back hours
    later."""

    def __init__(self, redis: Any, key: str, user_id: str, token: str) -> None:
        self._redis = redis
        self._key = key
        self.user_id = user_id
        self.token = token
        self.release_on_exit = True

    def disown(self) -> None:
        self.release_on_exit = False

    async def extend(self, ttl_seconds: int) -> bool:
        """Stretch the lock TTL only while the key still holds our token.

        Single-key Lua compare-and-extend (mirrors ``_UNLOCK_SCRIPT``): a
        blind ``SET XX`` succeeds against ANY existing value, so if our
        lock expired and a newer pass re-acquired the key, it would
        hijack that pass's token and stretch its TTL. The compare also
        refuses to recreate an expired key — an expired lock is never
        resurrected.

        Returns ``True`` when the TTL was extended, ``False`` when
        ownership was lost (key expired or re-acquired by a newer pass).
        Callers about to perform writes that assume exclusive graph
        ownership must abort on ``False`` (see ``DreamLockLostError``).
        """
        # ``cast`` because redis-py's stubs type ``eval`` as a bare
        # ``str`` — same workaround as the release paths below.
        extended = await cast(
            "Any",
            self._redis.eval(
                _EXTEND_SCRIPT, 1, self._key, self.token, str(ttl_seconds)
            ),
        )
        if not extended:
            logger.warning(
                "Dream lock for user %s expired before extend — "
                "not resurrecting it; another pass may already own the key",
                self.user_id[:12],
            )
        return bool(extended)

    async def held(self) -> bool:
        """Whether the key still holds this handle's token: one GET, raising
        when Redis does not answer within ``LOCK_CHECK_TIMEOUT_SECONDS``."""
        current = await asyncio.wait_for(
            self._redis.get(self._key), timeout=LOCK_CHECK_TIMEOUT_SECONDS
        )
        return _as_text(current) == self.token


@asynccontextmanager
async def dream_lock(
    scope: MemoryScope,
    ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS,
    token: str | None = None,
):
    """Acquire a per-user advisory lock for the dream pass.

    Raises ``DreamLockHeld`` if another process holds the lock.
    The lock is released on context exit via compare-and-delete on this
    acquire's ownership token — a late exit can't delete a newer pass's
    lock. On crash the TTL provides a fallback release after
    ``ttl_seconds``. *token*, minted by the pass so its record carries the
    lease before the lock is taken, is the ownership value (uuid4 if not).
    """
    # Lazy import so this module is cheap to import in tests that mock redis.
    from backend.data.redis_client import get_redis_async

    user_id = scope.owner_user_id
    redis = await get_redis_async()
    key = scope.redis_key("dream_lock")
    token = token or str(uuid.uuid4())

    acquired = await redis.set(key, token, nx=True, ex=ttl_seconds)
    if not acquired:
        raise DreamLockHeld(user_id)

    logger.info("Acquired dream lock for user %s (ttl=%ds)", user_id[:12], ttl_seconds)
    handle = DreamLockHandle(redis, key, user_id, token)
    try:
        yield handle
    finally:
        if not handle.release_on_exit:
            # Disowned by the batch path — the callback (or the TTL) owns
            # release now, so the lock spans the async batch lifetime.
            logger.info("Dream lock for user %s handed to batch callback", user_id[:12])
        else:
            try:
                # ``cast`` because redis-py's stubs type ``eval`` as a bare
                # ``str`` — same workaround as ``data/redis_helpers.py``.
                deleted = await cast("Any", redis.eval(_UNLOCK_SCRIPT, 1, key, token))
                if deleted:
                    logger.debug("Released dream lock for user %s", user_id[:12])
                else:
                    logger.warning(
                        "Dream lock for user %s no longer held our token at "
                        "release — expired and re-acquired by another pass; "
                        "leaving the new holder's lock alone",
                        user_id[:12],
                    )
            except Exception:
                # If release fails, the TTL will eventually clear the lock.
                logger.warning(
                    "Failed to release dream lock for user %s — TTL %ds will clear",
                    user_id[:12],
                    ttl_seconds,
                    exc_info=True,
                )


async def read_dream_lock_token(scope: MemoryScope) -> str | None:
    """Current holder's ownership token, or None when no lock is held.

    The batch path calls this while still holding the lock (at input-bundle
    persist time) to capture its own token; the batch callback — running
    hours later in a different process — passes it back to
    ``release_dream_lock`` for the compare-and-delete.
    """
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    return _as_text(await redis.get(scope.redis_key("dream_lock")))


async def extend_dream_lock(scope: MemoryScope, token: str, ttl_seconds: int) -> bool:
    """``DreamLockHandle.extend`` for a caller holding only the token (a batch
    callback renewing a disowned lock); the caller bounds it."""
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    key = scope.redis_key("dream_lock")
    return await DreamLockHandle(redis, key, scope.owner_user_id, token).extend(
        ttl_seconds
    )


def _as_text(raw: Any) -> str | None:
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


async def release_dream_lock(scope: MemoryScope, token: str | None) -> bool:
    """Release a disowned dream lock (batch path) once the pass terminates,
    and say whether the lock is no longer held under *token*: released now,
    or already another's or nobody's.

    Compare-and-delete on ``token``: a blind delete is NOT safe here — the
    callback can land close to (or after) the lock's TTL, by which point the
    key may already belong to a newer pass, and deleting it would let a
    third concurrent pass start. When the token is unknown (the input bundle
    that carries it expired, or is corrupted) the key is left alone: a
    lockout until its TTL, or until the reaper releases it with the token
    the pass's row keeps, beats releasing someone else's lock. ``False``
    then, and when the delete fails; never raises.
    """
    user_id = scope.owner_user_id
    if token is None:
        logger.warning(
            "No ownership token for disowned dream lock of user %s — "
            "leaving it for the TTL to clear",
            user_id[:12],
        )
        return False
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        deleted = await cast(
            "Any",
            redis.eval(
                _UNLOCK_SCRIPT,
                1,
                scope.redis_key("dream_lock"),
                token,
            ),
        )
        if deleted:
            logger.debug("Released disowned dream lock for user %s", user_id[:12])
        else:
            logger.warning(
                "Disowned dream lock for user %s no longer held our token — "
                "expired and re-acquired; leaving the new holder's lock alone",
                user_id[:12],
            )
    except Exception:
        logger.warning(
            "Failed to release disowned dream lock for user %s — TTL will clear",
            user_id[:12],
            exc_info=True,
        )
        return False
    return True
