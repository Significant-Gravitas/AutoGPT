"""The executor's liveness lease on a running copilot turn.

A turn's session meta says ``running`` for up to ``stream_ttl`` (an hour),
so on its own it cannot tell a live turn from one whose executor pod died
mid-turn. The lease can: a short-lived Redis key per turn that the executor
writes when it starts the turn and refreshes from its worker thread every
``TURN_LEASE_REFRESH_SECONDS``. Once the executor stops refreshing it the key
lapses within ``TURN_LEASE_TTL_SECONDS``, and a reader that finds the meta of
a claimed turn running without a lease ends that turn as failed (see
``stream_registry``).

Refreshed from the worker thread, not the turn's event loop: a loop busy for
a while is a slow turn, not a dead executor.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, cast

from backend.data import redis_client

logger = logging.getLogger(__name__)

TURN_LEASE_TTL_SECONDS = 30
TURN_LEASE_REFRESH_SECONDS = 10
_TURN_LEASE_PREFIX = "copilot:turn_lease:"

# Delete only while we still own it, so a release never removes a lease that
# a redelivered copy of the turn took over on another pod.
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) "
    "else return 0 end"
)


def turn_lease_key(turn_id: str) -> str:
    return f"{_TURN_LEASE_PREFIX}{turn_id}"


class TurnLease:
    """The lease one executor holds on one turn. Never raises: a Redis blip
    is logged, and the next refresh writes the key again."""

    def __init__(
        self,
        turn_id: str,
        owner_id: str,
        *,
        redis: Any = None,
        ttl_seconds: int = TURN_LEASE_TTL_SECONDS,
        refresh_seconds: float = TURN_LEASE_REFRESH_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.key = turn_lease_key(turn_id)
        # Resolved on first use, inside the error handling: building the
        # cluster client connects, and the lease must never raise.
        self._redis = redis
        self._owner_id = owner_id
        self._ttl_seconds = ttl_seconds
        self._refresh_seconds = refresh_seconds
        self._clock = clock
        self._written_at: float | None = None
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        return self._write()

    def refresh(self) -> bool:
        """Write the lease again once ``refresh_seconds`` have passed; a
        cheap no-op in between, so it can be called on every tick."""
        with self._lock:
            written_at = self._written_at
        if written_at is not None and (
            self._clock() - written_at < self._refresh_seconds
        ):
            return True
        return self._write()

    def release(self) -> None:
        try:
            self._client().eval(_RELEASE_LUA, 1, self.key, self._owner_id)
        except Exception as e:
            logger.warning(f"Failed to release turn lease {self.key}: {e}")

    def _write(self) -> bool:
        # SET rather than EXPIRE: a lease that lapsed during a Redis blip is
        # written again instead of staying gone for the rest of the turn.
        try:
            self._client().set(self.key, self._owner_id, ex=self._ttl_seconds)
        except Exception as e:
            logger.warning(f"Failed to write turn lease {self.key}: {e}")
            with self._lock:
                self._written_at = None
            return False
        with self._lock:
            self._written_at = self._clock()
        return True

    def _client(self) -> Any:
        if self._redis is None:
            self._redis = cast(Any, redis_client.get_redis())
        return self._redis


async def turn_lease_held(redis: Any, turn_id: str) -> bool:
    """Whether an executor currently holds ``turn_id``'s lease."""
    return bool(await redis.exists(turn_lease_key(turn_id)))
