"""The one way the backend reads Redis pub/sub.

Pub/sub keeps no history: an event published while a subscriber is away is
gone. So a subscription here never hides an outage. It confirms its
SSUBSCRIBE before anyone relies on it, never lets redis-py reconnect and
resubscribe behind its owner's back, and PINGs a quiet connection, so a shard
that stops answering without closing the socket (a partition, a hung node) is
noticed too. Each of those ends the stream with ``EventStreamLostError``; the
owner resubscribes and re-reads whatever state the events describe.

``backend/util/architecture_pubsub_test.py`` fails on pub/sub read anywhere
else.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Self

from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.asyncio.connection import AbstractConnection
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import (
    MovedError,
    RedisClusterException,
    RedisError,
    SlotNotCoveredError,
)

from backend.data import redis_client as redis

logger = logging.getLogger(__name__)

# Opening retries for minutes while Redis is down (``conn_retry``); give up
# sooner so the owner can react.
OPEN_TIMEOUT_S = 15.0
_LIVENESS_PING = "autogpt-pubsub-liveness"
# A connection holding only sharded subscriptions isn't in RESP2 subscribe
# mode, so Redis answers the PING with the bare message, which redis-py's
# get_message() garbles; in subscribe mode the answer is a pair. redis-py's own
# health check matches both forms for the same reason.
_PONG_REPLIES = (_LIVENESS_PING, ["pong", _LIVENESS_PING])
_OPEN_ERRORS = (TimeoutError, OSError, RedisError, RedisClusterException)
# The read after a PING's deadline, before the stream is reported lost: long
# enough for a reply held up by a stalled event loop to be read.
_LAST_LOOK_S = 1.0
# How long a failed open waits on the slot-map update before reporting the
# loss; the owner shouldn't wait on Redis reconnect retries.
_SLOT_MAP_WAIT_S = 5.0


class EventStreamLostError(Exception):
    """A pub/sub stream stopped delivering, or could not be opened. Events
    published meanwhile are gone: resubscribe, then re-read current state."""


class ShardedSubscription:
    """SSUBSCRIBEs over one connection pinned to the first channel's shard.

    Only ``open`` waits for Redis to confirm its channel; a channel added
    later is confirmed through ``messages()``, so state read right after
    ``add`` can still miss an event published meanwhile."""

    def __init__(self, client: AsyncRedis, pubsub: AsyncPubSub, name: str) -> None:
        self._client = client
        self._pubsub = pubsub
        self._name = name
        # A failed SSUBSCRIBE or SUNSUBSCRIBE write drops the connection, and
        # redis-py's next read or command would reconnect and resubscribe
        # without a word. This runs in place of the reconnect's handshake
        # (AUTH, HELLO, CLIENT SETINFO), right after the TCP connect, so the
        # reconnect ends the stream, and a shard that accepts the connection
        # but never answers can't hold it: that handshake has no read timeout.
        if pubsub.connection is not None:
            pubsub.connection.redis_connect_func = self._refuse_reconnect

    async def _refuse_reconnect(self, connection: AbstractConnection) -> None:
        # Left connected, it would read nothing: it isn't subscribed any more.
        await connection.disconnect()
        raise EventStreamLostError(f"{self._name}: the connection dropped")

    @classmethod
    async def open(cls, channel: str) -> Self:
        """Subscribe to ``channel`` on its shard, returning once Redis has
        confirmed it: state read after this can't miss a later event."""
        try:
            async with asyncio.timeout(OPEN_TIMEOUT_S):
                return await cls._open(channel)
        except MovedError as e:
            await _follow_moved_slot(e)
            raise EventStreamLostError(f"{channel} moved: {_describe(e)}") from e
        except _OPEN_ERRORS as e:
            await _refresh_slot_map()
            raise EventStreamLostError(
                f"could not subscribe to {channel}: {_describe(e)}"
            ) from e

    @classmethod
    async def _open(cls, channel: str) -> Self:
        try:
            client = await redis.connect_sharded_pubsub_async(channel)
        except RuntimeError as e:
            # resolve_shard_for_channel: no shard owns the slot (mid-reshard).
            raise SlotNotCoveredError(str(e)) from e
        pubsub = client.pubsub()
        try:
            await pubsub.connect()
            if pubsub.connection is None:
                raise RedisConnectionError(f"no pub/sub connection for {channel}")
            # On a read error redis-py would reconnect and resubscribe, and
            # what was published meanwhile would be lost without a word.
            pubsub.connection.retry = AsyncRetry(NoBackoff(), 0, supported_errors=())
            await pubsub.ssubscribe(channel)
            while True:
                message = await pubsub.get_message(timeout=OPEN_TIMEOUT_S)
                if message and message["type"] == "ssubscribe":
                    return cls(client, pubsub, channel)
        except BaseException:
            await _close_quietly(pubsub, client)
            raise

    async def add(self, channel: str) -> None:
        """SSUBSCRIBE another channel on this shard. Unlike ``open``, this
        doesn't wait for Redis to confirm: the reader of ``messages()`` gets
        the confirmation."""
        try:
            await self._pubsub.ssubscribe(channel)
        except (OSError, RedisError) as e:
            raise EventStreamLostError(
                f"could not subscribe to {channel}: {_describe(e)}"
            ) from e

    async def remove(self, channel: str) -> None:
        """SUNSUBSCRIBE; ``messages()`` yields Redis's confirmation."""
        try:
            await self._pubsub.sunsubscribe(channel)
        except (OSError, RedisError) as e:
            raise EventStreamLostError(f"{self._name}: {_describe(e)}") from e

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        """Yield every message (``smessage``, ``ssubscribe``,
        ``sunsubscribe``) until the stream is lost, then raise
        ``EventStreamLostError``: the connection failed, or after
        ``PUBSUB_PING_AFTER`` seconds without a word a PING got no reply
        within ``PUBSUB_PONG_TIMEOUT``. Never ends on its own."""
        loop = asyncio.get_running_loop()
        live = _Liveness(
            loop.time(), redis.PUBSUB_PING_AFTER, redis.PUBSUB_PONG_TIMEOUT
        )
        try:
            while True:
                if live.ping_due(loop.time()):
                    await self._pubsub.ping(_LIVENESS_PING)
                    live.pinged(loop.time())
                response = await self._pubsub.parse_response(
                    block=False, timeout=live.read_timeout(loop.time())
                )
                if response is None:
                    if live.lost(loop.time()):
                        raise EventStreamLostError(
                            f"{self._name}: no reply to a PING in "
                            f"{live.pong_timeout}s"
                        )
                    continue
                live.heard(loop.time())
                if response in _PONG_REPLIES:
                    continue
                if (message := await self._pubsub.handle_message(response)) is not None:
                    yield message
        except (OSError, RedisError) as e:
            raise EventStreamLostError(f"{self._name}: {_describe(e)}") from e

    async def aclose(self) -> None:
        await _close_quietly(self._pubsub, self._client)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


class _Liveness:
    """When a quiet stream is PINGed, and when an unanswered PING means the
    shard is gone. Times come from the caller's clock."""

    def __init__(self, now: float, ping_after: float, pong_timeout: float) -> None:
        self.ping_after = ping_after
        self.pong_timeout = pong_timeout
        self._heard_at = now
        self._pinged_at: float | None = None
        self._last_look = False

    def heard(self, now: float) -> None:
        self._heard_at, self._pinged_at, self._last_look = now, None, False

    def ping_due(self, now: float) -> bool:
        return self._pinged_at is None and now - self._heard_at >= self.ping_after

    def pinged(self, now: float) -> None:
        self._pinged_at = now

    def read_timeout(self, now: float) -> float:
        if self._last_look:
            return _LAST_LOOK_S
        if self._pinged_at is None:
            due = self._heard_at + self.ping_after
        else:
            due = self._pinged_at + self.pong_timeout
        return max(due - now, 0.01)

    def lost(self, now: float) -> bool:
        """After a read that returned nothing."""
        if self._pinged_at is None or now - self._pinged_at < self.pong_timeout:
            return False
        if not self._last_look:
            # A reply that came in while the event loop was stalled can lose
            # the race against the read's timeout: read once more.
            self._last_look = True
            return False
        return True


async def _follow_moved_slot(error: MovedError) -> None:
    # Off the loop, and bounded: get_redis() reconnects (with retries) if its
    # cache expired. A worker that outlives the bound still updates the cache.
    try:
        async with asyncio.timeout(_SLOT_MAP_WAIT_S):
            await asyncio.to_thread(redis.follow_moved_slot, error)
    except Exception:
        logger.warning("Could not follow a MOVED reply", exc_info=True)


def _describe(error: BaseException) -> str:
    # redis-py's repr drops the message ("network:ConnectionError").
    return f"{type(error).__name__}: {error}"


async def _refresh_slot_map() -> None:
    try:
        async with asyncio.timeout(_SLOT_MAP_WAIT_S):
            await asyncio.to_thread(redis.refresh_slot_map)
    except Exception:
        logger.debug("Could not refresh the Redis slot map", exc_info=True)


async def _close_quietly(pubsub: AsyncPubSub, client: AsyncRedis) -> None:
    for closeable in (pubsub, client):
        try:
            await closeable.aclose()
        except Exception:
            logger.debug("Failed to close a pub/sub connection", exc_info=True)
