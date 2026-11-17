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
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import MovedError, RedisClusterException, RedisError

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


class EventStreamLostError(Exception):
    """A pub/sub stream stopped delivering, or could not be opened. Events
    published meanwhile are gone: resubscribe, then re-read current state."""


class ShardedSubscription:
    """SSUBSCRIBEs over one connection pinned to the first channel's shard."""

    def __init__(self, client: AsyncRedis, pubsub: AsyncPubSub, name: str) -> None:
        self._client = client
        self._pubsub = pubsub
        self._name = name

    @classmethod
    async def open(cls, channel: str) -> Self:
        """Subscribe to ``channel`` on its shard, returning once Redis has
        confirmed it: state read after this can't miss a later event."""
        try:
            async with asyncio.timeout(OPEN_TIMEOUT_S):
                return await cls._open(channel)
        except MovedError as e:
            redis.follow_moved_slot(e)
            raise EventStreamLostError(f"{channel} moved: {_describe(e)}") from e
        except _OPEN_ERRORS as e:
            await _refresh_slot_map()
            raise EventStreamLostError(
                f"could not subscribe to {channel}: {_describe(e)}"
            ) from e

    @classmethod
    async def _open(cls, channel: str) -> Self:
        client = await redis.connect_sharded_pubsub_async(channel)
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
        ping_after, pong_timeout = redis.PUBSUB_PING_AFTER, redis.PUBSUB_PONG_TIMEOUT
        loop = asyncio.get_running_loop()
        heard_at = loop.time()
        pinged_at: float | None = None
        last_look = False
        try:
            while True:
                if pinged_at is None and loop.time() - heard_at >= ping_after:
                    await self._pubsub.ping(_LIVENESS_PING)
                    pinged_at = loop.time()
                due = (
                    heard_at + ping_after
                    if pinged_at is None
                    else pinged_at + pong_timeout
                )
                response = await self._pubsub.parse_response(
                    block=False, timeout=max(due - loop.time(), 0.01)
                )
                if response is not None:
                    heard_at, pinged_at, last_look = loop.time(), None, False
                    if response in _PONG_REPLIES:
                        continue
                    if (
                        message := await self._pubsub.handle_message(response)
                    ) is not None:
                        yield message
                    continue
                if pinged_at is None or loop.time() - pinged_at < pong_timeout:
                    continue
                if not last_look:
                    # A reply that came in while the event loop was stalled can
                    # lose the race against the read's timeout: read once more.
                    last_look = True
                    continue
                raise EventStreamLostError(
                    f"{self._name}: no reply to a PING in {pong_timeout}s"
                )
        except (OSError, RedisError) as e:
            raise EventStreamLostError(f"{self._name}: {_describe(e)}") from e

    async def aclose(self) -> None:
        await _close_quietly(self._pubsub, self._client)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


def _describe(error: BaseException) -> str:
    # redis-py's repr drops the message ("network:ConnectionError").
    return f"{type(error).__name__}: {error}"


async def _refresh_slot_map() -> None:
    try:
        await asyncio.to_thread(redis.refresh_slot_map)
    except Exception:
        logger.debug("Could not refresh the Redis slot map", exc_info=True)


async def _close_quietly(pubsub: AsyncPubSub, client: AsyncRedis) -> None:
    for closeable in (pubsub, client):
        try:
            await closeable.aclose()
        except Exception:
            logger.debug("Failed to close a pub/sub connection", exc_info=True)
