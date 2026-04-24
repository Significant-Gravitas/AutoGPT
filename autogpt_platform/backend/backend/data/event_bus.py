import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Generic, Optional, TypeVar

from pydantic import BaseModel
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.client import PubSub as AsyncPubSub
from redis.client import PubSub

from backend.data import redis_client as redis
from backend.util import json
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
config = Settings().config


M = TypeVar("M", bound=BaseModel)


# Dual-publish (classic PUBLISH + SPUBLISH) during the rolling window so
# old-image pods reading via PSUBSCRIBE still see every event. Flip to
# false once every pod is on the new image; remove the classic path in
# the cleanup PR.
DUAL_PUBLISH = os.getenv("EVENT_BUS_DUAL_PUBLISH", "true").lower() in (
    "1",
    "true",
    "yes",
)


def _assert_no_wildcard(channel_key: str) -> None:
    """Sharded pub/sub has no pattern-subscribe; fail fast on wildcards."""
    if "*" in channel_key:
        raise ValueError(
            f"channel_key {channel_key!r} contains a wildcard; sharded pub/sub "
            "(SSUBSCRIBE) requires exact channel names."
        )


class BaseRedisEventBus(Generic[M], ABC):
    Model: type[M]

    @property
    @abstractmethod
    def event_bus_name(self) -> str:
        pass

    @property
    def Message(self) -> type["_EventPayloadWrapper[M]"]:
        return _EventPayloadWrapper[self.Model]

    def _serialize_message(self, item: M, channel_key: str) -> tuple[str, str]:
        MAX_MESSAGE_SIZE = config.max_message_size_limit

        try:
            # Use backend.util.json.dumps which handles datetime and other complex types
            message = json.dumps(
                self.Message(payload=item), ensure_ascii=False, separators=(",", ":")
            )
        except UnicodeError:
            # Fallback to ASCII encoding if Unicode causes issues
            message = json.dumps(
                self.Message(payload=item), ensure_ascii=True, separators=(",", ":")
            )
            logger.warning(
                f"Unicode serialization failed, falling back to ASCII for channel {channel_key}"
            )

        # Check message size and truncate if necessary
        message_size = len(message.encode("utf-8"))
        if message_size > MAX_MESSAGE_SIZE:
            logger.warning(
                f"Message size {message_size} bytes exceeds limit {MAX_MESSAGE_SIZE} bytes for channel {channel_key}. "
                "Truncating payload to prevent Redis connection issues."
            )
            error_payload = {
                "payload": {
                    "event_type": "error_comms_update",
                    "error": "Payload too large for Redis transmission",
                    "original_size_bytes": message_size,
                    "max_size_bytes": MAX_MESSAGE_SIZE,
                }
            }
            message = json.dumps(
                error_payload, ensure_ascii=False, separators=(",", ":")
            )

        channel_name = f"{self.event_bus_name}/{channel_key}"
        logger.debug(f"[{channel_name}] Publishing an event to Redis {message}")
        return message, channel_name

    def _deserialize_message(self, msg: Any, channel_key: str) -> M | None:
        # ``smessage`` is the sharded pub/sub delivery type.  Accept both
        # sharded and classic message types so the cleanup PR can drop the
        # dual-publish without touching this code path.
        if msg["type"] not in ("smessage", "message", "pmessage"):
            return None
        try:
            logger.debug(f"[{channel_key}] Consuming an event from Redis {msg['data']}")
            return self.Message.model_validate_json(msg["data"]).payload
        except Exception as e:
            logger.error(f"Failed to parse event result from Redis {msg} {e}")

    def _build_channel_name(self, channel_key: str) -> str:
        return f"{self.event_bus_name}/{channel_key}"


class _EventPayloadWrapper(BaseModel, Generic[M]):
    """
    Wrapper model to allow `RedisEventBus.Model` to be a discriminated union
    of multiple event types.
    """

    payload: M


class RedisEventBus(BaseRedisEventBus[M], ABC):
    @property
    def pubsub_connection(self) -> Redis:
        # Dedicated standalone client: a subscribed connection blocks on
        # ``listen()`` and cannot share a socket with regular commands.
        return redis.get_redis_pubsub()

    def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        _assert_no_wildcard(channel_key)
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            cluster = redis.get_redis()
            # SPUBLISH routes to the single keyslot-owning shard, avoiding
            # the cluster-bus broadcast that classic PUBLISH triggers.
            cluster.execute_command("SPUBLISH", full_channel_name, message)
            if DUAL_PUBLISH:
                # Classic PUBLISH keeps old-image (PSUBSCRIBE) pods alive
                # during the rolling deploy window. Cleanup PR removes this.
                self.pubsub_connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    def listen_events(self, channel_key: str) -> Generator[M, None, None]:
        _assert_no_wildcard(channel_key)
        full_channel_name = self._build_channel_name(channel_key)

        # Sync cluster pubsub natively supports ssubscribe/smessage in
        # redis-py 6.x.
        cluster = redis.get_redis()
        pubsub: PubSub = cluster.pubsub()
        try:
            pubsub.ssubscribe(full_channel_name)
            for message in pubsub.listen():
                if event := self._deserialize_message(message, channel_key):
                    yield event
        finally:
            try:
                pubsub.sunsubscribe(full_channel_name)
            except Exception:
                logger.warning(
                    "Failed to SUNSUBSCRIBE from %s", full_channel_name, exc_info=True
                )
            try:
                pubsub.close()
            except Exception:
                logger.warning(
                    "Failed to close sharded pubsub for %s",
                    full_channel_name,
                    exc_info=True,
                )


class AsyncRedisEventBus(BaseRedisEventBus[M], ABC):
    def __init__(self):
        self._pubsub: AsyncPubSub | None = None
        self._pubsub_client: AsyncRedis | None = None

    async def get_pubsub_connection(self) -> AsyncRedis:
        # Kept for back-compat: the classic async client is still used by
        # the dual-publish classic PUBLISH path during the migration window.
        return await redis.get_redis_pubsub_async()

    async def close(self) -> None:
        """Close the shard-pinned PubSub connection if one was opened."""
        if self._pubsub is not None:
            try:
                await self._pubsub.aclose()
            except Exception:
                logger.warning("Failed to close PubSub connection", exc_info=True)
            finally:
                self._pubsub = None
        if self._pubsub_client is not None:
            try:
                await self._pubsub_client.aclose()
            except Exception:
                logger.warning(
                    "Failed to close shard-pinned Redis connection", exc_info=True
                )
            finally:
                self._pubsub_client = None

    async def publish_event(self, event: M, channel_key: str):
        """
        Publish an event to Redis. Gracefully handles connection failures
        by logging the error instead of raising exceptions.
        """
        _assert_no_wildcard(channel_key)
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            cluster = await redis.get_redis_async()
            # redis-py 6.x async cluster has no spublish() wrapper; route
            # via execute_command so the cluster client still handles MOVED.
            await cluster.execute_command("SPUBLISH", full_channel_name, message)
            if DUAL_PUBLISH:
                # Classic PUBLISH for the rolling-deploy compatibility window.
                connection = await self.get_pubsub_connection()
                await connection.publish(full_channel_name, message)
        except Exception:
            logger.exception(
                f"Failed to publish event to Redis channel {channel_key}. "
                "Event bus operation will continue without Redis connectivity."
            )

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        _assert_no_wildcard(channel_key)
        full_channel_name = self._build_channel_name(channel_key)

        # Pin a plain AsyncRedis to the shard that owns the keyslot; sharded
        # pub/sub is only delivered on that shard. redis-py 6.x async PubSub
        # has no ssubscribe wrapper, so we send the raw command and flip
        # ``channels`` ourselves so ``listen()`` sees a subscribed client.
        client = await redis.connect_sharded_pubsub_async(full_channel_name)
        self._pubsub_client = client
        pubsub = client.pubsub()
        self._pubsub = pubsub
        try:
            await pubsub.execute_command("SSUBSCRIBE", full_channel_name)
            # redis-py 6.x async PubSub.subscribed checks the ``channels``
            # dict; subscribe()/psubscribe() populate it, but our raw
            # SSUBSCRIBE does not. Flip it manually so ``listen()`` treats
            # the client as subscribed and keeps reading.
            pubsub.channels[full_channel_name] = None  # type: ignore[index]
            async for message in pubsub.listen():
                if event := self._deserialize_message(message, channel_key):
                    yield event
        finally:
            await self.close()

    async def wait_for_event(
        self, channel_key: str, timeout: Optional[float] = None
    ) -> M | None:
        try:
            return await asyncio.wait_for(
                anext(aiter(self.listen_events(channel_key))), timeout
            )
        except TimeoutError:
            return None
