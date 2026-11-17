import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import aclosing
from typing import Any, AsyncGenerator, Generic, Optional, TypeVar

from pydantic import BaseModel

from backend.data import redis_client as redis
from backend.data.sharded_pubsub import EventStreamLostError, ShardedSubscription
from backend.util import json
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
config = Settings().config


M = TypeVar("M", bound=BaseModel)

# Pause between subscription attempts while a stream can't be reopened.
RESUBSCRIBE_DELAY_S = 1.0


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
        # Accept sharded (smessage) and classic (message/pmessage) deliveries.
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
    def publish_event(self, event: M, channel_key: str):
        """Publish via SPUBLISH; swallow failures so Redis blips don't crash callers."""
        _assert_no_wildcard(channel_key)
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            cluster = redis.get_redis()
            cluster.execute_command("SPUBLISH", full_channel_name, message)
        except Exception:
            logger.exception(f"Failed to publish event to Redis channel {channel_key}")


class AsyncRedisEventBus(BaseRedisEventBus[M], ABC):
    async def close(self) -> None:
        """No-op kept for backward compatibility.

        Earlier revisions of this class stored the per-listen pubsub on the
        instance, requiring an external close. ``listen_events`` now owns its
        own client/pubsub locally so concurrent calls on a singleton (e.g.
        ``_webhook_event_bus``) cannot clobber each other's connection.
        """
        return None

    async def publish_event(self, event: M, channel_key: str):
        """Publish via SPUBLISH; swallow failures so Redis blips don't crash callers."""
        _assert_no_wildcard(channel_key)
        try:
            message, full_channel_name = self._serialize_message(event, channel_key)
            cluster = await redis.get_redis_async()
            # redis-py 6.x async cluster has no spublish(); execute_command handles MOVED.
            await cluster.execute_command("SPUBLISH", full_channel_name, message)
        except Exception:
            logger.exception(f"Failed to publish event to Redis channel {channel_key}")

    async def listen_events(self, channel_key: str) -> AsyncGenerator[M, None]:
        """Yield the channel's events from one subscription.

        Raises ``EventStreamLostError`` when the subscription can't be opened
        or is lost (see ``sharded_pubsub``): events published meanwhile are
        gone, so the caller resubscribes and re-reads what they describe, or
        uses ``listen_with_resync``.
        """
        _assert_no_wildcard(channel_key)
        full_channel_name = self._build_channel_name(channel_key)
        # Generator-local: concurrent listens on one instance (e.g. the
        # singleton _webhook_event_bus) must not share a connection.
        async with await ShardedSubscription.open(full_channel_name) as stream:
            async with aclosing(self._events(stream, channel_key)) as events:
                async for event in events:
                    yield event

    async def listen_with_resync(
        self, channel_key: str
    ) -> AsyncGenerator[M | None, None]:
        """Yield the channel's events for as long as the caller iterates,
        resubscribing whenever the stream is lost.

        ``None`` means events may have been missed: re-read the state they
        describe. It comes once each subscription is confirmed (so nothing
        after the re-read can be missed) and after each failed attempt while
        Redis is unreachable (so a caller waiting on a state change still
        notices it).
        """
        _assert_no_wildcard(channel_key)
        full_channel_name = self._build_channel_name(channel_key)
        while True:
            try:
                async with await ShardedSubscription.open(full_channel_name) as stream:
                    yield None
                    async with aclosing(self._events(stream, channel_key)) as events:
                        async for event in events:
                            yield event
            except EventStreamLostError as e:
                logger.warning(f"Event stream lost, resubscribing: {e}")
            yield None
            await asyncio.sleep(RESUBSCRIBE_DELAY_S)

    async def _events(
        self, stream: ShardedSubscription, channel_key: str
    ) -> AsyncGenerator[M, None]:
        async for message in stream.messages():
            # Pushed by the shard when the channel's slot moves elsewhere.
            if message["type"] == "sunsubscribe":
                raise EventStreamLostError(f"{channel_key}: the shard unsubscribed us")
            if event := self._deserialize_message(message, channel_key):
                yield event

    async def wait_for_event(
        self, channel_key: str, timeout: Optional[float] = None
    ) -> M | None:
        """The channel's next event, or ``None`` after ``timeout`` or when the
        stream is lost first."""
        try:
            async with aclosing(self.listen_events(channel_key)) as events:
                return await asyncio.wait_for(anext(events), timeout)
        except (TimeoutError, EventStreamLostError):
            return None
