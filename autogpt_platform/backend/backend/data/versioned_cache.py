"""A short-TTL cache shared by every API worker, safe to invalidate.

For lookups that are read on almost every page load but must show a change
right after it is made (billing state read from Stripe). Each key has a
random version token in Redis; every change replaces it, so the next read on
any worker misses. An entry is written only while the token it was read under
is still current, so a read that was in flight when a change landed can't put
the old answer back. A missing token (expired, or Redis lost its data) is
never treated as a version: the reader establishes a fresh one before it
fetches, and before it picks its in-flight key.

Within a process, concurrent reads of one key and token share one fetch. A
read stores its answer only while it is still the current one for its key, so
an answer read before a change this process made goes to the callers already
waiting for it and is never kept.

Redis is only an optimisation. Every operation has a short hard deadline, and
after a failure this process skips Redis for a few seconds, so an outage or a
hung connection costs a read at most one deadline and never blocks or fails
the change around it. Meanwhile answers are kept in process for a few seconds
to bound the traffic to the source; under churn across more keys than that
memo holds, it degrades to a fetch per read, still shared per key. When an
invalidation can't replace the token, this process reads the key as if Redis
were out until anything written under the old token has expired. A Redis
that isn't a cluster (the platform always runs one) gets the fallback all the
time.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Coroutine, Generic, TypeVar
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter, ValidationError

from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.data.subscription_lookup_unconfirmed import UnconfirmedInvalidations

logger = logging.getLogger(__name__)

T = TypeVar("T")
V = TypeVar("V")
Fetch = Callable[[], Coroutine[Any, Any, V]]

# Bounds connecting as well as the command: the shared client retries a
# failed connection for minutes on its own.
_REDIS_DEADLINE_SECONDS = 0.3
_REDIS_BACKOFF_SECONDS = 5.0
# How long a process reuses an answer while Redis can't be used; also how long
# another worker can show the old state after a change in that time.
_FALLBACK_TTL_SECONDS = 5.0
_FALLBACK_MAXSIZE = 2048
_VERSION_TTL_SECONDS = 24 * 60 * 60
# The in-flight key for reads while Redis can't be used; tokens are hex.
_NO_REDIS = "no-redis"

# KEYS: version, entry. ARGV: version read, entry, TTL.
_WRITE_IF_CURRENT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
end
return false
"""


class _Entry(BaseModel):
    version: str
    value: Any


class _RedisUnavailable(Exception):
    pass


class _Backoff:
    """Process-local, for every cache: skip Redis for a while after it failed."""

    def __init__(self) -> None:
        self.until = 0.0

    def active(self) -> bool:
        return time.monotonic() < self.until

    def trip(self) -> None:
        if not self.active():
            logger.warning(
                "Versioned cache Redis unavailable; reading sources directly"
                f" for {_REDIS_BACKOFF_SECONDS:.0f}s"
            )
        self.until = time.monotonic() + _REDIS_BACKOFF_SECONDS


_backoff = _Backoff()


class VersionedCache(Generic[V]):
    """``prefix`` names the Redis keys; ``value_type`` is what ``fetch``
    returns, stored as JSON."""

    def __init__(self, prefix: str, value_type: Any, *, ttl_seconds: int) -> None:
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._adapter: TypeAdapter[V] = TypeAdapter(value_type)
        # Reads running in this process, by (event loop, key, version).
        self._in_flight: dict[
            tuple[asyncio.AbstractEventLoop, str, str], asyncio.Task[V]
        ] = {}
        # Answers kept while Redis can't be used: key -> (expires at, answer).
        self._fallback: dict[str, tuple[float, V]] = {}
        # A mark outlives any entry written meanwhile: a write in flight
        # finishes within the Redis deadline, and entries live for the TTL.
        self._unconfirmed = UnconfirmedInvalidations(lifetime=ttl_seconds + 1)

    async def read(self, key: str, fetch: Fetch[V]) -> V:
        """The cached value, or ``fetch``'s answer when it isn't cached.
        ``fetch``'s exceptions reach every caller sharing it, and nothing is
        cached for them."""
        if self._is_unconfirmed(key):
            return await self._read_without_redis(key, fetch)
        try:
            version, cached = await self._read_entry(key)
            if version is None:
                # Established before the in-flight key is chosen, so a read
                # after Redis lost the token never joins one started before.
                version = await self._establish_version(key)
        except _RedisUnavailable:
            return await self._read_without_redis(key, fetch)
        if cached is not None and cached[0] == version:
            return cached[1]
        return await self._shared(
            key, version, lambda: self._fetch_and_store(key, version, fetch)
        )

    async def invalidate(self, key: str) -> None:
        """Make every worker fetch ``key`` fresh on its next read.

        Call it after the change has landed at the source, never before: a
        read started earlier could otherwise cache the old state under the
        new token. Never raises and returns within the Redis deadline,
        because the change it follows has already been made.
        """
        self._forget_locally(key)
        try:
            await _call_redis(
                lambda redis: redis.set(
                    self._version_key(key), uuid4().hex, ex=_VERSION_TTL_SECONDS
                ),
                skip_during_backoff=False,
            )
        except _RedisUnavailable:
            # The old token still vouches for entries written under it.
            self._unconfirmed.add(key, time.monotonic())
            logger.warning(
                f"Could not invalidate {self.prefix} for {key}; other workers"
                f" can show it stale for {self.ttl_seconds}s"
            )
        else:
            # From now on the script refuses anything read under the old token.
            self._unconfirmed.clear(key)

    async def _read_entry(self, key: str) -> tuple[str | None, tuple[str, V] | None]:
        keys = (self._version_key(key), self._entry_key(key))
        version, raw = await _call_redis(lambda redis: redis.mget(keys))
        return _text(version), self._parse_entry(_text(raw))

    async def _establish_version(self, key: str) -> str:
        """A fresh token for a key that has none, or the one another reader
        created first."""
        token = uuid4().hex
        version_key = self._version_key(key)
        created = await _call_redis(
            lambda redis: redis.set(
                version_key, token, nx=True, ex=_VERSION_TTL_SECONDS
            )
        )
        if created:
            return token
        current = _text(await _call_redis(lambda redis: redis.get(version_key)))
        if current:
            return current
        # Expired again in between: a token of our own only costs a miss.
        await _call_redis(
            lambda redis: redis.set(version_key, token, ex=_VERSION_TTL_SECONDS)
        )
        return token

    async def _fetch_and_store(self, key: str, version: str, fetch: Fetch[V]) -> V:
        # Another read may have stored it while this one was being started.
        try:
            current, cached = await self._read_entry(key)
        except _RedisUnavailable:
            current, cached = None, None
        if current == version and cached is not None and cached[0] == version:
            return cached[1]
        value = await fetch()
        if not self._still_current(key, version) or self._is_unconfirmed(key):
            return value
        entry = _Entry(
            version=version, value=self._adapter.dump_python(value, mode="json")
        )
        keys_and_args = (
            self._version_key(key),
            self._entry_key(key),
            version,
            entry.model_dump_json(),
            self.ttl_seconds,
        )
        try:
            await _call_redis(
                lambda redis: redis.eval(_WRITE_IF_CURRENT, 2, *keys_and_args)
            )
        except _RedisUnavailable:
            pass
        return value

    async def _read_without_redis(self, key: str, fetch: Fetch[V]) -> V:
        kept = self._fallback.get(key)
        if kept is not None and kept[0] > time.monotonic():
            return kept[1]
        return await self._shared(
            key, _NO_REDIS, lambda: self._fetch_and_keep(key, fetch)
        )

    async def _fetch_and_keep(self, key: str, fetch: Fetch[V]) -> V:
        value = await fetch()
        if not self._still_current(key, _NO_REDIS):
            return value
        self._fallback[key] = (time.monotonic() + _FALLBACK_TTL_SECONDS, value)
        if len(self._fallback) > _FALLBACK_MAXSIZE:
            # Dropping a kept answer only costs another fetch.
            del self._fallback[next(iter(self._fallback))]
        return value

    def _is_unconfirmed(self, key: str) -> bool:
        return self._unconfirmed.covers(key, time.monotonic())

    async def _shared(
        self, key: str, version: str, read: Callable[[], Coroutine[Any, Any, V]]
    ) -> V:
        """Run ``read`` once for every concurrent caller with the same key.
        The version is part of the key, so a read after a change never joins
        one started before it."""
        flight = (asyncio.get_running_loop(), key, version)
        task = self._in_flight.get(flight)
        if task is None:
            task = asyncio.create_task(read())
            self._in_flight[flight] = task
            task.add_done_callback(lambda done: self._drop_in_flight(flight, done))
        # One caller giving up must not cancel the read the others wait on.
        return await asyncio.shield(task)

    def _still_current(self, key: str, version: str) -> bool:
        """Whether the running read still owns its in-flight key: invalidating
        in this process drops it, and then its answer must not be kept."""
        flight = (asyncio.get_running_loop(), key, version)
        return self._in_flight.get(flight) is asyncio.current_task()

    def _drop_in_flight(
        self,
        flight: tuple[asyncio.AbstractEventLoop, str, str],
        task: asyncio.Task[V],
    ) -> None:
        if self._in_flight.get(flight) is task:
            del self._in_flight[flight]

    def _forget_locally(self, key: str) -> None:
        """Stop this process reusing what it read before a change: its
        fallback answer, and reads still running (whose version may not have
        changed if Redis couldn't be told)."""
        self._fallback.pop(key, None)
        for flight in [flight for flight in self._in_flight if flight[1] == key]:
            del self._in_flight[flight]

    def _version_key(self, key: str) -> str:
        # The hash tag keeps both keys on one cluster slot, for MGET and the
        # script.
        return f"{self.prefix}:{{{key}}}:version"

    def _entry_key(self, key: str) -> str:
        return f"{self.prefix}:{{{key}}}:lookup"

    def _parse_entry(self, raw: str | None) -> tuple[str, V] | None:
        if not raw:
            return None
        try:
            entry = _Entry.model_validate_json(raw)
            return entry.version, self._adapter.validate_python(entry.value)
        except ValidationError:
            return None


def _text(value: str | bytes | None) -> str | None:
    # The shared client decodes responses; its type hints don't say so.
    return value.decode() if isinstance(value, bytes) else value


async def _call_redis(
    operation: Callable[[AsyncRedisClient], Awaitable[T]],
    *,
    skip_during_backoff: bool = True,
) -> T:
    if skip_during_backoff and _backoff.active():
        raise _RedisUnavailable()
    try:
        return await asyncio.wait_for(
            _with_client(operation), timeout=_REDIS_DEADLINE_SECONDS
        )
    except Exception as exc:
        _backoff.trip()
        raise _RedisUnavailable() from exc


async def _with_client(operation: Callable[[AsyncRedisClient], Awaitable[T]]) -> T:
    return await operation(await get_redis_async())
