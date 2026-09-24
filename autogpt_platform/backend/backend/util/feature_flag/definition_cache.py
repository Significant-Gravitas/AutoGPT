"""Shared cache for PostHog's flag definitions.

Local evaluation makes every process poll PostHog for the flag definitions, and
PostHog bills one definitions fetch as ten flag requests, so the bill scales
with replica count. Here one elected process refreshes the definitions and
writes them to Redis; every other process reads that copy.

Implements PostHog's own ``FlagDefinitionCacheProvider`` protocol, which the
SDK calls only from its poller thread — never from a flag read, so no
evaluation can block on Redis.
"""

import json
import logging
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable

from posthog.flag_definition_cache import (
    FlagDefinitionCacheData,
    FlagDefinitionCacheProvider,
)
from prometheus_client import Counter

from backend.data.redis_client import connect_once
from backend.util.settings import FlagDefinitionCacheBackend, Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Not in backend.monitoring.instrumentation with its siblings: that package
# imports backend.util.metrics, which imports the flag module this one serves.
# The default registry is what /metrics renders, wherever a counter is declared.
CACHE_EVENTS = Counter(
    "autogpt_posthog_flag_definition_cache_events_total",
    "PostHog flag-definition cache events, by outcome",
    # refresher / follower / cached / stale / empty / stored / error — bounded
    labelnames=["outcome"],
)

_DATA_KEY = "posthog:flag_definitions"
_LOCK_KEY = "posthog:flag_definitions:refresher"

# The refresher renews its lock every poll, so a lock outliving two polls hands
# the job to another process within ~2 intervals of the refresher dying.
_LOCK_TTL_POLLS = 2
# Past this the refresher is late or gone. Stale definitions still beat none.
_STALE_AFTER_POLLS = 5
# One line per poll per process would drown the logs for as long as Redis is out.
_DEGRADED_LOG_INTERVAL = 300.0
# The SDK's poller thread waits on these; past them it falls back to fetching.
_REDIS_TIMEOUT_SECONDS = 2.0

# Renew and release only our own lock: a bare PEXPIRE or DEL would extend or
# free whichever process actually holds it.
_RENEW_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_LOCK = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


def get_flag_definition_cache() -> FlagDefinitionCacheProvider | None:
    """The provider the PostHog client shares its flag definitions through.

    None leaves the SDK polling PostHog once per process, which is what it does
    without a provider.
    """
    backend = settings.config.posthog_flag_definition_cache
    if backend is FlagDefinitionCacheBackend.NONE:
        return None
    if backend is FlagDefinitionCacheBackend.MEMORY:
        return MemoryFlagDefinitionCache(refresh_interval=refresh_interval_seconds())
    return RedisFlagDefinitionCache(
        refresh_interval=refresh_interval_seconds(),
        ttl=settings.config.posthog_flag_definition_cache_ttl_seconds,
    )


def refresh_interval_seconds() -> int:
    """How often the SDK polls: the refresher fetches, the rest re-read."""
    return settings.config.posthog_flag_definition_refresh_seconds


class RedisFlagDefinitionCache:
    """Elects one refresher through a Redis lock; the rest read its copy.

    Every failure here degrades to the behaviour we had without it — a direct
    fetch, or the definitions already in memory — and never raises at a caller.
    """

    def __init__(
        self,
        *,
        refresh_interval: int,
        ttl: int,
        redis_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._ttl = ttl
        self._lock_ttl_ms = refresh_interval * _LOCK_TTL_POLLS * 1000
        self._stale_after = refresh_interval * _STALE_AFTER_POLLS
        self._redis_factory = redis_factory or _connect
        self._client: Any = None
        self._instance = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._is_refresher = False
        self._degraded_at = 0.0
        # What this process's SDK last installed, fetched or read.
        self._definitions: FlagDefinitionCacheData | None = None

    def should_fetch_flag_definitions(self) -> bool:
        try:
            redis = self._redis()
            acquired = bool(
                redis.set(_LOCK_KEY, self._instance, nx=True, px=self._lock_ttl_ms)
            ) or bool(
                redis.eval(_RENEW_LOCK, 1, _LOCK_KEY, self._instance, self._lock_ttl_ms)
            )
            # The SDK stores only on a 200; a 304 poll would otherwise let the
            # shared copy lapse while the definitions are unchanged.
            if acquired and self._definitions is not None:
                self._store(redis, self._definitions)
        except Exception as e:
            self._drop_client()
            self._log_degraded("electing a flag-definition refresher", e)
            _record("error")
            # What the SDK falls back to on a provider error anyway: definitions
            # fetched per process beat definitions nobody refreshes.
            return True

        self._note_role(acquired)
        _record("refresher" if acquired else "follower")
        return acquired

    def get_flag_definitions(self) -> FlagDefinitionCacheData | None:
        try:
            raw = self._redis().get(_DATA_KEY)
        except Exception as e:
            self._drop_client()
            self._log_degraded("reading the shared flag definitions", e)
            _record("error")
            # None leaves the SDK on the definitions it already holds, and makes
            # it fetch directly only when it has none at all.
            return None

        if raw is None:
            _record("empty")
            return None

        try:
            cached = json.loads(raw)
            data: FlagDefinitionCacheData = cached["definitions"]
            age = time.time() - float(cached["fetched_at"])
        except (TypeError, ValueError, KeyError) as e:
            logger.warning(f"Unreadable shared PostHog flag definitions: {e}")
            _record("error")
            return None

        if age > self._stale_after:
            logger.warning(
                f"Shared PostHog flag definitions are {age:.0f}s old; serving them "
                "while a refresher catches up"
            )
            _record("stale")
        else:
            _record("cached")
        self._definitions = data
        return data

    def on_flag_definitions_received(self, data: FlagDefinitionCacheData) -> None:
        self._definitions = data
        try:
            self._store(self._redis(), data)
        except Exception as e:
            self._drop_client()
            self._log_degraded("sharing the refreshed flag definitions", e)
            _record("error")
            return

        logger.debug(f"Shared {len(data.get('flags') or [])} PostHog flag definitions")

    def shutdown(self) -> None:
        # Connecting here would hold shutdown for a Redis that is down anyway;
        # the lock then lapses on its own within two polls.
        if not self._is_refresher or self._client is None:
            return
        try:
            self._client.eval(_RELEASE_LOCK, 1, _LOCK_KEY, self._instance)
        except Exception as e:
            logger.warning(f"Could not release the flag-refresher lock: {e}")
        finally:
            self._is_refresher = False
            self._drop_client()

    def _redis(self) -> Any:
        if self._client is None:
            self._client = self._redis_factory()
        return self._client

    def _drop_client(self) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def _store(self, redis: Any, data: FlagDefinitionCacheData) -> None:
        payload = json.dumps({"fetched_at": time.time(), "definitions": data})
        redis.set(_DATA_KEY, payload, ex=self._ttl)
        _record("stored")

    def _note_role(self, is_refresher: bool) -> None:
        if is_refresher == self._is_refresher:
            return
        self._is_refresher = is_refresher
        logger.info(
            f"This process now refreshes the shared PostHog flag definitions "
            f"({self._instance})"
            if is_refresher
            else "Another process now refreshes the shared PostHog flag definitions"
        )

    def _log_degraded(self, what: str, error: Exception) -> None:
        now = time.monotonic()
        if self._degraded_at and now - self._degraded_at < _DEGRADED_LOG_INTERVAL:
            return
        self._degraded_at = now
        logger.warning(
            f"Redis unavailable while {what}: {error}. Each process polls PostHog "
            "for itself until it is back."
        )


class MemoryFlagDefinitionCache:
    """Process-local stand-in, for local dev and tests.

    One process is trivially its own refresher; this exists to exercise the
    provider path without Redis.
    """

    def __init__(self, *, refresh_interval: int) -> None:
        self._refresh_interval = refresh_interval
        self._data: FlagDefinitionCacheData | None = None
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def should_fetch_flag_definitions(self) -> bool:
        with self._lock:
            fresh = (
                self._data is not None
                and time.monotonic() - self._fetched_at < self._refresh_interval
            )
        _record("follower" if fresh else "refresher")
        return not fresh

    def get_flag_definitions(self) -> FlagDefinitionCacheData | None:
        with self._lock:
            data = self._data
        _record("cached" if data else "empty")
        return data

    def on_flag_definitions_received(self, data: FlagDefinitionCacheData) -> None:
        with self._lock:
            self._data = data
            self._fetched_at = time.monotonic()
        _record("stored")

    def shutdown(self) -> None:
        with self._lock:
            self._data = None


def _record(outcome: str) -> None:
    """In a healthy fleet one process records ``stored`` and the rest ``cached``."""
    CACHE_EVENTS.labels(outcome=outcome).inc()


def _connect() -> Any:
    # Not the shared get_redis(): its connect retries for minutes, and this
    # cache has a fallback of its own.
    return connect_once(timeout=_REDIS_TIMEOUT_SECONDS)
