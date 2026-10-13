"""Process-wide PostHog client shared by every server-side analytics emitter.

The client batches events on a background thread, so one instance per
process is both cheaper and safer than one per module. Emitters send through
:func:`capture`, which adds the base properties every event carries
(``environment`` and ``source``, see ``docs/engineering/tracking-plan.md``),
treats a missing client as "analytics disabled" and never lets tracking
raise into the request or execution that produced the event.
"""

import asyncio
import atexit
import logging
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from posthog import Posthog

from backend.util.posthog_events import PostHogEvent
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_client: Posthog | None = None
_initialized = False
_client_lock = Lock()


def get_posthog_client() -> Posthog | None:
    global _client, _initialized
    if _initialized:
        return _client
    with _client_lock:
        if _initialized:
            return _client

        try:
            _client = new_posthog_client()
            if _client is None:
                logger.debug("PostHog API key not configured, analytics disabled")
        except Exception:
            logger.warning("Failed to initialize PostHog analytics", exc_info=True)
            return None
        _initialized = True
        return _client


def new_posthog_client(**options: Any) -> Posthog | None:
    """A client of the caller's own, for a one-off run that needs `options`
    the shared one doesn't have, such as ``on_error`` to learn which uploads
    failed. None when analytics is off. The caller shuts it down."""
    settings = Settings()
    if not settings.secrets.posthog_api_key:
        return None
    return Posthog(
        settings.secrets.posthog_api_key, host=settings.secrets.posthog_host, **options
    )


# The client drops events once its in-memory queue is full.
FLUSH_EVERY = 500

# One event to send: the distinct id it is for, and the call that queues it on
# the client it is given, saying whether the client took it.
Send = tuple[str, Callable[[Posthog], bool]]


async def send_each(sends: Iterable[Send]) -> int:
    """Queue `sends` on a client of the run's own, flushing as it goes and
    shutting it down at the end, for a one-off run such as a backfill. Returns
    how many distinct ids PostHog didn't take: refused when queued, or in a
    batch whose upload still failed after the client's retries.

    Raises RuntimeError when analytics is off."""
    failed: set[str] = set()

    def upload_failed(_error: Exception, batch: list[dict]) -> None:
        failed.update(str(event.get("distinct_id")) for event in batch)

    client = new_posthog_client(on_error=upload_failed)
    if client is None:
        raise RuntimeError("PostHog is not configured")
    try:
        for index, (distinct_id, send) in enumerate(sends, start=1):
            if not send(client):
                failed.add(distinct_id)
            if index % FLUSH_EVERY == 0:
                await asyncio.to_thread(client.flush)
    finally:
        # Flushes what is left; failed uploads reach `upload_failed` first.
        await asyncio.to_thread(client.shutdown)
    return len(failed)


@cache
def _environment() -> str:
    return Settings().config.app_env.value


def capture(
    distinct_id: str | None,
    event: PostHogEvent,
    properties: Mapping[str, Any] | None = None,
    *,
    source: str = "platform",
    dedup_key: str | None = None,
    client: Posthog | None = None,
) -> bool:
    """Send one event for *distinct_id* with the base properties, on
    ``client`` (from `new_posthog_client`) or else the shared client. True
    when the client queued it, which is not yet delivery.

    No user means no event: a synthetic distinct id would create a person
    nobody can merge. The base properties are applied last, so a caller can
    never overwrite ``environment`` or ``source`` by accident.

    ``dedup_key`` is an idempotency key for a redelivery or requeue that can
    send the same event twice; it is carried both as ``$insert_id`` and as a
    deterministic event uuid, because PostHog's ingestion keys on the uuid,
    which the client otherwise randomises per call.
    """
    if not distinct_id:
        return False
    event_name = "<unknown>"
    try:
        event_name = event.value
        if client is None:
            client = get_posthog_client()
        if client is None:
            return False
        payload: dict[str, Any] = {
            **(properties or {}),
            "environment": _environment(),
            "source": source,
        }
        event_uuid = None
        if dedup_key is not None:
            payload["$insert_id"] = dedup_key
            event_uuid = str(
                uuid5(NAMESPACE_URL, f"{distinct_id}:{event_name}:{dedup_key}")
            )
        # The client returns None for an event it dropped, such as on a full
        # queue.
        queued = client.capture(
            distinct_id=distinct_id,
            event=event_name,
            properties=payload,
            uuid=event_uuid,
        )
        return queued is not None
    except Exception:
        logger.warning(
            "Failed to send PostHog event %s for %s",
            event_name,
            distinct_id,
            exc_info=True,
        )
        return False


def _shutdown() -> None:
    if _client is not None:
        _client.flush()
        _client.shutdown()


atexit.register(_shutdown)
