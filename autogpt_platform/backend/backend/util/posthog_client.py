"""Process-wide PostHog client shared by every server-side analytics emitter.

The client batches events on a background thread, so one instance per
process is both cheaper and safer than one per module. Emitters send through
:func:`capture`, which adds the base properties every event carries
(``environment`` and ``source``, see ``docs/engineering/tracking-plan.md``),
treats a missing client as "analytics disabled" and never lets tracking
raise into the request or execution that produced the event. An event that
must not be linked to any person goes through :func:`capture_anonymous`.
"""

import atexit
import logging
from collections.abc import Mapping
from functools import cache
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

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
            settings = Settings()
            if settings.secrets.posthog_api_key:
                _client = Posthog(
                    settings.secrets.posthog_api_key,
                    host=settings.secrets.posthog_host,
                )
            else:
                logger.debug("PostHog API key not configured, analytics disabled")
        except Exception:
            logger.warning("Failed to initialize PostHog analytics", exc_info=True)
            return None
        _initialized = True
        return _client


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
) -> None:
    """Send one event for *distinct_id* with the base properties.

    No user means no event: a synthetic distinct id would create a person
    nobody can merge. The base properties are applied last, so a caller can
    never overwrite ``environment`` or ``source`` by accident.

    ``dedup_key`` is an idempotency key for a redelivery or requeue that can
    send the same event twice; it is carried both as ``$insert_id`` and as a
    deterministic event uuid, because PostHog's ingestion keys on the uuid,
    which the client otherwise randomises per call.
    """
    if not distinct_id:
        return
    event_name = "<unknown>"
    try:
        event_name = event.value
        client = get_posthog_client()
        if client is None:
            return
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
        client.capture(
            distinct_id=distinct_id,
            event=event_name,
            properties=payload,
            uuid=event_uuid,
        )
    except Exception:
        logger.warning(
            "Failed to send PostHog event %s for %s",
            event_name,
            distinct_id,
            exc_info=True,
        )


def capture_anonymous(
    event: PostHogEvent,
    properties: Mapping[str, Any] | None = None,
    *,
    source: str = "platform",
) -> None:
    """Send one event that must not be linked to anyone, with the base properties.

    :func:`capture` refuses a synthetic distinct id because it would create a
    person nobody can merge. This is allowed because no person is created:
    ``$process_person_profile`` is false, so the fresh random id only groups
    this one event, and nothing ties that id to an account. It is for counts
    that must stay anonymous (account deletion feedback), so callers must not
    put a user id, email or name in *properties*.

    ``$process_person_profile`` and the base properties are applied last, so a
    caller cannot turn the person profile back on or overwrite ``environment``
    or ``source``. Failures are logged by event name only, never with the
    properties, which can hold free text the person typed.
    """
    event_name = "<unknown>"
    try:
        event_name = event.value
        client = get_posthog_client()
        if client is None:
            return
        client.capture(
            distinct_id=str(uuid4()),
            event=event_name,
            properties={
                **(properties or {}),
                "$process_person_profile": False,
                "environment": _environment(),
                "source": source,
            },
        )
    except Exception as e:
        logger.warning(
            f"Failed to send anonymous PostHog event {event_name}: {type(e).__name__}"
        )


def _shutdown() -> None:
    if _client is not None:
        _client.flush()
        _client.shutdown()


atexit.register(_shutdown)
