"""PostHog-backed feature flag evaluation.

Takes plain data rather than importing ``backend.util.feature_flag``: that
module dispatches to this one, so a back-import would be a cycle.
"""

import asyncio
import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from posthog import Posthog

from backend.util.feature_flag_definition_cache import (
    get_flag_definition_cache,
    refresh_interval_seconds,
)
from backend.util.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Without the personal key every read is a remote /flags call; on the default
# executor a burst of those starves every other `asyncio.to_thread` caller.
FLAG_READ_WORKERS = 8
_read_executor = ThreadPoolExecutor(
    max_workers=FLAG_READ_WORKERS, thread_name_prefix="posthog-flags"
)

_client: Posthog | None = None
# The executor evaluates flags from two event loops on two threads; a second
# client built by a racing first read would leak its poller thread.
_client_lock = threading.Lock()
_init_attempted = False
_shut_down = False


def is_configured() -> bool:
    """Whether PostHog has the project key a flag read needs."""
    return bool(settings.secrets.posthog_api_key)


async def evaluate_flag(
    flag_key: str,
    distinct_id: str,
    person_properties: dict[str, Any] | None = None,
    default: Any = None,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` for one raw flag read.

    ``evaluated`` is False whenever *default* is standing in for an answer
    PostHog could not give. Unlike LaunchDarkly this needs no inference:
    ``get_flag`` returns ``None`` exactly when the flag was not resolved, and
    ``False`` when it conclusively matched no release condition.

    A flag carrying a payload resolves to that payload, mirroring how the
    JSON-valued LaunchDarkly flags return their variation value.
    """
    client = get_flag_client()
    if client is None:
        logger.debug(f"PostHog not configured, using default={default} for {flag_key}")
        return default, False

    try:
        # evaluate_flags does network I/O whenever the local-evaluation poller
        # has no definition for the flag, so it cannot run on the event loop.
        snapshot = await asyncio.get_running_loop().run_in_executor(
            _read_executor,
            functools.partial(
                client.evaluate_flags,
                distinct_id,
                person_properties=person_properties or None,
                flag_keys=[flag_key],
            ),
        )
    except Exception as e:
        logger.warning(
            f"PostHog flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default, False

    value = snapshot.get_flag(flag_key)
    if value is None:
        logger.debug(f"PostHog returned no answer for {flag_key}, using {default}")
        return default, False

    # A conclusive "off" is the answer. Serving a payload here instead would
    # turn it into a non-boolean, which every caller reads as "could not
    # evaluate" — the one distinction this evaluator exists to preserve.
    if value is False:
        return False, True

    payload = snapshot.get_flag_payload(flag_key)
    return (payload if payload is not None else value), True


def initialize_posthog_flags() -> None:
    """Build the flag client eagerly so its definition poller is warm."""
    global _shut_down
    _shut_down = False
    get_flag_client()


def shutdown_posthog_flags() -> None:
    global _client, _init_attempted, _shut_down
    with _client_lock:
        client = _client
        _client = None
        # Clear the "did we try" gate even when no client was built, or an
        # unconfigured first attempt latches it shut for the life of the process.
        _init_attempted = False
        # A shadow read runs in a worker thread that outlives the cancelled
        # task, so refuse to rebuild here: that client's poller would have no closer.
        _shut_down = True
    if client is None:
        return

    client.shutdown()
    logger.info("PostHog feature flag client closed successfully")


def get_flag_client() -> Posthog | None:
    """The flag client singleton, or None when PostHog is unconfigured.

    Separate from ``copilot.tracking``'s analytics client because only this
    one carries the personal API key that enables local evaluation, and
    shutting one down must not silence the other.
    """
    global _client, _init_attempted
    if _client is not None or _init_attempted or _shut_down:
        return _client

    with _client_lock:
        if _client is not None or _init_attempted or _shut_down:
            return _client
        return _build_client()


def _build_client() -> Posthog | None:
    global _client, _init_attempted
    if not is_configured():
        _init_attempted = True
        logger.warning("PostHog API key not configured; flag reads will not resolve")
        return None

    personal_api_key = settings.secrets.posthog_personal_api_key
    # One shared refresher, so the definitions bill stops scaling with replica
    # count. Without a provider the SDK polls PostHog once per process.
    definition_cache = get_flag_definition_cache() if personal_api_key else None
    _client = Posthog(
        settings.secrets.posthog_api_key,
        host=settings.secrets.posthog_host,
        personal_api_key=personal_api_key or None,
        enable_local_evaluation=bool(personal_api_key),
        poll_interval=refresh_interval_seconds(),
        flag_definition_cache_provider=definition_cache,
    )
    _init_attempted = True
    logger.info(
        "PostHog feature flag client initialized "
        f"(local evaluation: {'on' if personal_api_key else 'off'}, "
        f"definition cache: {type(definition_cache).__name__ if definition_cache else 'off'})"
    )
    return _client
