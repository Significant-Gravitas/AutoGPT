"""PostHog-backed feature flag evaluation.

Takes plain data rather than importing ``backend.util.feature_flag``: that
module dispatches to this one, so a back-import would be a cycle.
"""

import asyncio
import functools
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from posthog import Posthog

from backend.util.settings import Settings

from .definition_cache import (
    MemoryFlagDefinitionCache,
    RedisFlagDefinitionCache,
    get_flag_definition_cache,
    refresh_interval_seconds,
)

logger = logging.getLogger(__name__)

settings = Settings()

# Without the secret key every read is a remote /flags call; on the default
# executor a burst of those starves every other `asyncio.to_thread` caller.
FLAG_READ_WORKERS = 8
_read_executor = ThreadPoolExecutor(
    max_workers=FLAG_READ_WORKERS, thread_name_prefix="posthog-flags"
)

_client: Posthog | None = None
# The provider _client loads through; it alone sees which loads succeeded.
_definition_cache: RedisFlagDefinitionCache | MemoryFlagDefinitionCache | None = None
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
    *,
    exposure: bool = False,
) -> tuple[Any, bool]:
    """``(value, evaluated)`` for one raw flag read.

    ``evaluated`` is False whenever *default* is standing in for an answer
    PostHog could not give. Unlike LaunchDarkly this needs no inference: the
    SDK returns no result exactly when the flag was not resolved, and a
    disabled one when it conclusively matched no release condition.

    A flag carrying a payload resolves to that payload, mirroring how the
    JSON-valued LaunchDarkly flags return their variation value. Only an
    *exposure* read sends the billed ``$feature_flag_called`` event.
    """
    client = get_flag_client()
    if client is None:
        logger.debug(f"PostHog not configured, using default={default} for {flag_key}")
        return default, False

    try:
        # The read does network I/O whenever the local-evaluation poller has
        # no definition for the flag, so it cannot run on the event loop.
        result = await asyncio.get_running_loop().run_in_executor(
            _read_executor,
            functools.partial(
                client.get_feature_flag_result,
                flag_key,
                distinct_id,
                person_properties=person_properties or None,
                send_feature_flag_events=exposure,
            ),
        )
    except Exception as e:
        logger.warning(
            f"PostHog flag evaluation failed for {flag_key}: {e}, using default={default}"
        )
        return default, False

    if result is None:
        logger.debug(f"PostHog returned no answer for {flag_key}, using {default}")
        return default, False

    # A conclusive "off" is the answer. Serving a payload here instead would
    # turn it into a non-boolean, which every caller reads as "could not
    # evaluate" — the one distinction this evaluator exists to preserve.
    if not result.enabled:
        return False, True

    return (result.payload if result.payload is not None else result.get_value()), True


def evaluate_flags_in_memory(
    flag_keys: list[str], distinct_id: str, person_properties: dict[str, Any]
) -> dict[str, Any]:
    """The flags the loaded definitions answer for *distinct_id*, on this thread.

    Never fetches definitions, never asks ``/flags`` and fires no
    ``$feature_flag_called``; a flag the definitions cannot decide is left out.
    """
    client = get_flag_client()
    # With no definitions loaded the SDK would fetch them before evaluating.
    if client is None or not client.feature_flags:
        return {}
    try:
        result = client.get_all_flags_and_payloads(
            distinct_id,
            person_properties=person_properties,
            only_evaluate_locally=True,
            flag_keys_to_evaluate=flag_keys,
        )
    except Exception as e:
        logger.warning(f"PostHog in-memory flag evaluation failed: {e}")
        return {}
    payloads = result.get("featureFlagPayloads") or {}
    return {
        key: value if value is False else _decoded(payloads.get(key), value)
        for key, value in (result.get("featureFlags") or {}).items()
    }


def defined_flag_keys() -> frozenset[str] | None:
    """Every flag the loaded definitions hold, or None while that is unknown."""
    client = get_flag_client()
    if client is None or client.feature_flags is None:
        return None
    # The SDK also empties its definitions on a rejected key or a quota limit;
    # only a load that succeeded empty means the project has no flags.
    if not client.feature_flags and not _loaded_empty():
        return None
    return frozenset(flag["key"] for flag in client.feature_flags)


def initialize_posthog_flags() -> None:
    """Build the flag client and load its definitions before any read needs them."""
    global _shut_down
    _shut_down = False
    client = get_flag_client()
    if client is None or not client.enable_local_evaluation:
        return

    # The SDK otherwise makes this same call on the first flag read, which then
    # waits for it; going through it keeps the shared cache deciding who fetches.
    try:
        client.load_feature_flags()
    except Exception as e:
        logger.warning(f"Loading PostHog flag definitions at startup failed: {e}")
    if client.feature_flags is None:
        logger.warning(
            "PostHog flag definitions did not load at startup; the first flag read "
            "will retry"
        )


def shutdown_posthog_flags() -> None:
    global _client, _definition_cache, _init_attempted, _shut_down
    with _client_lock:
        client = _client
        _client = None
        _definition_cache = None
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
    one carries the secret key that enables local evaluation, and
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
    global _client, _definition_cache, _init_attempted
    if not is_configured():
        _init_attempted = True
        logger.warning("PostHog API key not configured; flag reads will not resolve")
        return None

    secret_key = settings.secrets.posthog_secret_key
    # One shared refresher, so the definitions bill stops scaling with replica
    # count. Without a provider the SDK polls PostHog once per process.
    definition_cache = get_flag_definition_cache() if secret_key else None
    _definition_cache = definition_cache
    _client = Posthog(
        settings.secrets.posthog_api_key,
        host=settings.secrets.posthog_host,
        secret_key=secret_key or None,
        enable_local_evaluation=bool(secret_key),
        poll_interval=refresh_interval_seconds(),
        flag_definition_cache_provider=definition_cache,
    )
    _init_attempted = True
    logger.info(
        "PostHog feature flag client initialized "
        f"(local evaluation: {'on' if secret_key else 'off'}, "
        f"definition cache: {type(definition_cache).__name__ if definition_cache else 'off'})"
    )
    return _client


def _decoded(payload: Any, value: Any) -> Any:
    """A matched flag's payload, which definitions hold as JSON text, else *value*."""
    if not payload:
        return value
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return payload


def _loaded_empty() -> bool:
    last = _definition_cache.definitions if _definition_cache else None
    return last is not None and not last.get("flags")
