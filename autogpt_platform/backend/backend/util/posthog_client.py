"""Process-wide PostHog client shared by every server-side analytics emitter.

The client batches events on a background thread, so one instance per
process is both cheaper and safer than one per module. Emitters must treat
``None`` as "analytics disabled" and never let tracking raise into the
request or execution that produced the event.
"""

import atexit
import logging

from posthog import Posthog

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_client: Posthog | None = None


def get_posthog_client() -> Posthog | None:
    global _client
    if _client is not None:
        return _client

    settings = Settings()
    if not settings.secrets.posthog_api_key:
        logger.debug("PostHog API key not configured, analytics disabled")
        return None

    _client = Posthog(
        settings.secrets.posthog_api_key,
        host=settings.secrets.posthog_host,
    )
    logger.info(
        "PostHog client initialized with host: %s", settings.secrets.posthog_host
    )
    return _client


def _shutdown() -> None:
    if _client is not None:
        _client.flush()
        _client.shutdown()


atexit.register(_shutdown)
