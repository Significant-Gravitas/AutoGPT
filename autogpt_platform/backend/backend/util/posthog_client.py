"""Process-wide PostHog client shared by every server-side analytics emitter.

The client batches events on a background thread, so one instance per
process is both cheaper and safer than one per module. Emitters must treat
``None`` as "analytics disabled" and never let tracking raise into the
request or execution that produced the event.
"""

import atexit
import logging
from threading import Lock

from posthog import Posthog

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


def _shutdown() -> None:
    if _client is not None:
        _client.flush()
        _client.shutdown()


atexit.register(_shutdown)
