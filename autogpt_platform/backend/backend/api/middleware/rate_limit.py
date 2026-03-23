"""
Rate limiting middleware using slowapi (backed by Redis when available).

Default limits are intentionally conservative for public/unauthenticated
endpoints and more generous for authenticated ones. Adjust per-endpoint
limits in rest_api.py by passing a custom `limit` string to @limiter.limit().

Limit string format examples:
  "60/minute"   – 60 requests per minute
  "5/second"    – 5 requests per second
  "1000/hour"   – 1 000 requests per hour
"""

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_settings = Settings()


def _build_storage_uri() -> str:
    """Return a Redis storage URI when Redis is configured, else in-memory."""
    redis_url = getattr(_settings.config, "redis_url", None)
    if redis_url:
        return str(redis_url)
    # Fall back to in-memory (not suitable for multi-process deployments,
    # but safe for local development and single-worker staging).
    logger.warning(
        "Rate limiter: Redis URL not configured — falling back to in-memory "
        "storage. Limits will NOT be shared across worker processes."
    )
    return "memory://"


# ---------------------------------------------------------------------------
# Limiter singleton — import this in rest_api.py to attach to the app
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_build_storage_uri(),
    # Default limit applied to every decorated endpoint that does not
    # specify its own limit string.
    default_limits=["200/minute"],
)
