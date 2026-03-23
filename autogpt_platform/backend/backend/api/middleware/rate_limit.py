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

import base64
import json
import logging

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

_settings = Settings()


def get_user_id_from_request(request: Request) -> str:
    """
    Extract the JWT ``sub`` claim from the Bearer token for per-user rate
    limiting.  No cryptographic verification is performed here — the existing
    auth middleware handles that.  Falls back to the remote IP address when
    the header is absent or the token is malformed.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload_b64 = auth[7:].split(".")[1]
            # Pad to a multiple of 4 for urlsafe_b64decode
            payload_b64 += "=" * (-len(payload_b64) % 4)
            sub = json.loads(base64.urlsafe_b64decode(payload_b64)).get("sub")
            if sub:
                return str(sub)
        except Exception:
            pass
    return get_remote_address(request)


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
