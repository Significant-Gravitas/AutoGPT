"""Per-IP rate limiting for the public LLM catalog endpoint.

Fixed one-minute window on a single Redis key per IP (cluster-safe — one key
hashes to one slot, so the INCR+EXPIRE pipeline needs no cross-slot MULTI).

Deliberately FAIL-OPEN, unlike copilot's spend limiter: this protects read
capacity for public facts, not money. A Redis brown-out must not take down
catalog distribution — the CDN cache in front is the real shield; this limit
is the backstop against cache-busting abuse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import fastapi
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.util.settings import Config

logger = logging.getLogger(__name__)

config = Config()

_WINDOW_SECONDS = 60
# Key TTL is 2x the window so a clock-edge INCR never resurrects a dead key.
_KEY_TTL_SECONDS = 120


def get_client_ip(request: fastapi.Request) -> str:
    """Best-effort client IP.

    Behind the cloud LB the socket peer is the LB hop, and the true client is
    appended to ``X-Forwarded-For``. ``llm_catalog_client_ip_xff_depth`` picks
    which entry from the END of the list to trust (1 = last entry, the one the
    LB itself appended; anything earlier is client-forgeable). Self-hosted
    installs without a proxy have no XFF header and use the socket peer.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        depth = max(1, config.llm_catalog_client_ip_xff_depth)
        if parts:
            return parts[-depth] if depth <= len(parts) else parts[0]
    return request.client.host if request.client else "unknown"


async def check_catalog_rate_limit(ip: str) -> bool:
    """Return True if the request is allowed, False if over the limit."""
    minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    key = f"llm_catalog:rl:{ip}:{minute}"
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incrby(key, 1)
            pipe.expire(key, _KEY_TTL_SECONDS)
            count, _ = await pipe.execute()
        return int(count) <= config.llm_catalog_rate_limit_per_minute
    except (RedisError, RedisClusterException, RedisConnectionError, OSError):
        logger.warning(
            "Catalog rate-limit check failed — allowing request (fail-open)",
            exc_info=True,
        )
        return True
