"""Observability for model-routing fallbacks.

When LaunchDarkly (or an admin routing cell) references a model slug the
registry doesn't know or has disabled, the resolver falls through to the next
layer. That's the safe behavior — but a typo'd slug in LD would otherwise be
invisible until someone wonders why an experiment isn't running. Every such
refusal is recorded here (single Redis hash key — cluster-safe) so the admin
UI can surface "LD referenced `kimi-k3s` 400× in the last day, no such model".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_WARNINGS_KEY = "llm_route:warnings"
_WARNINGS_TTL_SECONDS = 7 * 24 * 3600


class RouteWarning(BaseModel):
    slug: str
    reason: str
    count: int
    last_seen: datetime
    last_layer: str  # "ld" | "db"


async def record_route_warning(slug: str, reason: str, layer: str) -> None:
    """Best-effort; never lets observability break routing."""
    try:
        redis = await get_redis_async()
        raw = await redis.hget(_WARNINGS_KEY, slug)
        count = (json.loads(raw).get("count", 0) if raw else 0) + 1
        entry = {
            "reason": reason,
            "count": count,
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "last_layer": layer,
        }
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(_WARNINGS_KEY, slug, json.dumps(entry))
            pipe.expire(_WARNINGS_KEY, _WARNINGS_TTL_SECONDS)
            await pipe.execute()
    except Exception:
        logger.warning("Failed to record route warning for %s", slug, exc_info=True)


async def get_route_warnings() -> list[RouteWarning]:
    """Recent routing refusals, most-hit first. Empty on any Redis trouble."""
    try:
        redis = await get_redis_async()
        entries = await redis.hgetall(_WARNINGS_KEY)
    except Exception:
        logger.warning("Failed to read route warnings", exc_info=True)
        return []
    warnings: list[RouteWarning] = []
    for slug, raw in entries.items():
        try:
            data = json.loads(raw)
            warnings.append(
                RouteWarning(
                    slug=slug if isinstance(slug, str) else slug.decode(),
                    reason=data["reason"],
                    count=data["count"],
                    last_seen=data["last_seen"],
                    last_layer=data.get("last_layer", "ld"),
                )
            )
        except Exception:
            continue
    return sorted(warnings, key=lambda w: w.count, reverse=True)
