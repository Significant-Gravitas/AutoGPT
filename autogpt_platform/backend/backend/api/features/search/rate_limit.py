"""Per-user QPS rate limit for ``/api/search/global``.

Search-as-you-type calls fan out to a paid OpenAI embedding for every
non-empty query (see ``unified_hybrid_search`` → ``embed_query``). The
200 ms frontend debounce keeps normal typing well under any reasonable
cap, so this limiter exists to put a hard ceiling on key-held clients
or scripts that bypass the debounce and would otherwise burn embedding
spend without backpressure.
"""

from __future__ import annotations

from backend.api.utils.rate_limit import RateLimiter

GLOBAL_SEARCH_WINDOW_SECONDS = 60
GLOBAL_SEARCH_MAX_REQUESTS = 120

_limiter = RateLimiter(
    "search:global",
    max_requests=GLOBAL_SEARCH_MAX_REQUESTS,
    window_seconds=GLOBAL_SEARCH_WINDOW_SECONDS,
)


async def enforce_global_search_rate_limit(user_id: str) -> None:
    """Raise HTTP 429 when ``user_id`` exceeds the per-window cap."""
    await _limiter.check(user_id)
