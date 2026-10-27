"""V2 External API - Per-Endpoint Rate Limiters"""

from fastapi import Response

from backend.api.utils.rate_limit import RateLimiter

media_upload_limiter = RateLimiter(
    "v2:media_upload", max_requests=10, window_seconds=300
)
search_limiter = RateLimiter("v2:search", max_requests=30, window_seconds=60)
graph_exec_limiter = RateLimiter("v2:graph_exec", max_requests=60, window_seconds=60)
file_upload_limiter = RateLimiter("v2:file_upload", max_requests=20, window_seconds=300)
# Every call fans out to uncached Stripe reads (proration, period end, pending
# change), so the cap is the internal endpoint's: 60/min/user.
subscription_limiter = RateLimiter(
    "v2:subscription", max_requests=60, window_seconds=60
)


async def enforce(limiter: RateLimiter, user_id: str, response: Response) -> None:
    """Apply `limiter` and report its window instead of the global one: it is the
    narrower cap, so it is the one the caller has to back off on."""
    if state := await limiter.check(user_id):
        response.headers.update(state.headers())
