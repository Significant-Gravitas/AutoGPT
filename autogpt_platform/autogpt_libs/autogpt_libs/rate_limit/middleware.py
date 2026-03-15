from fastapi import HTTPException, Request
from starlette.middleware.base import RequestResponseEndpoint

from .limiter import RateLimiter


async def rate_limit_middleware(request: Request, call_next: RequestResponseEndpoint):
    """FastAPI middleware for rate limiting API requests."""
    limiter = RateLimiter()

    if not request.url.path.startswith("/api"):
        return await call_next(request)

    api_key = request.headers.get("Authorization")
    if not api_key:
        return await call_next(request)

    api_key = api_key.replace("Bearer ", "")

    is_allowed, remaining, reset_time = await limiter.check_rate_limit(api_key)

    if not is_allowed:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limiter.max_requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(reset_time)

    return response
