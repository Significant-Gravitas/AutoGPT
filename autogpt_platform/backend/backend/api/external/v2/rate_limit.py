"""
V2 External API - Rate Limiting

Simple in-memory sliding window rate limiter per user.
"""

import time
from collections import defaultdict

from fastapi import HTTPException


class RateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        """Check if the request is within rate limits. Raises 429 if exceeded."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Remove expired timestamps
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > cutoff]

        if len(self._requests[key]) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
            )

        self._requests[key].append(now)


# Pre-configured rate limiters for specific endpoints
media_upload_limiter = RateLimiter(max_requests=10, window_seconds=300)  # 10 / 5min
search_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 / min
execute_limiter = RateLimiter(max_requests=60, window_seconds=60)  # 60 / min
file_upload_limiter = RateLimiter(max_requests=20, window_seconds=300)  # 20 / 5min
