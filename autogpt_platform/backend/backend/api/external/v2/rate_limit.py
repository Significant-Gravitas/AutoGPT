"""V2 External API - Per-Endpoint Rate Limiters"""

from backend.api.utils.rate_limit import RateLimiter

media_upload_limiter = RateLimiter(
    "v2:media_upload", max_requests=10, window_seconds=300
)
search_limiter = RateLimiter("v2:search", max_requests=30, window_seconds=60)
graph_exec_limiter = RateLimiter("v2:graph_exec", max_requests=60, window_seconds=60)
file_upload_limiter = RateLimiter("v2:file_upload", max_requests=20, window_seconds=300)
