import time
from typing import Tuple

from redis import Redis

from .config import RATE_LIMIT_SETTINGS


class RateLimiter:
    def __init__(
        self,
        redis_host: str = RATE_LIMIT_SETTINGS.redis_host,
        redis_port: str = RATE_LIMIT_SETTINGS.redis_port,
        redis_password: str = RATE_LIMIT_SETTINGS.redis_password,
        requests_per_minute: int = RATE_LIMIT_SETTINGS.requests_per_minute,
    ):
        self.redis = Redis(
            host=redis_host,
            port=int(redis_port),
            password=redis_password,
            decode_responses=True,
        )
        self.window = 60
        self.max_requests = requests_per_minute

    async def check_rate_limit(self, api_key_id: str) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limits.

        Args:
            api_key_id: The API key identifier to check

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_time)
        """
        now = time.time()
        window_start = now - self.window
        key = f"ratelimit:{api_key_id}:1min"

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcount(key, window_start, now)
        pipe.expire(key, self.window)

        _, _, request_count, _ = pipe.execute()

        remaining = max(0, self.max_requests - request_count)
        reset_time = int(now + self.window)

        return request_count <= self.max_requests, remaining, reset_time
