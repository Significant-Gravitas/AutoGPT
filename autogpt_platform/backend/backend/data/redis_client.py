import logging

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from backend.util.cache import cached, thread_cached
from backend.util.retry import conn_retry
from backend.util.settings import Settings

settings = Settings()

logger = logging.getLogger(__name__)


@conn_retry("Redis", "Acquiring connection")
def connect(decode_responses: bool = True) -> Redis:
    c = Redis(
        host=settings.config.redis_host,
        port=settings.config.redis_port,
        password=settings.config.redis_password or None,
        decode_responses=decode_responses,
    )
    c.ping()
    return c


@conn_retry("Redis", "Releasing connection")
def disconnect():
    get_redis().close()


@cached(ttl_seconds=3600)
def get_redis() -> Redis:
    return connect()


@conn_retry("AsyncRedis", "Acquiring connection")
async def connect_async() -> AsyncRedis:
    c = AsyncRedis(
        host=settings.config.redis_host,
        port=settings.config.redis_port,
        password=settings.config.redis_password or None,
        decode_responses=True,
    )
    await c.ping()
    return c


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    c = await get_redis_async()
    await c.close()


@thread_cached
async def get_redis_async() -> AsyncRedis:
    return await connect_async()
