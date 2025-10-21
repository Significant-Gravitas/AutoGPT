import logging
import os

from autogpt_libs.utils.cache import cached, thread_cached
from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from backend.util.retry import conn_retry
from backend.util.settings import Settings

load_dotenv()

HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", None)

logger = logging.getLogger(__name__)

# Check if we should use in-memory implementations
_settings = Settings()
USE_IN_MEMORY = _settings.config.standalone_mode or os.getenv("STANDALONE_MODE", "").lower() in ("true", "1", "yes")


@conn_retry("Redis", "Acquiring connection")
def connect() -> Redis:
    if USE_IN_MEMORY:
        from backend.data.inmemory_redis import InMemoryRedis
        logger.info("Using in-memory Redis client (standalone mode)")
        return InMemoryRedis()

    c = Redis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=True,
    )
    c.ping()
    return c


@conn_retry("Redis", "Releasing connection")
def disconnect():
    get_redis().close()


@cached()
def get_redis() -> Redis:
    return connect()


@conn_retry("AsyncRedis", "Acquiring connection")
async def connect_async() -> AsyncRedis:
    if USE_IN_MEMORY:
        from backend.data.inmemory_redis import InMemoryAsyncRedis
        logger.info("Using in-memory async Redis client (standalone mode)")
        return InMemoryAsyncRedis()

    c = AsyncRedis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
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
