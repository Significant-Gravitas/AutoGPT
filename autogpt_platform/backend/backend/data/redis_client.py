import logging
import os

from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from backend.util.cache import cached, thread_cached
from backend.util.retry import conn_retry

load_dotenv()

HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Default socket timeouts so a wedged Redis endpoint can't hang callers
# indefinitely — long-running code paths (cluster_lock refresh in particular)
# rely on these to fail-fast instead of blocking on no-response TCP. Override
# via env if a specific deployment needs a different budget.
#
# 30s matches the convention in ``backend.data.rabbitmq`` and leaves ~6x
# headroom over the largest ``xread(block=5000)`` wait in stream_registry.
# The connect timeout is shorter (5s) because initial connects should be
# fast; a slow connect usually means the endpoint is genuinely unreachable.
SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "30"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
# How often redis-py sends a PING on idle connections to detect half-open
# sockets; cheap and avoids waiting for the OS TCP keepalive (~2h default).
HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

logger = logging.getLogger(__name__)


@conn_retry("Redis", "Acquiring connection")
def connect() -> Redis:
    c = Redis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
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
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
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
