import logging
import os

from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from backend.util.retry import conn_retry

load_dotenv()

HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", "password")

logger = logging.getLogger(__name__)
connection: Redis | None = None
connection_async: AsyncRedis | None = None


@conn_retry("Redis", "Acquiring connection")
def connect() -> Redis:
    global connection
    if connection:
        return connection

    c = Redis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=True,
    )
    c.ping()
    connection = c
    return connection


@conn_retry("Redis", "Releasing connection")
def disconnect():
    global connection
    if connection:
        connection.close()
    connection = None


def get_redis(auto_connect: bool = True) -> Redis:
    if connection:
        return connection
    if auto_connect:
        return connect()
    raise RuntimeError("Redis connection is not established")


@conn_retry("AsyncRedis", "Acquiring connection")
async def connect_async() -> AsyncRedis:
    global connection_async
    if connection_async:
        return connection_async

    c = AsyncRedis(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=True,
    )
    await c.ping()
    connection_async = c
    return connection_async


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    global connection_async
    if connection_async:
        await connection_async.close()
    connection_async = None


async def get_redis_async(auto_connect: bool = True) -> AsyncRedis:
    if connection_async:
        return connection_async
    if auto_connect:
        return await connect_async()
    raise RuntimeError("AsyncRedis connection is not established")
