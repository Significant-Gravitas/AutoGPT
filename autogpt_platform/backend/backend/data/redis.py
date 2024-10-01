import logging
import os

from dotenv import load_dotenv
from redis import Redis

from backend.util.retry import conn_retry

load_dotenv()

HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", "password")
QUEUE_NAME = os.getenv("REDIS_QUEUE", "execution_events")

logger = logging.getLogger(__name__)
connection: Redis | None = None


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


def get_redis() -> Redis:
    if not connection:
        raise RuntimeError("Redis connection is not established")
    return connection
