import logging
import os
import zlib
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from dotenv import load_dotenv
from prisma import Prisma
from pydantic import BaseModel, Field, field_validator

from backend.util.retry import conn_retry

load_dotenv()

PRISMA_SCHEMA = os.getenv("PRISMA_SCHEMA", "schema.prisma")
os.environ["PRISMA_SCHEMA_PATH"] = PRISMA_SCHEMA


def add_param(url: str, key: str, value: str) -> str:
    p = urlparse(url)
    qs = dict(parse_qsl(p.query))
    qs[key] = value
    return urlunparse(p._replace(query=urlencode(qs)))


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432")

CONN_LIMIT = os.getenv("DB_CONNECTION_LIMIT")
if CONN_LIMIT:
    DATABASE_URL = add_param(DATABASE_URL, "connection_limit", CONN_LIMIT)

CONN_TIMEOUT = os.getenv("DB_CONNECT_TIMEOUT")
if CONN_TIMEOUT:
    DATABASE_URL = add_param(DATABASE_URL, "connect_timeout", CONN_TIMEOUT)

POOL_TIMEOUT = os.getenv("DB_POOL_TIMEOUT")
if POOL_TIMEOUT:
    DATABASE_URL = add_param(DATABASE_URL, "pool_timeout", POOL_TIMEOUT)

HTTP_TIMEOUT = int(POOL_TIMEOUT) if POOL_TIMEOUT else None

prisma = Prisma(
    auto_register=True,
    http={"timeout": HTTP_TIMEOUT},
    datasource={"url": DATABASE_URL},
)


logger = logging.getLogger(__name__)


@conn_retry("Prisma", "Acquiring connection")
async def connect():
    if prisma.is_connected():
        return

    await prisma.connect()

    if not prisma.is_connected():
        raise ConnectionError("Failed to connect to Prisma.")

    # Connection acquired from a pool like Supabase somehow still possibly allows
    # the db client obtains a connection but still reject query connection afterward.
    try:
        await prisma.execute_raw("SELECT 1")
    except Exception as e:
        raise ConnectionError("Failed to connect to Prisma.") from e


@conn_retry("Prisma", "Releasing connection")
async def disconnect():
    if not prisma.is_connected():
        return

    await prisma.disconnect()

    if prisma.is_connected():
        raise ConnectionError("Failed to disconnect from Prisma.")


@asynccontextmanager
async def transaction():
    async with prisma.tx() as tx:
        yield tx


@asynccontextmanager
async def locked_transaction(key: str):
    lock_key = zlib.crc32(key.encode("utf-8"))
    async with transaction() as tx:
        await tx.execute_raw(f"SELECT pg_advisory_xact_lock({lock_key})")
        yield tx


class BaseDbModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("id", mode="before")
    def set_model_id(cls, id: str) -> str:
        # In case an empty ID is submitted
        return id or str(uuid4())
