import logging
import os
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


def is_connected():
    return prisma.is_connected()


@conn_retry("Prisma", "Acquiring connection")
async def connect():
    if prisma.is_connected():
        return

    await prisma.connect()

    if not prisma.is_connected():
        raise ConnectionError("Failed to connect to Prisma.")

    # Connection acquired from a pool like Supabase somehow still possibly allows
    # the db client obtains a connection but still reject query connection afterward.
    # try:
    #     await prisma.execute_raw("SELECT 1")
    # except Exception as e:
    #     raise ConnectionError("Failed to connect to Prisma.") from e


@conn_retry("Prisma", "Releasing connection")
async def disconnect():
    if not prisma.is_connected():
        return

    await prisma.disconnect()

    if prisma.is_connected():
        raise ConnectionError("Failed to disconnect from Prisma.")


# Transaction timeout constant (in milliseconds)
TRANSACTION_TIMEOUT = 15000  # 15 seconds - Increased from 5s to prevent timeout errors


@asynccontextmanager
async def transaction(timeout: int = TRANSACTION_TIMEOUT):
    """
    Create a database transaction with optional timeout.

    Args:
        timeout: Transaction timeout in milliseconds. If None, uses TRANSACTION_TIMEOUT (15s).
    """
    async with prisma.tx(timeout=timeout) as tx:
        yield tx


@asynccontextmanager
async def locked_transaction(key: str, timeout: int = TRANSACTION_TIMEOUT):
    """
    Create a transaction and take a per-key advisory *transaction* lock.

    - Uses a 64-bit lock id via hashtextextended(key, 0) to avoid 32-bit collisions.
    - Bound by lock_timeout and statement_timeout so it won't block indefinitely.
    - Lock is held for the duration of the transaction and auto-released on commit/rollback.

    Args:
        key: String lock key (e.g., "usr_trx_<uuid>").
        timeout: Transaction/lock/statement timeout in milliseconds.
    """
    async with transaction(timeout=timeout) as tx:
        # Ensure we don't wait longer than desired
        # Note: SET LOCAL doesn't support parameterized queries, must use string interpolation
        await tx.execute_raw(f"SET LOCAL statement_timeout = '{int(timeout)}ms'")  # type: ignore[arg-type]
        await tx.execute_raw(f"SET LOCAL lock_timeout = '{int(timeout)}ms'")  # type: ignore[arg-type]

        # Block until acquired or lock_timeout hits
        try:
            await tx.execute_raw(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                key,
            )
        except Exception as e:
            # Normalize PG's lock timeout error to TimeoutError for callers
            if "lock timeout" in str(e).lower():
                raise TimeoutError(
                    f"Could not acquire lock for key={key!r} within {timeout}ms"
                ) from e
            raise

        yield tx


def get_database_schema() -> str:
    """Extract database schema from DATABASE_URL."""
    parsed_url = urlparse(DATABASE_URL)
    query_params = dict(parse_qsl(parsed_url.query))
    return query_params.get("schema", "public")


async def query_raw_with_schema(query_template: str, *args) -> list[dict]:
    """Execute raw SQL query with proper schema handling."""
    schema = get_database_schema()
    schema_prefix = f"{schema}." if schema != "public" else ""
    formatted_query = query_template.format(schema_prefix=schema_prefix)

    import prisma as prisma_module

    result = await prisma_module.get_client().query_raw(
        formatted_query, *args  # type: ignore
    )

    return result


class BaseDbModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("id", mode="before")
    def set_model_id(cls, id: str) -> str:
        # In case an empty ID is submitted
        return id or str(uuid4())
