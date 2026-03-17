import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
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


# Transaction timeout constant:
# increased from 15s to prevent timeout errors during graph creation under load.
TRANSACTION_TIMEOUT = timedelta(seconds=30)


@asynccontextmanager
async def transaction(timeout: timedelta = TRANSACTION_TIMEOUT):
    """
    Create a database transaction with optional timeout.

    Args:
        timeout: Transaction timeout as a timedelta.
            Defaults to `TRANSACTION_TIMEOUT` (30s).
    """
    async with prisma.tx(timeout=timeout) as tx:
        yield tx


def get_database_schema() -> str:
    """Extract database schema from DATABASE_URL."""
    parsed_url = urlparse(DATABASE_URL)
    query_params = dict(parse_qsl(parsed_url.query))
    return query_params.get("schema", "public")


async def _raw_with_schema(
    query_template: str,
    *args,
    execute: bool = False,
    client: Prisma | None = None,
) -> list[dict] | int:
    """Internal: Execute raw SQL with proper schema handling.

    Use query_raw_with_schema() or execute_raw_with_schema() instead.

    Supports placeholders:
        - {schema_prefix}: Table/type prefix (e.g., "platform".)
        - {schema}: Raw schema name for application tables (e.g., platform)

    Note on pgvector types:
        Use unqualified ::vector and <=> operator in queries. PostgreSQL resolves
        these via search_path, which includes the schema where pgvector is installed
        on all environments (local, CI, dev).

    Args:
        query_template: SQL query with {schema_prefix} and/or {schema} placeholders
        *args: Query parameters
        execute: If False, executes SELECT query. If True, executes INSERT/UPDATE/DELETE.
        client: Optional Prisma client for transactions (only used when execute=True).

    Returns:
        - list[dict] if execute=False (query results)
        - int if execute=True (number of affected rows)

    Example with vector type:
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"Embedding" (vec) VALUES ($1::vector)',
            embedding_data
        )
    """
    schema = get_database_schema()
    schema_prefix = f'"{schema}".' if schema != "public" else ""

    formatted_query = query_template.format(
        schema_prefix=schema_prefix,
        schema=schema,
    )

    import prisma as prisma_module

    db_client = client if client else prisma_module.get_client()

    if execute:
        result = await db_client.execute_raw(formatted_query, *args)  # type: ignore
    else:
        result = await db_client.query_raw(formatted_query, *args)  # type: ignore

    return result


async def query_raw_with_schema(query_template: str, *args) -> list[dict]:
    """Execute raw SQL SELECT query with proper schema handling.

    Args:
        query_template: SQL query with {schema_prefix} and/or {schema} placeholders
        *args: Query parameters

    Returns:
        List of result rows as dictionaries

    Example:
        results = await query_raw_with_schema(
            'SELECT * FROM {schema_prefix}"User" WHERE id = $1',
            user_id
        )
    """
    return await _raw_with_schema(query_template, *args, execute=False)  # type: ignore


async def execute_raw_with_schema(
    query_template: str,
    *args,
    client: Prisma | None = None,
) -> int:
    """Execute raw SQL command (INSERT/UPDATE/DELETE) with proper schema handling.

    Args:
        query_template: SQL query with {schema_prefix} and/or {schema} placeholders
        *args: Query parameters
        client: Optional Prisma client for transactions

    Returns:
        Number of affected rows

    Example:
        await execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"User" (id, name) VALUES ($1, $2)',
            user_id, name,
            client=tx  # Optional transaction client
        )
    """
    return await _raw_with_schema(query_template, *args, execute=True, client=client)  # type: ignore


class BaseDbModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator("id", mode="before")
    def set_model_id(cls, id: str) -> str:
        # In case an empty ID is submitted
        return id or str(uuid4())
