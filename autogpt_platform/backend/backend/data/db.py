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

# Ensure search_path includes common extension schemas for pgvector resolution.
# This fixes random "type 'vector' does not exist" errors when connection pool
# returns connections that don't have pgvector's schema in their search_path.
# We extract the app schema and prepend it, then add 'extensions' and 'public'
# where pgvector is commonly installed (Supabase uses 'extensions', local uses 'public').
_parsed = urlparse(DATABASE_URL)
_query_params = dict(parse_qsl(_parsed.query))
_app_schema = _query_params.get("schema", "public")
# Build search_path: app_schema first (for tables), then extension schemas
_search_path = f"{_app_schema},extensions,public" if _app_schema != "public" else "public,extensions"
# Add search_path via PostgreSQL options parameter
if "options" in _query_params:
    # Append to existing options
    _query_params["options"] = _query_params["options"] + f" -c search_path={_search_path}"
else:
    _query_params["options"] = f"-c search_path={_search_path}"
DATABASE_URL = urlunparse(_parsed._replace(query=urlencode(_query_params)))

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


async def get_connection_debug_info() -> dict:
    """Get diagnostic info about the current database connection.

    Useful for debugging "table does not exist" or "type does not exist" errors
    that may indicate connections going to different database instances.

    Returns dict with: search_path, current_schema, server_version, pg_backend_pid
    """
    import prisma as prisma_module

    try:
        result = await prisma_module.get_client().query_raw(
            """
            SELECT
                current_setting('search_path') as search_path,
                current_schema() as current_schema,
                current_database() as current_database,
                inet_server_addr() as server_addr,
                inet_server_port() as server_port,
                pg_backend_pid() as backend_pid,
                version() as server_version
            """
        )
        return result[0] if result else {}
    except Exception as e:
        return {"error": str(e)}


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
        these via search_path. The connection's search_path is configured at module
        load to include common extension schemas (public, extensions) where pgvector
        may be installed across different environments (local, CI, Supabase).

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

    try:
        if execute:
            result = await db_client.execute_raw(formatted_query, *args)  # type: ignore
        else:
            result = await db_client.query_raw(formatted_query, *args)  # type: ignore
        return result
    except Exception as e:
        error_msg = str(e)
        # Log connection debug info for "does not exist" errors to help diagnose
        # whether connections are going to different database instances
        if "does not exist" in error_msg:
            try:
                debug_info = await get_connection_debug_info()
                logger.error(
                    f"Database object not found. Connection debug info: {debug_info}. "
                    f"Query template: {query_template[:200]}... Error: {error_msg}"
                )
            except Exception:
                logger.error(
                    f"Database object not found (debug info unavailable). "
                    f"Query template: {query_template[:200]}... Error: {error_msg}"
                )
        raise


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
