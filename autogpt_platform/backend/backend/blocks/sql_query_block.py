import asyncio
import re
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

import sqlparse
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName
from backend.util.request import resolve_and_check_blocked

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="database",
    api_key=SecretStr("postgresql://test_user:test_pass@localhost:5432/test_db"),
    title="Mock Database credentials",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

DatabaseCredentials = APIKeyCredentials
DatabaseCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DATABASE],
    Literal["api_key"],
]


def DatabaseCredentialsField() -> DatabaseCredentialsInput:
    return CredentialsField(
        description="Database connection URL (e.g., postgresql://user:pass@host:5432/db)",
    )


class DatabaseType(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    MSSQL = "mssql"


# Defense-in-depth: reject queries containing data-modifying keywords.
# These are checked against parsed SQL tokens (not raw text) so column names
# and string literals do not cause false positives.
_DISALLOWED_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COPY",
    "EXECUTE",
    "CALL",
    "SET",
    "RESET",
    "DISCARD",
    "NOTIFY",
    "DO",
    # SELECT ... INTO creates tables (PG/MSSQL) or writes files (MySQL OUTFILE/DUMPFILE)
    "INTO",
    "OUTFILE",
    "DUMPFILE",
}


def _sanitize_error(error_msg: str, connection_string: str) -> str:
    """Remove connection string and credentials from error messages."""
    sanitized = error_msg.replace(connection_string, "<connection_string>")
    sanitized = re.sub(r"password=[^\s&]+", "password=***", sanitized)
    sanitized = re.sub(r"://[^@]+@", "://***:***@", sanitized)
    return sanitized


def _extract_keyword_tokens(parsed: sqlparse.sql.Statement) -> list[str]:
    """Extract top-level keyword tokens from a parsed SQL statement.

    Walks the token tree and collects Keyword and DML tokens, skipping
    tokens that are inside string literals, identifiers, or parenthesized groups.
    """
    keywords: list[str] = []
    for token in parsed.flatten():
        if token.ttype in (
            sqlparse.tokens.Keyword,
            sqlparse.tokens.Keyword.DML,
            sqlparse.tokens.Keyword.DDL,
            sqlparse.tokens.Keyword.DCL,
        ):
            keywords.append(token.normalized.upper())
    return keywords


def _validate_query_is_read_only(query: str) -> str | None:
    """Validate that a SQL query is read-only (SELECT/WITH only).

    Uses sqlparse to properly tokenize the query, distinguishing keywords
    from string literals, comments, and identifiers. This prevents bypass
    via quoted comment injection or multi-statement attacks.

    Returns an error message if the query is not read-only, None otherwise.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        return "Query is empty."

    # Parse the SQL using sqlparse for proper tokenization
    statements = sqlparse.parse(stripped)

    # Filter out empty statements (e.g. from trailing semicolons)
    statements = [s for s in statements if s.tokens and str(s).strip()]

    if not statements:
        return "Query is empty."

    # Reject multiple statements — prevents injection via semicolons
    if len(statements) > 1:
        return "Only single statements are allowed."

    stmt = statements[0]
    stmt_type = stmt.get_type()

    # sqlparse returns 'SELECT' for SELECT and WITH...SELECT queries
    if stmt_type != "SELECT":
        return "Only SELECT queries are allowed."

    # Defense-in-depth: check parsed keyword tokens for disallowed keywords
    keywords = _extract_keyword_tokens(stmt)
    for kw in keywords:
        # Normalize multi-word tokens (e.g. "SET LOCAL" -> "SET")
        base_kw = kw.split()[0] if " " in kw else kw
        if base_kw in _DISALLOWED_KEYWORDS:
            return f"Disallowed SQL keyword: {kw}"

    return None


def _serialize_value(value: Any) -> Any:
    """Convert database-specific types to JSON-serializable Python types."""
    if isinstance(value, Decimal):
        # Use int for whole numbers; use str for fractional to preserve exact
        # precision (float would silently round high-precision analytics values).
        if value == value.to_integral_value():
            return int(value)
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (bytes, memoryview)):
        if isinstance(value, memoryview):
            return bytes(value).hex()
        return value.hex()
    return value


# Map DatabaseType enum values to the expected SQLAlchemy driver prefix.
_DATABASE_TYPE_TO_DRIVER = {
    DatabaseType.POSTGRES: "postgresql",
    DatabaseType.MYSQL: "mysql",
    DatabaseType.SQLITE: "sqlite",
    DatabaseType.MSSQL: "mssql",
}


def _validate_connection_url(
    connection_string: str, database_type: DatabaseType
) -> str | None:
    """Validate that the connection URL matches the selected database type.

    Returns an error message on mismatch, None if valid.
    """
    try:
        url = make_url(connection_string)
    except Exception:
        return "Invalid database connection URL."

    expected_prefix = _DATABASE_TYPE_TO_DRIVER[database_type]
    # SQLAlchemy drivername can be "postgresql+psycopg2", "mysql+pymysql", etc.
    driver_base = url.drivername.split("+")[0]
    if driver_base != expected_prefix:
        return (
            f"Connection URL driver '{url.drivername}' does not match "
            f"selected database type '{database_type.value}'. "
            f"Expected a URL starting with '{expected_prefix}://'."
        )
    return None


class SQLQueryBlock(Block):
    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description="SQL SELECT query to execute",
            placeholder="SELECT * FROM analytics.daily_active_users LIMIT 10",
        )
        database_type: DatabaseType = SchemaField(
            default=DatabaseType.POSTGRES,
            description="Type of database to connect to",
        )
        timeout: int = SchemaField(
            default=30,
            description="Query timeout in seconds (max 120)",
            ge=1,
            le=120,
        )
        max_rows: int = SchemaField(
            default=1000,
            description="Maximum number of rows to return (max 10000)",
            ge=1,
            le=10000,
        )
        credentials: DatabaseCredentialsInput = DatabaseCredentialsField()

    class Output(BlockSchemaOutput):
        results: list[dict[str, Any]] = SchemaField(
            description="Query results as a list of row dictionaries"
        )
        columns: list[str] = SchemaField(
            description="Column names from the query result"
        )
        row_count: int = SchemaField(description="Number of rows returned")
        error: str = SchemaField(description="Error message if the query failed")

    def __init__(self):
        super().__init__(
            id="4dc35c0f-4fd8-465e-9616-5a216f1ba2bc",
            description=(
                "Executes a read-only SQL query against a database "
                "and returns the results. Supports PostgreSQL, MySQL, "
                "SQLite, and MSSQL via SQLAlchemy. "
                "Only SELECT queries are allowed."
            ),
            categories={BlockCategory.DATA},
            input_schema=SQLQueryBlock.Input,
            output_schema=SQLQueryBlock.Output,
            test_input={
                "query": "SELECT 1 AS test_col",
                "database_type": DatabaseType.POSTGRES,
                "timeout": 30,
                "max_rows": 1000,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", [{"test_col": 1}]),
                ("columns", ["test_col"]),
                ("row_count", 1),
            ],
            test_mock={
                "execute_query": lambda *args, **kwargs: (
                    [{"test_col": 1}],
                    ["test_col"],
                ),
                "check_host_allowed": lambda *args, **kwargs: None,
            },
        )

    @staticmethod
    async def check_host_allowed(host: str) -> None:
        """Validate that the given host is not a private/blocked address.

        Raises ValueError or OSError if the host is blocked.
        Extracted as a method so it can be mocked during block tests.
        """
        await resolve_and_check_blocked(host)

    @staticmethod
    def execute_query(
        connection_string: str,
        query: str,
        timeout: int,
        max_rows: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Execute a read-only SQL query and return (rows, columns).

        Uses SQLAlchemy to connect to any supported database. The query
        is limited to `max_rows` results via DBAPI fetchmany.
        """
        engine = create_engine(
            connection_string,
            connect_args=(
                {"connect_timeout": 10} if "sqlite" not in connection_string else {}
            ),
            pool_pre_ping=True,
            pool_recycle=300,
        )
        try:
            with engine.connect() as conn:
                # Use AUTOCOMMIT so SET commands take effect immediately and
                # apply to the explicit transaction we open below.
                conn = conn.execution_options(isolation_level="AUTOCOMMIT")

                # Set session-level read-only and timeout before starting
                # the read-only transaction.
                if engine.dialect.name == "postgresql":
                    conn.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                    conn.execute(text("SET default_transaction_read_only = ON"))
                elif engine.dialect.name == "mysql":
                    conn.execute(
                        text(f"SET SESSION MAX_EXECUTION_TIME = {timeout * 1000}")
                    )
                    conn.execute(text("SET SESSION TRANSACTION READ ONLY"))

                # Execute the user query inside an explicit transaction so
                # the read-only setting applies to it.
                conn.execute(text("BEGIN"))
                try:
                    result = conn.execute(text(query))
                    columns = list(result.keys()) if result.returns_rows else []
                    rows = result.fetchmany(max_rows)
                    results = [
                        {col: _serialize_value(val) for col, val in zip(columns, row)}
                        for row in rows
                    ]
                finally:
                    conn.execute(text("ROLLBACK"))
            return results, columns
        finally:
            engine.dispose()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: DatabaseCredentials,
        **kwargs,
    ) -> BlockOutput:
        # Validate query is read-only
        error = _validate_query_is_read_only(input_data.query)
        if error:
            yield "error", error
            return

        connection_string = credentials.api_key.get_secret_value()

        # Validate connection URL matches the selected database type
        url_error = _validate_connection_url(
            connection_string, input_data.database_type
        )
        if url_error:
            yield "error", url_error
            return

        # SSRF protection: parse the connection URL and validate that the
        # host is not internal.
        if input_data.database_type == DatabaseType.SQLITE:
            # SQLite allows arbitrary filesystem access and lacks transaction-
            # level read-only enforcement. Disable until a sandboxing strategy
            # (e.g. allowlisted paths, read-only URI mode) is implemented.
            yield "error", "SQLite is not supported for remote execution."
            return
        else:
            # Network databases: extract host from SQLAlchemy URL and
            # verify it is not a private/blocked address.
            try:
                sa_url = make_url(connection_string)
            except Exception:
                yield "error", "Invalid connection string format."
                return

            host = sa_url.host or ""

            if not host:
                yield "error", "Connection string must specify a database host."
                return

            # Reject Unix socket paths (host starting with '/')
            if host.startswith("/"):
                yield "error", "Unix socket connections are not allowed."
                return

            try:
                await self.check_host_allowed(host)
            except (ValueError, OSError) as e:
                yield "error", f"Blocked host: {str(e).strip()}"
                return

        try:
            results, columns = await asyncio.to_thread(
                self.execute_query,
                connection_string=connection_string,
                query=input_data.query,
                timeout=input_data.timeout,
                max_rows=input_data.max_rows,
            )
            yield "results", results
            yield "columns", columns
            yield "row_count", len(results)
        except OperationalError as e:
            error_msg = _sanitize_error(str(e).strip(), connection_string)
            if "timeout" in error_msg.lower() or "cancel" in error_msg.lower():
                yield "error", f"Query timed out after {input_data.timeout}s."
            elif "connect" in error_msg.lower():
                yield "error", f"Failed to connect to database: {error_msg}"
            else:
                yield "error", f"Database error: {error_msg}"
        except ProgrammingError as e:
            msg = _sanitize_error(str(e).strip(), connection_string)
            yield "error", f"SQL error: {msg}"
        except DBAPIError as e:
            msg = _sanitize_error(str(e).strip(), connection_string)
            yield "error", f"Database error: {msg}"
