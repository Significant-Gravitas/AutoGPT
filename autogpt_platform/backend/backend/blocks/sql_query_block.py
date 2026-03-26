import asyncio
import re
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

import sqlparse
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import (
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
    UserPasswordCredentials,
)
from backend.integrations.providers import ProviderName
from backend.util.request import resolve_and_check_blocked

TEST_CREDENTIALS = UserPasswordCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="database",
    username=SecretStr("test_user"),
    password=SecretStr("test_pass"),
    title="Mock Database credentials",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

DatabaseCredentials = UserPasswordCredentials
DatabaseCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.DATABASE],
    Literal["user_password"],
]


def DatabaseCredentialsField() -> DatabaseCredentialsInput:
    return CredentialsField(
        description="Database username and password",
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

    # Reject multiple statements -- prevents injection via semicolons
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


def _validate_single_statement(
    query: str,
) -> tuple[str | None, sqlparse.sql.Statement | None]:
    """Validate that the query contains exactly one non-empty SQL statement.

    Returns (error_message, parsed_statement). If error_message is not None,
    the query is invalid and parsed_statement will be None.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        return "Query is empty.", None

    # Parse the SQL using sqlparse for proper tokenization
    statements = sqlparse.parse(stripped)

    # Filter out empty statements and comment-only statements
    statements = [
        s
        for s in statements
        if s.tokens
        and str(s).strip()
        and not all(
            t.is_whitespace or t.ttype in sqlparse.tokens.Comment for t in s.flatten()
        )
    ]

    if not statements:
        return "Query is empty.", None

    # Reject multiple statements -- prevents injection via semicolons
    if len(statements) > 1:
        return "Only single statements are allowed.", None

    return None, statements[0]


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

# Default ports for each database type.
_DATABASE_TYPE_DEFAULT_PORT = {
    DatabaseType.POSTGRES: 5432,
    DatabaseType.MYSQL: 3306,
    DatabaseType.MSSQL: 1433,
}


class SQLQueryBlock(Block):
    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description="SQL query to execute",
            placeholder="SELECT * FROM analytics.daily_active_users LIMIT 10",
        )
        database_type: DatabaseType = SchemaField(
            default=DatabaseType.POSTGRES,
            description="Type of database to connect to",
        )
        host: str = SchemaField(
            description="Database host",
            placeholder="db.supabase.co",
        )
        port: int = SchemaField(
            default=0,
            description=(
                "Database port. Leave at 0 to use the default for the selected "
                "database type (5432 for PostgreSQL, 3306 for MySQL, 1433 for MSSQL)"
            ),
        )
        database: str = SchemaField(
            description="Database name",
            placeholder="postgres",
        )
        read_only: bool = SchemaField(
            default=True,
            description=(
                "When enabled (default), only SELECT queries are allowed "
                "and the database session is set to read-only mode. "
                "Disable to allow write operations (INSERT, UPDATE, DELETE, etc.)."
            ),
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
        affected_rows: int = SchemaField(
            description="Number of rows affected by a write query (INSERT/UPDATE/DELETE)"
        )
        error: str = SchemaField(description="Error message if the query failed")

    def __init__(self):
        super().__init__(
            id="4dc35c0f-4fd8-465e-9616-5a216f1ba2bc",
            description=(
                "Execute a SQL query. Read-only by default for safety "
                "-- disable to allow write operations. "
                "Supports PostgreSQL, MySQL, and MSSQL via SQLAlchemy."
            ),
            categories={BlockCategory.DATA},
            input_schema=SQLQueryBlock.Input,
            output_schema=SQLQueryBlock.Output,
            test_input={
                "query": "SELECT 1 AS test_col",
                "database_type": DatabaseType.POSTGRES,
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
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
                "execute_query": lambda *_args, **_kwargs: (
                    [{"test_col": 1}],
                    ["test_col"],
                    -1,
                ),
                "check_host_allowed": lambda *_args, **_kwargs: ["127.0.0.1"],
            },
        )

    @staticmethod
    async def check_host_allowed(host: str) -> list[str]:
        """Validate that the given host is not a private/blocked address.

        Returns the list of resolved IP addresses so the caller can pin the
        connection to the validated IP (preventing DNS rebinding / TOCTOU).
        Raises ValueError or OSError if the host is blocked.
        Extracted as a method so it can be mocked during block tests.
        """
        return await resolve_and_check_blocked(host)

    @staticmethod
    def execute_query(
        connection_string: str,
        query: str,
        timeout: int,
        max_rows: int,
        read_only: bool = True,
    ) -> tuple[list[dict[str, Any]], list[str], int]:
        """Execute a SQL query and return (rows, columns, affected_rows).

        Uses SQLAlchemy to connect to any supported database.
        For SELECT queries, rows are limited to ``max_rows`` via DBAPI fetchmany.
        For write queries, affected_rows contains the rowcount from the driver.
        When ``read_only`` is True, the database session is set to read-only
        mode and the transaction is always rolled back.
        """
        # Determine driver-specific connection timeout argument.
        # pyodbc (MSSQL) uses "timeout", while PostgreSQL/MySQL use "connect_timeout".
        if "sqlite" in connection_string:
            connect_args: dict[str, Any] = {}
        elif "mssql" in connection_string:
            connect_args = {"timeout": 10}
        else:
            connect_args = {"connect_timeout": 10}

        engine = create_engine(
            connection_string,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        try:
            with engine.connect() as conn:
                # Use AUTOCOMMIT so SET commands take effect immediately and
                # apply to the explicit transaction we open below.
                conn = conn.execution_options(isolation_level="AUTOCOMMIT")

                # Set session-level timeout (always) and read-only
                # (when read_only=True) before starting the transaction.
                if engine.dialect.name == "postgresql":
                    conn.execute(text(f"SET statement_timeout = {timeout * 1000}"))
                    if read_only:
                        conn.execute(text("SET default_transaction_read_only = ON"))
                elif engine.dialect.name == "mysql":
                    conn.execute(
                        text(f"SET SESSION MAX_EXECUTION_TIME = {timeout * 1000}")
                    )
                    if read_only:
                        conn.execute(text("SET SESSION TRANSACTION READ ONLY"))

                # Execute the user query inside an explicit transaction so
                # the read-only setting (if enabled) applies to it.
                conn.execute(text("BEGIN"))
                try:
                    result = conn.execute(text(query))

                    affected = result.rowcount if not result.returns_rows else -1

                    columns = list(result.keys()) if result.returns_rows else []
                    rows = result.fetchmany(max_rows) if result.returns_rows else []
                    results = [
                        {col: _serialize_value(val) for col, val in zip(columns, row)}
                        for row in rows
                    ]
                finally:
                    if read_only:
                        conn.execute(text("ROLLBACK"))
                    else:
                        conn.execute(text("COMMIT"))
            return results, columns, affected
        finally:
            engine.dispose()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: DatabaseCredentials,
        **_kwargs: Any,
    ) -> BlockOutput:
        # Multi-statement prevention applies in both modes (security against injection)
        stmt_error, _ = _validate_single_statement(input_data.query)
        if stmt_error:
            yield "error", stmt_error
            return

        # When read_only (default), enforce SELECT-only queries
        if input_data.read_only:
            ro_error = _validate_query_is_read_only(input_data.query)
            if ro_error:
                yield "error", ro_error
                return

        # SQLite is not supported for remote execution
        if input_data.database_type == DatabaseType.SQLITE:
            yield "error", "SQLite is not supported for remote execution."
            return

        host = input_data.host.strip()
        if not host:
            yield "error", "Database host is required."
            return

        # Reject Unix socket paths (host starting with '/')
        if host.startswith("/"):
            yield "error", "Unix socket connections are not allowed."
            return

        # SSRF protection: validate that the host is not internal.
        # We use the resolved IP address for the actual connection to prevent
        # DNS rebinding (TOCTOU) attacks where the DNS record changes between
        # the check and the connection.
        try:
            resolved_ips = await self.check_host_allowed(host)
        except (ValueError, OSError) as e:
            yield "error", f"Blocked host: {str(e).strip()}"
            return

        # Pin the connection to the first resolved IP to prevent DNS rebinding.
        pinned_host = resolved_ips[0] if resolved_ips else host

        # Build the SQLAlchemy connection URL from discrete fields.
        # URL.create() accepts the raw password without URL-encoding,
        # so special characters like @, #, ! work correctly.
        drivername = _DATABASE_TYPE_TO_DRIVER[input_data.database_type]
        port = input_data.port or _DATABASE_TYPE_DEFAULT_PORT.get(
            input_data.database_type, 5432
        )
        username = credentials.username.get_secret_value()
        password = credentials.password.get_secret_value()
        connection_url = URL.create(
            drivername=drivername,
            username=username,
            password=password,
            host=pinned_host,
            port=port,
            database=input_data.database,
        )
        connection_string = connection_url.render_as_string(hide_password=False)

        try:
            results, columns, affected = await asyncio.to_thread(
                self.execute_query,
                connection_string=connection_string,
                query=input_data.query,
                timeout=input_data.timeout,
                max_rows=input_data.max_rows,
                read_only=input_data.read_only,
            )
            yield "results", results
            yield "columns", columns
            yield "row_count", len(results)
            if affected >= 0:
                yield "affected_rows", affected
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
