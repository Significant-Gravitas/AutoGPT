import asyncio
import re
from decimal import Decimal
from typing import Any, Literal

import psycopg2
import psycopg2.extensions
import psycopg2.extras
import sqlparse
from pydantic import SecretStr

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
    provider="postgres",
    api_key=SecretStr("postgresql://test_user:test_pass@localhost:5432/test_db"),
    title="Mock Postgres credentials",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

PostgresCredentials = APIKeyCredentials
PostgresCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.POSTGRES],
    Literal["api_key"],
]


def PostgresCredentialsField() -> PostgresCredentialsInput:
    return CredentialsField(
        description=(
            "PostgreSQL connection string in the format: "
            "postgresql://user:password@host:port/dbname"
        ),
    )


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
    """Convert PostgreSQL-specific types to JSON-serializable Python types."""
    if isinstance(value, Decimal):
        # Use int for whole numbers; use str for fractional to preserve exact
        # precision (float would silently round high-precision analytics values).
        if value == value.to_integral_value():
            return int(value)
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, memoryview):
        return bytes(value).hex()
    return value


class SQLQueryBlock(Block):
    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description="SQL SELECT query to execute",
            placeholder="SELECT * FROM analytics.daily_active_users LIMIT 10",
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
        credentials: PostgresCredentialsInput = PostgresCredentialsField()

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
                "Executes a read-only SQL query against a PostgreSQL database "
                "and returns the results. Only SELECT queries are allowed."
            ),
            categories={BlockCategory.DATA},
            input_schema=SQLQueryBlock.Input,
            output_schema=SQLQueryBlock.Output,
            test_input={
                "query": "SELECT 1 AS test_col",
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
                )
            },
        )

    @staticmethod
    def execute_query(
        connection_string: str,
        query: str,
        timeout: int,
        max_rows: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Execute a read-only SQL query and return (rows, columns).

        Uses a server-side named cursor so that only `max_rows` are fetched
        from the database, avoiding client-side memory exhaustion for large
        result sets.
        """
        conn = psycopg2.connect(
            connection_string,
            connect_timeout=10,
            options=f"-c statement_timeout={timeout * 1000}",
        )
        try:
            # Server-side cursors require a transaction (no autocommit).
            conn.set_session(readonly=True, autocommit=False)
            with conn.cursor(
                name="sql_query_cursor",
                cursor_factory=psycopg2.extras.RealDictCursor,
            ) as cur:
                cur.itersize = max_rows
                cur.execute(query)
                columns = (
                    [desc[0] for desc in cur.description] if cur.description else []
                )
                rows = cur.fetchmany(max_rows)
                results = [
                    {k: _serialize_value(v) for k, v in row.items()} for row in rows
                ]
            return results, columns
        finally:
            conn.close()

    async def run(
        self,
        input_data: Input,
        *,
        credentials: PostgresCredentials,
        **kwargs,
    ) -> BlockOutput:
        # Validate query is read-only
        error = _validate_query_is_read_only(input_data.query)
        if error:
            yield "error", error
            return

        connection_string = credentials.api_key.get_secret_value()

        # SSRF protection: parse the connection string using psycopg2's DSN
        # parser (handles both URI and libpq key=value formats) and validate
        # that the host is not internal. Reject Unix socket paths entirely.
        try:
            dsn_params = psycopg2.extensions.parse_dsn(connection_string)
        except psycopg2.ProgrammingError:
            yield "error", "Invalid connection string format."
            return

        host = dsn_params.get("host", "")
        hostaddr = dsn_params.get("hostaddr", "")

        # Reject if no host is specified (would default to Unix socket)
        if not host and not hostaddr:
            yield "error", "Connection string must specify a database host."
            return

        # Reject Unix socket paths (host starting with '/')
        if host.startswith("/"):
            yield "error", "Unix socket connections are not allowed."
            return

        # Validate each specified host/hostaddr against SSRF blocklist
        hosts_to_check = [h for h in [host, hostaddr] if h]
        for h in hosts_to_check:
            try:
                await resolve_and_check_blocked(h)
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
        except psycopg2.OperationalError as e:
            error_msg = _sanitize_error(str(e).strip(), connection_string)
            if "timeout" in error_msg.lower() or "cancel" in error_msg.lower():
                yield "error", f"Query timed out after {input_data.timeout}s."
            elif "connect" in error_msg.lower():
                yield "error", f"Failed to connect to database: {error_msg}"
            else:
                yield "error", f"Database error: {error_msg}"
        except psycopg2.ProgrammingError as e:
            msg = _sanitize_error(str(e).strip(), connection_string)
            yield "error", f"SQL error: {msg}"
        except psycopg2.Error as e:
            msg = _sanitize_error(str(e).strip(), connection_string)
            yield "error", f"Database error: {msg}"
