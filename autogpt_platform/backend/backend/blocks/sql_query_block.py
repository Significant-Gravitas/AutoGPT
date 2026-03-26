import asyncio
import re
from decimal import Decimal
from typing import Any, Literal
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
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
# NOTE: This regex matches keywords inside string literals and identifiers too
# (e.g. WHERE action = 'DELETE'). This is intentional — the DB-level readonly
# session is the primary safety net; this is a secondary check that favors
# safety over permissiveness. Ambiguous keywords that are harmless on a
# read-only connection (COMMENT, ANALYZE, LOCK, CLUSTER, REINDEX, VACUUM)
# are intentionally excluded to avoid false positives on column names.
_DISALLOWED_SQL_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|EXECUTE|DO\s+\$|CALL|SET\s+(?!LOCAL\s+statement_timeout)"
    r"|RESET|DISCARD|NOTIFY)\b",
    re.IGNORECASE,
)


def _sanitize_error(error_msg: str, connection_string: str) -> str:
    """Remove connection string and credentials from error messages."""
    sanitized = error_msg.replace(connection_string, "<connection_string>")
    sanitized = re.sub(r"password=[^\s&]+", "password=***", sanitized)
    sanitized = re.sub(r"://[^@]+@", "://***:***@", sanitized)
    return sanitized


def _validate_query_is_read_only(query: str) -> str | None:
    """Validate that a SQL query is read-only (SELECT/WITH only).

    Returns an error message if the query is not read-only, None otherwise.
    """
    # Strip SQL comments (-- and /* */).
    # NOTE: This also strips comment-like patterns inside string literals
    # (e.g. SELECT '--text' becomes SELECT ''). This is intentional —
    # we prefer false positives over allowing bypass via string-embedded comments.
    stripped = re.sub(r"--[^\n]*", "", query)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip().rstrip(";").strip()

    if not stripped:
        return "Query is empty."

    # Check that the query starts with SELECT or WITH (for CTEs)
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return "Only SELECT queries are allowed."

    # Defense-in-depth: check for disallowed keywords
    match = _DISALLOWED_SQL_PATTERNS.search(stripped)
    if match:
        return f"Disallowed SQL keyword: {match.group(0).upper()}"

    return None


def _serialize_value(value: Any) -> Any:
    """Convert PostgreSQL-specific types to JSON-serializable Python types."""
    if isinstance(value, Decimal):
        # Use int if there's no fractional part, else float
        return int(value) if value == value.to_integral_value() else float(value)
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
        """Execute a read-only SQL query and return (rows, columns)."""
        conn = psycopg2.connect(
            connection_string,
            connect_timeout=10,
            options=f"-c statement_timeout={timeout * 1000}",
        )
        try:
            conn.set_session(readonly=True, autocommit=True)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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

        # SSRF protection: validate the database host is not internal
        parsed = urlparse(connection_string)
        if parsed.hostname:
            try:
                await resolve_and_check_blocked(parsed.hostname)
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
