import asyncio
from typing import Any, Literal

from pydantic import SecretStr
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.blocks.sql_query_helpers import (
    _DATABASE_TYPE_DEFAULT_PORT,
    _DATABASE_TYPE_TO_DRIVER,
    DatabaseType,
    _execute_query,
    _sanitize_error,
    _validate_query_is_read_only,
    _validate_single_statement,
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


class SQLQueryBlock(Block):
    class Input(BlockSchemaInput):
        database_type: DatabaseType = SchemaField(
            default=DatabaseType.POSTGRES,
            description="Database engine",
            advanced=False,
        )
        host: SecretStr = SchemaField(
            description=(
                "Database hostname or IP address. "
                "Treated as a secret to avoid leaking infrastructure details. "
                "Private/internal IPs are blocked (SSRF protection)."
            ),
            placeholder="db.example.com",
            secret=True,
        )
        port: int | None = SchemaField(
            default=None,
            description=(
                "Database port (leave empty for default: "
                "PostgreSQL: 5432, MySQL: 3306, MSSQL: 1433)"
            ),
            ge=1,
            le=65535,
        )
        database: str = SchemaField(
            description="Name of the database to connect to",
            placeholder="my_database",
        )
        query: str = SchemaField(
            description="SQL query to execute",
            placeholder="SELECT * FROM analytics.daily_active_users LIMIT 10",
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
        truncated: bool = SchemaField(
            description=(
                "True when the result set was capped by max_rows, "
                "indicating additional rows exist in the database"
            )
        )
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
                ("truncated", False),
            ],
            test_mock={
                "execute_query": lambda *_args, **_kwargs: (
                    [{"test_col": 1}],
                    ["test_col"],
                    -1,
                    False,
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
        connection_url: URL | str,
        query: str,
        timeout: int,
        max_rows: int,
        read_only: bool = True,
        database_type: DatabaseType = DatabaseType.POSTGRES,
    ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
        """Execute a SQL query and return (rows, columns, affected_rows, truncated).

        Delegates to ``_execute_query`` in ``sql_query_helpers``.
        Extracted as a method so it can be mocked during block tests.
        """
        return _execute_query(
            connection_url=connection_url,
            query=query,
            timeout=timeout,
            max_rows=max_rows,
            read_only=read_only,
            database_type=database_type,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: DatabaseCredentials,
        **_kwargs: Any,
    ) -> BlockOutput:
        # Validate query structure and read-only constraints.
        error = self._validate_query(input_data)
        if error:
            yield "error", error
            return

        # Validate host and resolve for SSRF protection.
        host, pinned_host, error = await self._resolve_host(input_data)
        if error:
            yield "error", error
            return

        # Build connection URL and execute.
        port = input_data.port or _DATABASE_TYPE_DEFAULT_PORT[input_data.database_type]
        username = credentials.username.get_secret_value()
        connection_url = URL.create(
            drivername=_DATABASE_TYPE_TO_DRIVER[input_data.database_type],
            username=username,
            password=credentials.password.get_secret_value(),
            host=pinned_host,
            port=port,
            database=input_data.database,
        )
        conn_str = connection_url.render_as_string(hide_password=True)
        db_name = input_data.database

        def _sanitize(err: Exception) -> str:
            return _sanitize_error(
                str(err).strip(),
                conn_str,
                host=pinned_host,
                original_host=host,
                username=username,
                port=port,
                database=db_name,
            )

        try:
            results, columns, affected, truncated = await asyncio.to_thread(
                self.execute_query,
                connection_url=connection_url,
                query=input_data.query,
                timeout=input_data.timeout,
                max_rows=input_data.max_rows,
                read_only=input_data.read_only,
                database_type=input_data.database_type,
            )
            yield "results", results
            yield "columns", columns
            yield "row_count", len(results)
            yield "truncated", truncated
            if affected >= 0:
                yield "affected_rows", affected
        except OperationalError as e:
            yield (
                "error",
                self._classify_operational_error(
                    _sanitize(e),
                    input_data.timeout,
                ),
            )
        except ProgrammingError as e:
            yield "error", f"SQL error: {_sanitize(e)}"
        except DBAPIError as e:
            yield "error", f"Database error: {_sanitize(e)}"
        except ModuleNotFoundError:
            yield (
                "error",
                (
                    f"Database driver not available for "
                    f"{input_data.database_type.value}. "
                    f"Please contact the platform administrator."
                ),
            )

    @staticmethod
    def _validate_query(input_data: "SQLQueryBlock.Input") -> str | None:
        """Validate query structure and read-only constraints."""
        stmt_error, parsed_stmt = _validate_single_statement(input_data.query)
        if stmt_error:
            return stmt_error
        assert parsed_stmt is not None
        if input_data.read_only:
            return _validate_query_is_read_only(parsed_stmt)
        return None

    async def _resolve_host(
        self, input_data: "SQLQueryBlock.Input"
    ) -> tuple[str, str, str | None]:
        """Validate and resolve the database host. Returns (host, pinned_ip, error)."""
        host = input_data.host.get_secret_value().strip()
        if not host:
            return "", "", "Database host is required."
        if host.startswith("/"):
            return host, "", "Unix socket connections are not allowed."
        try:
            resolved_ips = await self.check_host_allowed(host)
        except (ValueError, OSError) as e:
            return host, "", f"Blocked host: {str(e).strip()}"
        return host, resolved_ips[0], None

    @staticmethod
    def _classify_operational_error(sanitized_msg: str, timeout: int) -> str:
        """Classify an already-sanitized OperationalError for user display."""
        lower = sanitized_msg.lower()
        if "timeout" in lower or "cancel" in lower:
            return f"Query timed out after {timeout}s."
        if "connect" in lower:
            return f"Failed to connect to database: {sanitized_msg}"
        return f"Database error: {sanitized_msg}"
