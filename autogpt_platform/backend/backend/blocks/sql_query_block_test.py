"""Tests for SQLQueryBlock: single-statement validation, URL validation,
error sanitization, SSRF protection, and read/write mode behavior."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr
from sqlalchemy.exc import OperationalError

from backend.blocks.sql_query_block import (
    DatabaseType,
    SQLQueryBlock,
    UserPasswordCredentials,
    _sanitize_error,
    _serialize_value,
    _validate_query_is_read_only,
    _validate_single_statement,
)


class TestValidateSingleStatement:
    """Tests for _validate_single_statement multi-statement prevention."""

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 1",
            "SELECT * FROM users",
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x = 1",
            "DELETE FROM t WHERE id = 1",
            "CREATE TABLE t (id INT)",
            "DROP TABLE t",
            "SELECT * FROM users;",
            "  SELECT 1  ",
        ],
    )
    def test_valid_single_statements(self, query: str):
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM users; DELETE FROM users",
            "SELECT * FROM users; DROP TABLE users",
            "SELECT 1; INSERT INTO users VALUES (1)",
            "INSERT INTO t VALUES (1); DROP TABLE t",
            "SELECT '--'; SET LOCAL statement_timeout = 0; SELECT pg_sleep(1000)",
            "SELECT 1; SET search_path TO public",
            "SELECT 1; RESET ALL",
        ],
    )
    def test_multi_statement_rejected(self, query: str):
        error, stmt = _validate_single_statement(query)
        assert error == "Only single statements are allowed."
        assert stmt is None

    def test_empty_query(self):
        error, stmt = _validate_single_statement("")
        assert error == "Query is empty."
        assert stmt is None

    def test_whitespace_only_query(self):
        error, stmt = _validate_single_statement("   ")
        assert error == "Query is empty."
        assert stmt is None

    def test_semicolon_only(self):
        error, stmt = _validate_single_statement(";")
        assert error == "Query is empty."
        assert stmt is None

    def test_comment_only_query(self):
        error, stmt = _validate_single_statement("-- just a comment")
        assert error == "Query is empty."
        assert stmt is None


class TestValidateQueryIsReadOnly:
    """Tests for _validate_query_is_read_only read-only enforcement."""

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 1",
            "SELECT * FROM users",
            "SELECT * FROM users WHERE id = 1",
            "  SELECT 1  ",
        ],
    )
    def test_valid_select_queries(self, query: str):
        assert _validate_query_is_read_only(query) is None

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET x = 1",
            "DELETE FROM t WHERE id = 1",
            "DROP TABLE t",
            "CREATE TABLE t (id INT)",
        ],
    )
    def test_non_select_rejected(self, query: str):
        result = _validate_query_is_read_only(query)
        assert result is not None


class TestSerializeValue:
    """Tests for _serialize_value type conversion."""

    def test_decimal_integer(self):
        assert _serialize_value(Decimal("42")) == 42
        assert isinstance(_serialize_value(Decimal("42")), int)

    def test_decimal_fractional(self):
        # Fractional decimals are serialized as strings to preserve exact precision
        assert _serialize_value(Decimal("3.14")) == "3.14"
        assert isinstance(_serialize_value(Decimal("3.14")), str)

    def test_decimal_high_precision(self):
        # High-precision values must not lose precision via float conversion
        val = Decimal("123456789.123456789012345678")
        result = _serialize_value(val)
        assert result == "123456789.123456789012345678"
        assert isinstance(result, str)

    def test_datetime(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        assert _serialize_value(dt) == "2024-01-01T12:00:00"

    def test_date(self):
        d = date(2024, 6, 15)
        assert _serialize_value(d) == "2024-06-15"

    def test_time(self):
        t = time(14, 30, 45)
        assert _serialize_value(t) == "14:30:45"

    def test_memoryview(self):
        mv = memoryview(b"\xde\xad\xbe\xef")
        assert _serialize_value(mv) == "deadbeef"

    def test_bytes(self):
        assert _serialize_value(b"\xca\xfe") == "cafe"

    def test_passthrough_string(self):
        assert _serialize_value("hello") == "hello"

    def test_passthrough_int(self):
        assert _serialize_value(42) == 42

    def test_passthrough_none(self):
        assert _serialize_value(None) is None

    def test_passthrough_list(self):
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]


class TestSanitizeError:
    """Tests for _sanitize_error credential and infrastructure scrubbing."""

    def test_connection_string_replaced(self):
        conn = "postgresql://user:secret@host:5432/db"
        error = f"could not connect to server: {conn}"
        result = _sanitize_error(error, conn)
        assert "secret" not in result
        assert "<connection_string>" in result

    def test_password_param_scrubbed(self):
        conn = "postgresql://user:secret@host:5432/db"
        error = "FATAL: password=secret123 authentication failed"
        result = _sanitize_error(error, conn)
        assert "secret123" not in result
        assert "password=***" in result

    def test_url_credentials_scrubbed(self):
        conn = "postgresql://user:secret@host:5432/db"
        error = "connection to postgresql://admin:hunter2@db.internal:5432/prod failed"
        result = _sanitize_error(error, conn)
        assert "hunter2" not in result
        assert "***:***@" in result

    def test_clean_error_unchanged(self):
        conn = "postgresql://user:secret@host:5432/db"
        error = "relation 'foo' does not exist"
        result = _sanitize_error(error, conn)
        assert result == error

    def test_mysql_connection_string_replaced(self):
        conn = "mysql://admin:s3cret@db.example.com:3306/mydb"
        error = f"Can't connect to MySQL server: {conn}"
        result = _sanitize_error(error, conn)
        assert "s3cret" not in result
        assert "<connection_string>" in result

    def test_hostname_scrubbed(self):
        """Database hostname must be replaced with <host>."""
        conn = "postgresql://user:pass@52.73.47.72:5432/db"
        error = (
            'connection to server at "db.adfjtextkuilwuhzdjpf.supabase.co" '
            "(52.73.47.72), port 5432 failed: FATAL: "
            'password authentication failed for user "postgres"'
        )
        result = _sanitize_error(
            error, conn, host="52.73.47.72", username="postgres", port=5432
        )
        assert "52.73.47.72" not in result
        assert "postgres" not in result
        assert "5432" not in result
        assert "<host>" in result
        assert "<user>" in result
        assert "password authentication failed" in result

    def test_ip_addresses_scrubbed(self):
        """Any IPv4 address in the error must be replaced with <ip>."""
        conn = "postgresql://user:pass@10.0.0.5:5432/db"
        error = "could not connect to 10.0.0.5 (192.168.1.100): Connection refused"
        result = _sanitize_error(error, conn, host="10.0.0.5")
        assert "10.0.0.5" not in result
        assert "192.168.1.100" not in result
        assert "<host>" in result or "<ip>" in result

    def test_username_scrubbed(self):
        """Database username must be replaced with <user>."""
        conn = "postgresql://analytics_ro:pass@1.2.3.4:5432/db"
        error = 'FATAL: role "analytics_ro" does not exist'
        result = _sanitize_error(error, conn, username="analytics_ro")
        assert "analytics_ro" not in result
        assert "<user>" in result

    def test_username_single_quoted_scrubbed(self):
        """Single-quoted usernames (MySQL/MSSQL) must be scrubbed."""
        conn = "mysql+pymysql://app_user:pass@1.2.3.4:3306/db"
        error = "Access denied for user 'app_user'@'1.2.3.4'"
        result = _sanitize_error(error, conn, username="app_user", host="1.2.3.4")
        assert "app_user" not in result
        assert "<user>" in result

    def test_username_mssql_login_failed_scrubbed(self):
        """MSSQL 'Login failed for user' with single quotes must be scrubbed."""
        conn = "mssql+pymssql://sa:pass@1.2.3.4:1433/db"
        error = "Login failed for user 'sa'. Reason: Password did not match."
        result = _sanitize_error(error, conn, username="sa")
        assert "'sa'" not in result
        assert "<user>" in result

    def test_port_scrubbed(self):
        """Database port must be replaced with <port>."""
        conn = "postgresql://user:pass@1.2.3.4:6543/db"
        error = "could not connect to <host> (<ip>), port 6543 failed"
        result = _sanitize_error(error, conn, port=6543)
        assert "6543" not in result
        assert "<port>" in result

    def test_port_colon_format_scrubbed(self):
        """Port in host:port format must also be replaced."""
        conn = "postgresql://user:pass@1.2.3.4:6543/db"
        error = "could not connect to <host>:6543"
        result = _sanitize_error(error, conn, port=6543)
        assert "6543" not in result
        assert ":<port>" in result

    def test_realistic_postgres_error_fully_sanitized(self):
        """Full realistic Postgres connection error must not leak any infra details."""
        conn = "postgresql://postgres:mypass@52.73.47.72:5432/postgres"
        error = (
            "(psycopg2.OperationalError) connection to server at "
            '"db.adfjtextkuilwuhzdjpf.supabase.co" (52.73.47.72), '
            "port 5432 failed: FATAL: "
            'password authentication failed for user "postgres"'
        )
        result = _sanitize_error(
            error,
            conn,
            host="52.73.47.72",
            original_host="db.adfjtextkuilwuhzdjpf.supabase.co",
            username="postgres",
            port=5432,
        )
        assert "52.73.47.72" not in result
        assert "adfjtextkuilwuhzdjpf" not in result
        assert "supabase" not in result
        assert "postgres" not in result
        assert "5432" not in result
        # The actual error type is preserved for the LLM
        assert "password authentication failed" in result


# ---------------------------------------------------------------------------
# Helpers for run()-level integration tests
# ---------------------------------------------------------------------------


def _make_credentials(
    username: str = "user", password: str = "pass"
) -> UserPasswordCredentials:
    return UserPasswordCredentials(
        id="01234567-89ab-cdef-0123-456789abcdef",
        provider="database",
        username=SecretStr(username),
        password=SecretStr(password),
        title="test creds",
    )


def _make_input(
    creds: UserPasswordCredentials,
    query: str = "SELECT 1",
    database_type: DatabaseType = DatabaseType.POSTGRES,
    host: str = "db.example.com",
    port: Optional[int] = 5432,
    database: str = "db",
    read_only: bool = False,
    timeout: int = 30,
    max_rows: int = 100,
) -> SQLQueryBlock.Input:
    return SQLQueryBlock.Input(
        query=query,
        database_type=database_type,
        host=SecretStr(host),
        port=port,
        database=database,
        read_only=read_only,
        timeout=timeout,
        max_rows=max_rows,
        credentials={  # type: ignore[arg-type]
            "provider": "database",
            "id": creds.id,
            "type": "user_password",
            "title": "t",
        },
    )


async def _collect_outputs(
    block: SQLQueryBlock,
    input_data: SQLQueryBlock.Input,
    credentials: UserPasswordCredentials,
) -> dict[str, Any]:
    """Run the block and collect all yielded (name, value) pairs."""
    outputs: dict[str, Any] = {}
    async for name, value in block.run(input_data, credentials=credentials):
        outputs[name] = value
    return outputs


# ---------------------------------------------------------------------------
# Integration tests: SQLQueryBlock.run() -- SSRF, SQLite, error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockRunSSRF:
    """SSRF protection tests exercised through the block's run() method."""

    async def test_private_ip_127_rejected(self):
        """Scenario 6: loopback 127.0.0.1 must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="127.0.0.1")
        # Mock check_host_allowed to simulate the real SSRF check raising
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address 127.0.0.1 "
                "for hostname 127.0.0.1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_private_ip_10_rejected(self):
        """Scenario 7: internal 10.x.x.x must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="10.0.0.1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address 10.0.0.1 "
                "for hostname 10.0.0.1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_unix_socket_rejected(self):
        """Scenario 8: Unix socket paths in host must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="/var/run/postgresql")
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Unix socket" in outputs["error"]

    async def test_missing_hostname_rejected(self):
        """Scenario 9: Empty hostname must be rejected."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="  ")
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "host" in outputs["error"].lower()

    async def test_private_ip_172_rejected(self):
        """172.16.x.x private range must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="172.16.0.1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address 172.16.0.1 "
                "for hostname 172.16.0.1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_private_ip_192_168_rejected(self):
        """192.168.x.x private range must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="192.168.1.1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address 192.168.1.1 "
                "for hostname 192.168.1.1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]


@pytest.mark.asyncio
class TestSQLQueryBlockRunSQLite:
    """SQLite must be explicitly disabled (scenario 13)."""

    async def test_sqlite_disabled_with_clear_error(self):
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, database_type=DatabaseType.SQLITE)
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "SQLite" in outputs["error"]
        assert "not supported" in outputs["error"]


@pytest.mark.asyncio
class TestSQLQueryBlockRunErrorHandling:
    """Error handling: connection failures and timeouts."""

    async def test_connection_failure_sanitized_no_credentials(self):
        """Connection error must not leak credentials."""
        block = SQLQueryBlock()
        creds = _make_credentials(username="admin", password="supersecret")
        input_data = _make_input(creds)
        # Mock SSRF check to allow the host
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        # Mock execute_query to raise an OperationalError with credentials in msg
        conn_url = "postgresql://admin:supersecret@db.example.com:5432/db"
        block.execute_query = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[assignment]
            OperationalError(
                f"could not connect to server: Connection refused\n"
                f'\tIs the server running on host "{conn_url}"?',
                params=None,
                orig=Exception("connection refused"),
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "supersecret" not in outputs["error"]
        assert "connect" in str(outputs["error"]).lower()

    async def test_connection_error_no_hostname_or_ip_leak(self):
        """Connection error output must not leak hostname, IP, username, or port."""
        block = SQLQueryBlock()
        creds = _make_credentials(username="postgres", password="secret")
        input_data = _make_input(
            creds, host="db.prod.supabase.co", port=5432, database="mydb"
        )
        block.check_host_allowed = AsyncMock(return_value=["52.73.47.72"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[assignment]
            OperationalError(
                'connection to server at "db.prod.supabase.co" (52.73.47.72), '
                "port 5432 failed: FATAL: "
                'password authentication failed for user "postgres"',
                params=None,
                orig=Exception("auth failed"),
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        error = outputs["error"]
        assert "db.prod.supabase.co" not in error
        assert "52.73.47.72" not in error
        assert '"postgres"' not in error
        assert "password authentication failed" in error

    async def test_query_timeout_clean_error(self):
        """Timeout yields clean user-facing message."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, query="SELECT pg_sleep(1000)", timeout=5)
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[assignment]
            OperationalError(
                "canceling statement due to statement timeout",
                params=None,
                orig=Exception("timeout"),
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "timed out" in str(outputs["error"]).lower()
        assert "5s" in str(outputs["error"])

    async def test_successful_select_returns_results(self):
        """Happy path: mocked SELECT returns correct structure."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, query="SELECT id, name FROM users LIMIT 2")
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        mock_rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        mock_cols = ["id", "name"]
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert outputs["results"] == mock_rows
        assert outputs["columns"] == mock_cols
        assert outputs["row_count"] == 2
        # SELECT does not produce affected_rows
        assert "affected_rows" not in outputs

    async def test_multi_statement_rejected_at_run_level(self):
        """Multi-statement queries are rejected before any connection attempt."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds, query="SELECT 1; DROP TABLE users", read_only=False
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "single statements" in str(outputs["error"]).lower()

    async def test_special_chars_in_password(self):
        """Passwords with special characters should work without encoding issues."""
        block = SQLQueryBlock()
        creds = _make_credentials(username="analytics_ly", password="Qw13!@#1!\\a2")
        input_data = _make_input(creds)
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        mock_rows = [{"id": 1}]
        mock_cols = ["id"]
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["results"] == mock_rows


# ---------------------------------------------------------------------------
# Write mode tests (read_only=False)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockWriteMode:
    """Tests for read_only=False (write mode) behavior."""

    async def test_insert_allowed(self):
        """INSERT should succeed when read_only=False."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="INSERT INTO users (name) VALUES ('Alice')",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["results"] == []
        assert outputs["row_count"] == 0
        assert outputs["affected_rows"] == 1

    async def test_update_allowed(self):
        """UPDATE should succeed when read_only=False."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="UPDATE users SET name = 'Bob' WHERE id = 1",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["affected_rows"] == 1

    async def test_delete_allowed(self):
        """DELETE should succeed when read_only=False."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="DELETE FROM users WHERE id = 1",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["affected_rows"] == 1

    async def test_create_table_allowed(self):
        """CREATE TABLE should succeed when read_only=False."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="CREATE TABLE new_table (id INT PRIMARY KEY, name TEXT)",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 0)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs

    async def test_drop_table_allowed(self):
        """DROP TABLE should succeed when read_only=False."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="DROP TABLE old_table",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 0)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs

    async def test_multi_statement_rejected_even_in_write_mode(self):
        """Multi-statement injection is blocked regardless of read_only setting."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="INSERT INTO t VALUES (1); DROP TABLE t",
            read_only=False,
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "single statements" in str(outputs["error"]).lower()

    async def test_ssrf_protection_still_active_in_write_mode(self):
        """SSRF checks apply regardless of read_only setting."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            host="127.0.0.1",
            query="INSERT INTO t VALUES (1)",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError("Access to blocked IP 127.0.0.1 is not allowed.")
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_sqlite_blocked_in_write_mode(self):
        """SQLite remains disabled regardless of read_only setting."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="INSERT INTO t VALUES (1)",
            database_type=DatabaseType.SQLITE,
            read_only=False,
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "SQLite" in outputs["error"]

    async def test_affected_rows_returned_for_write(self):
        """Write queries return affected_rows count."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="UPDATE users SET active = true WHERE last_login > '2024-01-01'",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([], [], 42)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["affected_rows"] == 42

    async def test_select_in_write_mode_no_affected_rows(self):
        """SELECT in write mode still returns results, not affected_rows."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT id, name FROM users LIMIT 2",
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        mock_rows = [{"id": 1, "name": "Alice"}]
        mock_cols = ["id", "name"]
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert outputs["results"] == mock_rows
        assert outputs["columns"] == mock_cols
        assert outputs["row_count"] == 1
        # affected_rows should not be yielded for SELECT (returns_rows=True -> -1)
        assert "affected_rows" not in outputs


# ---------------------------------------------------------------------------
# Read-only mode tests (read_only=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockReadOnlyMode:
    """Tests for read_only=True behavior."""

    async def test_select_works_in_read_only_mode(self):
        """SELECT should succeed in read-only mode."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1 AS col",
            read_only=True,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: ([{"col": 1}], ["col"], -1)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["results"] == [{"col": 1}]

    async def test_insert_rejected_in_read_only_mode(self):
        """INSERT should be rejected in read-only mode."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="INSERT INTO t VALUES (1)",
            read_only=True,
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "SELECT" in outputs["error"]

    async def test_read_only_default_is_true(self):
        """read_only should default to True on the Input schema."""
        creds = _make_credentials()
        # Do not pass read_only; rely on the default
        input_data = SQLQueryBlock.Input(
            query="SELECT 1",
            database_type=DatabaseType.POSTGRES,
            host=SecretStr("db.example.com"),
            port=5432,
            database="db",
            timeout=30,
            max_rows=100,
            credentials={  # type: ignore[arg-type]
                "provider": "database",
                "id": creds.id,
                "type": "user_password",
                "title": "t",
            },
        )
        assert input_data.read_only is True


# ---------------------------------------------------------------------------
# Default port tests (Thread 25: port should auto-detect per database_type)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockDefaultPort:
    """Port should default based on the selected database_type."""

    async def test_mysql_default_port_3306(self):
        """MySQL should default to port 3306 when port is 0."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MYSQL,
            port=0,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(**kwargs: Any) -> tuple[list[dict[str, Any]], list[str], int]:
            captured_conn_str["value"] = kwargs["connection_string"]
            return [{"id": 1}], ["id"], -1

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":3306/" in captured_conn_str["value"]

    async def test_mssql_default_port_1433(self):
        """MSSQL should default to port 1433 when port is 0."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MSSQL,
            port=0,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(**kwargs: Any) -> tuple[list[dict[str, Any]], list[str], int]:
            captured_conn_str["value"] = kwargs["connection_string"]
            return [{"id": 1}], ["id"], -1

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":1433/" in captured_conn_str["value"]

    async def test_postgres_default_port_5432(self):
        """PostgreSQL should default to port 5432 when port is 0."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.POSTGRES,
            port=0,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(**kwargs: Any) -> tuple[list[dict[str, Any]], list[str], int]:
            captured_conn_str["value"] = kwargs["connection_string"]
            return [{"id": 1}], ["id"], -1

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":5432/" in captured_conn_str["value"]

    async def test_explicit_port_overrides_default(self):
        """Explicitly provided port should be used regardless of database_type."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MYSQL,
            port=3307,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(**kwargs: Any) -> tuple[list[dict[str, Any]], list[str], int]:
            captured_conn_str["value"] = kwargs["connection_string"]
            return [{"id": 1}], ["id"], -1

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":3307/" in captured_conn_str["value"]


# ---------------------------------------------------------------------------
# DNS pinning tests (Thread 24: TOCTOU / DNS rebinding prevention)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockDNSPinning:
    """Connection should use the resolved IP, not the original hostname."""

    async def test_connection_uses_resolved_ip(self):
        """The connection string should contain the resolved IP, not the hostname."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="evil.example.com")
        block.check_host_allowed = AsyncMock(return_value=["93.184.216.34"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(**kwargs: Any) -> tuple[list[dict[str, Any]], list[str], int]:
            captured_conn_str["value"] = kwargs["connection_string"]
            return [{"id": 1}], ["id"], -1

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        # The resolved IP should be in the connection string, not the hostname
        assert "93.184.216.34" in captured_conn_str["value"]
        assert "evil.example.com" not in captured_conn_str["value"]
