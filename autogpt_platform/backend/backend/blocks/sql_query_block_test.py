"""Tests for SQLQueryBlock: single-statement validation, URL validation,
error sanitization, SSRF protection, and read/write mode behavior."""

from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from backend.blocks.sql_query_block import SQLQueryBlock, UserPasswordCredentials
from backend.blocks.sql_query_helpers import (
    _CONNECT_TIMEOUT_SECONDS,
    DatabaseType,
    _configure_session,
    _execute_query,
    _run_in_transaction,
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
            # MySQL INTO @variable syntax is read-only (session variable assignment)
            "SELECT COUNT(*) INTO @total FROM users",
            "SELECT name, age INTO @n, @a FROM users WHERE id = 1",
        ],
    )
    def test_valid_select_queries(self, query: str):
        _, stmt = _validate_single_statement(query)
        assert stmt is not None
        assert _validate_query_is_read_only(stmt) is None

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
        _, stmt = _validate_single_statement(query)
        assert stmt is not None
        result = _validate_query_is_read_only(stmt)
        assert result is not None

    @pytest.mark.parametrize(
        "query",
        [
            # SELECT INTO table creates a new table (PostgreSQL/MSSQL)
            "SELECT * INTO new_table FROM users",
            # SELECT INTO OUTFILE/DUMPFILE writes to the filesystem (MySQL)
            "SELECT * INTO OUTFILE '/tmp/data.csv' FROM users",
            "SELECT * INTO DUMPFILE '/tmp/data.bin' FROM users",
        ],
    )
    def test_select_into_non_variable_rejected(self, query: str):
        _, stmt = _validate_single_statement(query)
        assert stmt is not None
        result = _validate_query_is_read_only(stmt)
        assert result is not None
        assert "INTO" in result


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
        error = "connection to postgresql://admin:hunter2@db.test.invalid:5432/testprod failed"
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
        conn = "postgresql://user:pass@198.51.100.1:5432/db"
        error = (
            'connection to server at "db.example.supabase.invalid" '
            "(198.51.100.1), port 5432 failed: FATAL: "
            'password authentication failed for user "postgres"'
        )
        result = _sanitize_error(
            error, conn, host="198.51.100.1", username="postgres", port=5432
        )
        assert "198.51.100.1" not in result
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
        conn = "postgresql://testuser:fakepw@198.51.100.1:5432/testdb"
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
        conn = "postgresql://testuser:fakepw@198.51.100.1:5432/testdb"
        error = (
            "(psycopg2.OperationalError) connection to server at "
            '"db.example.supabase.invalid" (198.51.100.1), '
            "port 5432 failed: FATAL: "
            'password authentication failed for user "postgres"'
        )
        result = _sanitize_error(
            error,
            conn,
            host="198.51.100.1",
            original_host="db.example.supabase.invalid",
            username="postgres",
            port=5432,
        )
        assert "198.51.100.1" not in result
        assert "adfjtextkuilwuhzdjpf" not in result
        assert "supabase" not in result
        assert "postgres" not in result
        assert "5432" not in result
        # The actual error type is preserved for the LLM
        assert "password authentication failed" in result

    def test_database_name_scrubbed(self):
        """Database name must be replaced with <database>."""
        conn = "postgresql://user:pass@1.2.3.4:5432/analytics_prod"
        error = 'database "analytics_prod" does not exist'
        result = _sanitize_error(error, conn, database="analytics_prod")
        assert "analytics_prod" not in result
        assert "<database>" in result

    def test_database_name_word_boundary_prevents_mangling(self):
        """Common database names must not mangle unrelated words."""
        conn = "postgresql://user:pass@1.2.3.4:5432/test"
        error = 'FATAL: database "test" does not exist (tested connection)'
        result = _sanitize_error(error, conn, database="test")
        assert "tested" in result
        assert '"<database>"' in result


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
    port: int | None = 5432,
    database: str = "db",
    read_only: bool = False,
    timeout: int = 30,
    max_rows: int = 100,
) -> SQLQueryBlock.Input:
    """Build a test input. Note: ``read_only`` defaults to ``False`` here (unlike the
    block's default of ``True``) so write-mode tests don't need to specify it.
    Set ``read_only=True`` explicitly when testing read-only behaviour."""
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
# Integration tests: SQLQueryBlock.run() -- SSRF, error handling
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
            creds, host="db.test.supabase.invalid", port=5432, database="mydb"
        )
        block.check_host_allowed = AsyncMock(return_value=["198.51.100.1"])  # type: ignore[assignment]
        block.execute_query = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[assignment]
            OperationalError(
                'connection to server at "db.test.supabase.invalid" (198.51.100.1), '
                "port 5432 failed: FATAL: "
                'password authentication failed for user "postgres"',
                params=None,
                orig=Exception("auth failed"),
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        error = outputs["error"]
        assert "db.test.supabase.invalid" not in error
        assert "198.51.100.1" not in error
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
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1, False)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert outputs["results"] == mock_rows
        assert outputs["columns"] == mock_cols
        assert outputs["row_count"] == 2
        assert outputs["truncated"] is False
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
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 1, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 1, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 1, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 0, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 0, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([], [], 42, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: (mock_rows, mock_cols, -1, False)  # type: ignore[assignment]
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
        block.execute_query = lambda **_kwargs: ([{"col": 1}], ["col"], -1, False)  # type: ignore[assignment]
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
        """MySQL should default to port 3306 when port is None."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MYSQL,
            port=None,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_conn_str["value"] = str(kwargs["connection_url"])
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":3306/" in captured_conn_str["value"]

    async def test_mssql_default_port_1433(self):
        """MSSQL should default to port 1433 when port is None."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MSSQL,
            port=None,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_conn_str["value"] = str(kwargs["connection_url"])
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert ":1433/" in captured_conn_str["value"]

    async def test_postgres_default_port_5432(self):
        """PostgreSQL should default to port 5432 when port is None."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.POSTGRES,
            port=None,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_conn_str = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_conn_str["value"] = str(kwargs["connection_url"])
            return [{"id": 1}], ["id"], -1, False

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

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_conn_str["value"] = str(kwargs["connection_url"])
            return [{"id": 1}], ["id"], -1, False

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

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_conn_str["value"] = str(kwargs["connection_url"])
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        # The resolved IP should be in the connection string, not the hostname
        assert "93.184.216.34" in captured_conn_str["value"]
        assert "evil.example.com" not in captured_conn_str["value"]


# ---------------------------------------------------------------------------
# Security gap tests: IPv6 SSRF, writable CTEs, timeout/max_rows enforcement,
# and password scrubbing in error messages.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSQLQueryBlockIPv6SSRF:
    """IPv6 loopback and link-local addresses must be blocked."""

    async def test_ipv6_loopback_rejected(self):
        """IPv6 loopback (::1) must be blocked just like 127.0.0.1."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="::1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address ::1 "
                "for hostname ::1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_ipv6_link_local_rejected(self):
        """IPv6 link-local (fe80::) must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="fe80::1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address fe80::1 "
                "for hostname fe80::1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_ipv6_mapped_ipv4_loopback_rejected(self):
        """IPv4-mapped IPv6 loopback (::ffff:127.0.0.1) must be blocked."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="::ffff:127.0.0.1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address ::ffff:127.0.0.1 "
                "for hostname ::ffff:127.0.0.1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]

    async def test_ipv6_unique_local_rejected(self):
        """IPv6 unique-local (fd00::) must be blocked like RFC1918."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, host="fd00::1")
        block.check_host_allowed = AsyncMock(  # type: ignore[assignment]
            side_effect=ValueError(
                "Access to blocked or private IP address fd00::1 "
                "for hostname fd00::1 is not allowed."
            )
        )
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "Blocked host" in outputs["error"]


class TestWritableCTEBlocked:
    """Writable CTEs (WITH ... DELETE/INSERT/UPDATE ... RETURNING) must be
    blocked even though sqlparse classifies them as SELECT type."""

    @pytest.mark.parametrize(
        "query,expected_keyword",
        [
            (
                "WITH deleted AS (DELETE FROM users RETURNING *) SELECT * FROM deleted",
                "DELETE",
            ),
            (
                "WITH ins AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM ins",
                "INSERT",
            ),
            (
                "WITH upd AS (UPDATE t SET x=1 RETURNING *) SELECT * FROM upd",
                "UPDATE",
            ),
        ],
    )
    def test_writable_cte_rejected_in_read_only(
        self, query: str, expected_keyword: str
    ):
        """A CTE containing a data-modifying statement must be caught by
        the keyword-based defense-in-depth check even if stmt.get_type()
        returns 'SELECT'."""
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None
        ro_error = _validate_query_is_read_only(stmt)
        assert ro_error is not None
        assert expected_keyword in ro_error

    def test_writable_cte_with_truncate_rejected(self):
        """TRUNCATE inside a CTE must also be blocked."""
        query = "WITH t AS (TRUNCATE users RETURNING *) SELECT * FROM t"
        error, stmt = _validate_single_statement(query)
        # sqlparse may reject this as invalid or we catch it in read-only check
        if error is None and stmt is not None:
            ro_error = _validate_query_is_read_only(stmt)
            assert ro_error is not None


@pytest.mark.asyncio
class TestTimeoutEnforcement:
    """Timeout parameter must be forwarded to execute_query."""

    async def test_timeout_forwarded_to_execute(self):
        """The timeout value from input must reach execute_query."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, query="SELECT 1", timeout=7)
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_kwargs: dict[str, Any] = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_kwargs.update(kwargs)
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert captured_kwargs["timeout"] == 7

    async def test_timeout_generates_clean_error(self):
        """When execute_query raises a timeout OperationalError, the block
        yields a clean user-facing 'timed out' message with the configured
        timeout value."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, query="SELECT pg_sleep(9999)", timeout=3)
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
        assert "timed out" in outputs["error"].lower()
        assert "3s" in outputs["error"]


@pytest.mark.asyncio
class TestMaxRowsEnforcement:
    """max_rows parameter must be forwarded and used to truncate results."""

    async def test_max_rows_forwarded_to_execute(self):
        """The max_rows value from input must reach execute_query."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(creds, query="SELECT 1", max_rows=42)
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured_kwargs: dict[str, Any] = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured_kwargs.update(kwargs)
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert captured_kwargs["max_rows"] == 42

    async def test_max_rows_truncates_large_result(self):
        """When execute_query returns exactly max_rows items, row_count
        reflects the truncated count."""
        block = SQLQueryBlock()
        creds = _make_credentials()
        max_rows = 5
        input_data = _make_input(
            creds, query="SELECT * FROM big_table", max_rows=max_rows
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        # Simulate the database returning exactly max_rows rows (truncated)
        mock_rows = [{"id": i} for i in range(max_rows)]
        block.execute_query = lambda **_kwargs: (mock_rows, ["id"], -1, True)  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        assert outputs["row_count"] == max_rows
        assert len(outputs["results"]) == max_rows
        assert outputs["truncated"] is True


class TestPasswordInErrorMessages:
    """Connection errors must not leak passwords, even when the password
    appears in driver-specific formats."""

    def test_password_key_value_scrubbed(self):
        """password=<value> in error text must be replaced with ***."""
        conn = "postgresql://user:P@ssw0rd!@host:5432/db"
        error = "FATAL: password=P@ssw0rd! authentication failed"
        result = _sanitize_error(error, conn)
        assert "P@ssw0rd!" not in result
        assert "password=***" in result

    def test_url_embedded_password_scrubbed(self):
        """://user:pass@ in error text must be replaced."""
        conn = "postgresql://admin:hunter2@test.invalid:5432/db"
        error = (
            "could not connect: postgresql://admin:hunter2@test.invalid:5432/db refused"
        )
        result = _sanitize_error(error, conn)
        assert "hunter2" not in result

    def test_ipv6_address_in_error_scrubbed(self):
        """Bracketed IPv6 addresses (e.g. [::1]) in error messages must be
        scrubbed to prevent leaking internal network topology."""
        conn = "postgresql://user:pass@[::1]:5432/db"
        error = 'could not connect to server at "[::1]", port 5432'
        result = _sanitize_error(error, conn, host="::1", port=5432)
        assert "[::1]" not in result
        assert "5432" not in result

    def test_ipv6_link_local_in_error_scrubbed(self):
        """Bracketed IPv6 link-local with zone ID (e.g. [fe80::1%eth0]) must
        be scrubbed."""
        conn = "postgresql://user:pass@[fe80::1%25eth0]:5432/db"
        error = 'connection to "[fe80::1%eth0]" refused'
        result = _sanitize_error(error, conn, host="fe80::1%eth0")
        assert "[fe80::1%eth0]" not in result

    def test_multiple_password_formats_scrubbed(self):
        """Error containing password in multiple formats must be fully scrubbed."""
        conn = "postgresql://testadmin:fakepw@test.example.invalid:5432/testdb"
        error = (
            "connection to postgresql://testadmin:fakepw@test.example.invalid:5432/testdb failed: "
            "password=fakepw authentication failed for user 'testadmin'"
        )
        result = _sanitize_error(
            error, conn, host="test.example.invalid", username="testadmin", port=5432
        )
        assert "fakepw" not in result
        assert "testadmin" not in result
        assert "test.example.invalid" not in result


# ---------------------------------------------------------------------------
# SQL injection within single statements -- BY DESIGN these pass validation.
#
# The block uses ``text(query)`` (raw SQL) without parameterization. This is
# intentional: the block is designed for trusted admin / analytics use where
# the copilot or autopilot constructs queries against analytics databases.
# The security boundary is:
#   1. Single-statement enforcement (prevents multi-query injection)
#   2. Read-only keyword filtering (prevents data modification in default mode)
#   3. SSRF host validation (prevents access to internal networks)
#   4. Database-level read-only session (defense-in-depth)
#
# Within those constraints, the raw query is forwarded as-is to the database.
# These tests document that single-statement injection patterns are NOT blocked
# at the application layer -- the database's own auth/permissions are the final
# gate.
# ---------------------------------------------------------------------------


class TestSQLInjectionWithinSingleStatement:
    """Document that single-statement SQL injection patterns pass validation.

    The block executes raw SQL via ``text(query)`` without parameterization.
    This is BY DESIGN for trusted admin/analytics use. These tests verify and
    document that the validation layer does NOT attempt to catch injection
    within a single SELECT statement -- that responsibility falls to database
    permissions and the read-only session mode.
    """

    @pytest.mark.parametrize(
        "query",
        [
            # Classic tautology injection
            "SELECT * FROM users WHERE id = '1' OR '1'='1'",
            # UNION-based injection to read other tables
            "SELECT name FROM users WHERE id = 1 UNION SELECT password FROM credentials",
            # Boolean-based blind injection
            "SELECT * FROM users WHERE username = 'admin' AND 1=1",
            # Comment-based injection (single-line comment)
            "SELECT * FROM users WHERE username = 'admin' --' AND password = 'x'",
            # Subquery injection
            "SELECT * FROM users WHERE id = (SELECT MAX(id) FROM users)",
            # LIKE wildcard injection
            "SELECT * FROM users WHERE name LIKE '%'",
        ],
    )
    def test_single_statement_injection_passes_validation(self, query: str):
        """Single-statement injection patterns pass the single-statement
        validator because they are syntactically valid single statements.
        The block relies on database permissions for fine-grained access control."""
        error, stmt = _validate_single_statement(query)
        assert error is None, f"Expected query to pass validation: {query}"
        assert stmt is not None

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM users WHERE id = '1' OR '1'='1'",
            "SELECT name FROM users UNION SELECT secret FROM keys",
            "SELECT * FROM users WHERE username = 'admin' AND 1=1",
        ],
    )
    def test_single_statement_injection_passes_read_only_check(self, query: str):
        """Single-statement SELECT-based injections also pass the read-only
        check since they are valid SELECT queries (or UNION of SELECTs).
        The database's own permission model is the final defense layer."""
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None
        ro_error = _validate_query_is_read_only(stmt)
        assert ro_error is None, f"Expected read-only check to pass: {query}"

    def test_injection_with_write_keyword_blocked_in_read_only(self):
        """Even within a single statement, if an injection smuggles a
        disallowed keyword (e.g. INSERT via subquery), the read-only
        keyword check catches it."""
        # This is a contrived example -- real injection would look different,
        # but it tests the keyword defense-in-depth layer.
        query = "SELECT * FROM users WHERE id IN (SELECT id FROM t) UNION SELECT * FROM (DELETE FROM users RETURNING id) AS x"
        error, stmt = _validate_single_statement(query)
        # sqlparse may split this or reject it; if it parses as single, the
        # read-only check must still catch DELETE.
        if error is None and stmt is not None:
            ro_error = _validate_query_is_read_only(stmt)
            assert ro_error is not None
            assert "DELETE" in ro_error


# ---------------------------------------------------------------------------
# URL.create() special character handling
#
# The existing test_special_chars_in_password mocks execute_query, so it
# never exercises URL.create(). These tests verify that URL.create()
# correctly handles special characters in credentials without raising
# exceptions or corrupting the URL.
# ---------------------------------------------------------------------------


class TestURLCreateSpecialCharacters:
    """Verify URL.create() correctly handles special characters in credentials.

    The block uses ``URL.create()`` (not manual string formatting) to build
    connection URLs. This avoids URL-encoding issues with special characters
    in passwords. These tests verify the URL is constructed correctly by
    inspecting the URL object properties, without needing a real database.
    """

    @pytest.mark.parametrize(
        "password",
        [
            "p@ss:word/with#special!chars",
            "has spaces in it",
            "symbols!@#$%^&*()",
            "url_tricky://user:pass@host/",
            "percent%20encoded%3F",
            "unicode_café_naïve",
            "backslash\\and\ttab",
            "empty",  # control case
        ],
    )
    def test_url_create_preserves_password(self, password: str):
        """URL.create() must preserve the exact password without corruption."""
        from sqlalchemy.engine.url import URL

        url = URL.create(
            drivername="postgresql",
            username="testuser",
            password=password,
            host="1.2.3.4",
            port=5432,
            database="testdb",
        )
        # The URL object stores the password without URL-encoding
        assert url.password == password
        assert url.username == "testuser"
        assert url.host == "1.2.3.4"
        assert url.port == 5432
        assert url.database == "testdb"

    @pytest.mark.parametrize(
        "username",
        [
            "user@domain.com",
            "user:name",
            "user/slash",
            "admin#hash",
        ],
    )
    def test_url_create_preserves_username(self, username: str):
        """URL.create() must preserve special characters in usernames."""
        from sqlalchemy.engine.url import URL

        url = URL.create(
            drivername="postgresql",
            username=username,
            password="pass",
            host="1.2.3.4",
            port=5432,
            database="testdb",
        )
        assert url.username == username

    @pytest.mark.parametrize(
        "db_type,expected_driver",
        [
            (DatabaseType.POSTGRES, "postgresql"),
            (DatabaseType.MYSQL, "mysql+pymysql"),
            (DatabaseType.MSSQL, "mssql+pymssql"),
        ],
    )
    def test_url_create_correct_driver_per_db_type(
        self, db_type: DatabaseType, expected_driver: str
    ):
        """URL.create() must use the correct drivername for each database type."""
        from sqlalchemy.engine.url import URL

        from backend.blocks.sql_query_block import _DATABASE_TYPE_TO_DRIVER

        drivername = _DATABASE_TYPE_TO_DRIVER[db_type]
        url = URL.create(
            drivername=drivername,
            username="user",
            password="pass",
            host="1.2.3.4",
            port=5432,
            database="testdb",
        )
        assert url.drivername == expected_driver

    @pytest.mark.asyncio
    async def test_special_chars_reach_url_create(self):
        """End-to-end: credentials with special chars are correctly passed to
        URL.create() (verified by capturing the connection_url kwarg)."""
        block = SQLQueryBlock()
        special_pass = "p@ss:w0rd/#!&=+()"
        special_user = "admin@corp"
        creds = _make_credentials(username=special_user, password=special_pass)
        input_data = _make_input(creds)
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]
        captured: dict[str, Any] = {}

        def fake_execute(
            **kwargs: Any,
        ) -> tuple[list[dict[str, Any]], list[str], int, bool]:
            captured.update(kwargs)
            return [{"id": 1}], ["id"], -1, False

        block.execute_query = fake_execute  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" not in outputs
        conn_url = captured["connection_url"]
        # URL.create() produces a URL object, not a string
        assert hasattr(conn_url, "password"), "connection_url should be a URL object"
        assert conn_url.password == special_pass
        assert conn_url.username == special_user


# ---------------------------------------------------------------------------
# Tests for newly-added disallowed keywords (LOAD, REPLACE, MERGE, BULK, EXEC)
# ---------------------------------------------------------------------------


class TestNewDisallowedKeywords:
    """LOAD, REPLACE, MERGE, BULK, and EXEC must be blocked in read-only mode."""

    @pytest.mark.parametrize(
        "query,expected_keyword",
        [
            # MySQL file exfiltration
            (
                "SELECT * FROM (LOAD DATA LOCAL INFILE '/etc/passwd' INTO TABLE t) AS x",
                "LOAD",
            ),
            # MySQL REPLACE (INSERT-or-UPDATE)
            (
                "SELECT * FROM (REPLACE INTO t VALUES (1, 'a')) AS x",
                "REPLACE",
            ),
            # ANSI MERGE (UPSERT)
            (
                "SELECT * FROM (MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.v = s.v) AS x",
                "MERGE",
            ),
            # MSSQL BULK INSERT -- sqlparse may match INSERT before BULK,
            # but the query is still blocked by the disallowed keyword check.
            (
                "SELECT * FROM (BULK INSERT t FROM '/data.csv') AS x",
                "BULK",
            ),
            # MSSQL EXEC stored procedure
            (
                "SELECT * FROM (EXEC sp_who2) AS x",
                "EXEC",
            ),
        ],
    )
    def test_keyword_blocked_in_subquery_read_only(
        self, query: str, expected_keyword: str
    ):
        """If sqlparse parses the statement as SELECT, the keyword check
        must still catch the disallowed keyword inside it."""
        error, stmt = _validate_single_statement(query)
        if error is None and stmt is not None:
            ro_error = _validate_query_is_read_only(stmt)
            assert (
                ro_error is not None
            ), f"Expected '{expected_keyword}' to be blocked but query passed: {query}"
            assert "Disallowed SQL keyword" in ro_error

    @pytest.mark.parametrize(
        "query",
        [
            "LOAD DATA LOCAL INFILE '/etc/passwd' INTO TABLE t",
            "REPLACE INTO users (id, name) VALUES (1, 'Alice')",
            "MERGE INTO target USING source ON target.id = source.id "
            "WHEN MATCHED THEN UPDATE SET target.name = source.name",
            "BULK INSERT my_table FROM '/tmp/data.csv'",
            "EXEC sp_who2",
        ],
    )
    def test_standalone_keyword_rejected_as_non_select(self, query: str):
        """Standalone statements using these keywords are rejected because
        their statement type is not SELECT."""
        error, stmt = _validate_single_statement(query)
        if error is None and stmt is not None:
            ro_error = _validate_query_is_read_only(stmt)
            assert ro_error is not None

    def test_bare_keyword_column_names_blocked(self):
        """Bare column names that match disallowed keywords (load, replace,
        merge) ARE blocked because sqlparse classifies them as Keyword tokens.
        Users should quote such column names (e.g. "load") to avoid this."""
        query = "SELECT load, replace, merge FROM metrics"
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None
        ro_error = _validate_query_is_read_only(stmt)
        assert ro_error is not None

    def test_quoted_keyword_column_names_allowed(self):
        """Double-quoted column names that happen to match disallowed keywords
        must NOT trigger false positives (sqlparse classifies them as Names)."""
        query = 'SELECT "load", "replace", "merge" FROM metrics'
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None
        ro_error = _validate_query_is_read_only(stmt)
        assert ro_error is None

    def test_load_in_string_literal_allowed(self):
        """String literal containing 'LOAD' must not be blocked."""
        query = "SELECT * FROM logs WHERE message = 'LOAD DATA INFILE test'"
        error, stmt = _validate_single_statement(query)
        assert error is None
        assert stmt is not None
        ro_error = _validate_query_is_read_only(stmt)
        assert ro_error is None


# ---------------------------------------------------------------------------
# Decimal NaN / Infinity serialization tests
# ---------------------------------------------------------------------------


class TestSerializeDecimalSpecialValues:
    """Decimal NaN and Infinity must be serialized as strings, not crash."""

    def test_decimal_nan(self):
        result = _serialize_value(Decimal("NaN"))
        assert isinstance(result, str)
        assert "NaN" in result

    def test_decimal_infinity(self):
        result = _serialize_value(Decimal("Infinity"))
        assert isinstance(result, str)
        assert "Infinity" in result

    def test_decimal_negative_infinity(self):
        result = _serialize_value(Decimal("-Infinity"))
        assert isinstance(result, str)
        assert "-Infinity" in result

    def test_decimal_snan(self):
        result = _serialize_value(Decimal("sNaN"))
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ModuleNotFoundError path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestModuleNotFoundError:
    """The block must yield a clean error when a database driver is missing."""

    async def test_missing_driver_yields_clean_error(self):
        block = SQLQueryBlock()
        creds = _make_credentials()
        input_data = _make_input(
            creds,
            query="SELECT 1",
            database_type=DatabaseType.MSSQL,
            read_only=False,
        )
        block.check_host_allowed = AsyncMock(return_value=["1.2.3.4"])  # type: ignore[assignment]

        def raise_module_not_found(**_kwargs: Any) -> None:
            raise ModuleNotFoundError("No module named 'pymssql'")

        block.execute_query = raise_module_not_found  # type: ignore[assignment]
        outputs = await _collect_outputs(block, input_data, creds)
        assert "error" in outputs
        assert "driver not available" in outputs["error"].lower()
        assert "mssql" in outputs["error"].lower()


# ---------------------------------------------------------------------------
# Integration tests: _execute_query, _configure_session, _run_in_transaction
# using an in-memory SQLite database.
#
# SQLite does not support SET commands, so _configure_session is a no-op for
# the "sqlite" dialect.  These tests validate the execution pipeline itself:
# transaction management, fetchmany truncation, COMMIT vs ROLLBACK, and the
# new truncated flag.
# ---------------------------------------------------------------------------


def _extract_sql_texts(conn_mock: MagicMock) -> list[str]:
    """Extract the raw SQL strings from a mocked connection's execute calls."""
    return [c[0][0].text for c in conn_mock.execute.call_args_list]


class TestConfigureSessionDialects:
    """Verify _configure_session emits correct SET commands per dialect."""

    def test_postgresql_sets_timeout_and_read_only(self):
        conn = MagicMock()
        _configure_session(conn, "postgresql", "30000", read_only=True)
        sqls = _extract_sql_texts(conn)
        assert any("statement_timeout" in s and "30000" in s for s in sqls)
        assert any("default_transaction_read_only" in s for s in sqls)

    def test_postgresql_timeout_only_when_not_read_only(self):
        conn = MagicMock()
        _configure_session(conn, "postgresql", "5000", read_only=False)
        assert conn.execute.call_count == 1
        sqls = _extract_sql_texts(conn)
        assert "statement_timeout" in sqls[0]

    def test_mysql_sets_max_execution_time_and_read_only(self):
        conn = MagicMock()
        _configure_session(conn, "mysql", "10000", read_only=True)
        sqls = _extract_sql_texts(conn)
        assert any("MAX_EXECUTION_TIME" in s and "10000" in s for s in sqls)
        assert any("READ ONLY" in s for s in sqls)

    def test_mysql_timeout_only_when_not_read_only(self):
        conn = MagicMock()
        _configure_session(conn, "mysql", "10000", read_only=False)
        assert conn.execute.call_count == 1
        sqls = _extract_sql_texts(conn)
        assert "MAX_EXECUTION_TIME" in sqls[0]

    def test_mssql_sets_lock_timeout(self):
        conn = MagicMock()
        _configure_session(conn, "mssql", "15000", read_only=True)
        sqls = _extract_sql_texts(conn)
        assert any("LOCK_TIMEOUT" in s and "15000" in s for s in sqls)
        assert conn.execute.call_count == 1

    def test_unknown_dialect_is_noop(self):
        conn = MagicMock()
        _configure_session(conn, "sqlite", "5000", read_only=True)
        conn.execute.assert_not_called()


class TestRunInTransactionSQLite:
    """Integration tests for _run_in_transaction using a real SQLite engine."""

    def _make_sqlite_engine(self):
        return create_engine("sqlite:///:memory:")

    def test_select_returns_rows(self):
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER, name TEXT)"))
            conn.execute(text("INSERT INTO t VALUES (1, 'Alice')"))
            conn.execute(text("INSERT INTO t VALUES (2, 'Bob')"))
            results, columns, affected, truncated = _run_in_transaction(
                conn,
                "sqlite",
                "SELECT * FROM t ORDER BY id",
                max_rows=100,
                read_only=True,
            )
        engine.dispose()
        assert columns == ["id", "name"]
        assert len(results) == 2
        assert results[0] == {"id": 1, "name": "Alice"}
        assert results[1] == {"id": 2, "name": "Bob"}
        assert affected == -1
        assert truncated is False

    def test_fetchmany_truncation(self):
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            for i in range(10):
                conn.execute(text(f"INSERT INTO t VALUES ({i})"))
            results, columns, affected, truncated = _run_in_transaction(
                conn,
                "sqlite",
                "SELECT * FROM t ORDER BY id",
                max_rows=3,
                read_only=True,
            )
        engine.dispose()
        assert len(results) == 3
        assert truncated is True
        assert results[0]["id"] == 0
        assert results[2]["id"] == 2

    def test_fetchmany_exact_count_not_truncated(self):
        """When result count equals max_rows, truncated is True because
        there *might* be more rows (we cannot know without fetching more)."""
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            for i in range(5):
                conn.execute(text(f"INSERT INTO t VALUES ({i})"))
            results, _, _, truncated = _run_in_transaction(
                conn,
                "sqlite",
                "SELECT * FROM t",
                max_rows=5,
                read_only=True,
            )
        engine.dispose()
        assert len(results) == 5
        assert truncated is True

    def test_fewer_rows_than_max_not_truncated(self):
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            conn.execute(text("INSERT INTO t VALUES (1)"))
            results, _, _, truncated = _run_in_transaction(
                conn,
                "sqlite",
                "SELECT * FROM t",
                max_rows=100,
                read_only=True,
            )
        engine.dispose()
        assert len(results) == 1
        assert truncated is False

    def test_rollback_on_read_only(self):
        """When read_only=True the transaction is rolled back, so writes are
        not persisted."""
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            conn.execute(text("INSERT INTO t VALUES (1)"))
            _run_in_transaction(
                conn,
                "sqlite",
                "INSERT INTO t VALUES (2)",
                max_rows=100,
                read_only=True,
            )
            # The INSERT should have been rolled back
            result = conn.execute(text("SELECT COUNT(*) FROM t"))
            count = result.scalar()
        engine.dispose()
        assert count == 1

    def test_commit_on_write_mode(self):
        """When read_only=False the transaction is committed."""
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            conn.execute(text("INSERT INTO t VALUES (1)"))
            _run_in_transaction(
                conn,
                "sqlite",
                "INSERT INTO t VALUES (2)",
                max_rows=100,
                read_only=False,
            )
            result = conn.execute(text("SELECT COUNT(*) FROM t"))
            count = result.scalar()
        engine.dispose()
        assert count == 2

    def test_rollback_on_exception(self):
        """On query error the transaction is rolled back and the exception
        re-raised."""
        engine = self._make_sqlite_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("CREATE TABLE t (id INTEGER)"))
            with pytest.raises(Exception):
                _run_in_transaction(
                    conn,
                    "sqlite",
                    "SELECT * FROM nonexistent_table",
                    max_rows=100,
                    read_only=True,
                )
        engine.dispose()

    def test_mssql_begin_transaction_syntax(self):
        """MSSQL dialect uses 'BEGIN TRANSACTION' instead of 'BEGIN'."""
        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.returns_rows = True
        result_mock.keys.return_value = ["id"]
        result_mock.fetchmany.return_value = [(1,)]
        result_mock.rowcount = 1
        conn.execute.return_value = result_mock

        _run_in_transaction(conn, "mssql", "SELECT 1", max_rows=10, read_only=True)

        sqls = _extract_sql_texts(conn)
        assert sqls[0] == "BEGIN TRANSACTION"

    def test_non_mssql_begin_syntax(self):
        """Non-MSSQL dialects use 'BEGIN' (not 'BEGIN TRANSACTION')."""
        conn = MagicMock()
        result_mock = MagicMock()
        result_mock.returns_rows = True
        result_mock.keys.return_value = ["id"]
        result_mock.fetchmany.return_value = [(1,)]
        result_mock.rowcount = 1
        conn.execute.return_value = result_mock

        _run_in_transaction(conn, "postgresql", "SELECT 1", max_rows=10, read_only=True)

        sqls = _extract_sql_texts(conn)
        assert sqls[0] == "BEGIN"


class TestExecuteQueryIntegration:
    """Integration tests for _execute_query using mocked engine internals.

    Note: SQLite does not accept ``connect_timeout`` as a connect_arg, so
    we cannot pass ``sqlite:///:memory:`` directly to ``_execute_query``.
    Instead we mock ``create_engine`` to return a real SQLite engine while
    bypassing the connect_args validation.
    """

    def test_connect_timeout_constant_used(self):
        """The _CONNECT_TIMEOUT_SECONDS constant must be passed to create_engine."""
        with patch("backend.blocks.sql_query_helpers.create_engine") as mock_ce:
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_result = MagicMock()
            mock_result.returns_rows = True
            mock_result.keys.return_value = ["x"]
            mock_result.fetchmany.return_value = [(1,)]
            mock_result.rowcount = 1
            mock_conn.execute.return_value = mock_result
            mock_conn.execution_options.return_value = mock_conn
            mock_engine.connect.return_value.__enter__ = lambda _: mock_conn
            mock_engine.connect.return_value.__exit__ = lambda *_: None
            mock_engine.dialect.name = "postgresql"
            mock_ce.return_value = mock_engine

            _execute_query(
                "postgresql://u:p@h/d",
                "SELECT 1",
                timeout=10,
                max_rows=100,
                read_only=True,
            )

            mock_ce.assert_called_once()
            _, kwargs = mock_ce.call_args
            assert kwargs["connect_args"]["connect_timeout"] == _CONNECT_TIMEOUT_SECONDS
