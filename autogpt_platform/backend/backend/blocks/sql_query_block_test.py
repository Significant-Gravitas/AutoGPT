"""Tests for SQLQueryBlock query validation, URL validation, and error sanitization."""

from datetime import date, datetime
from decimal import Decimal

import pytest

from backend.blocks.sql_query_block import (
    DatabaseType,
    _sanitize_error,
    _serialize_value,
    _validate_connection_url,
    _validate_query_is_read_only,
)


class TestValidateQueryIsReadOnly:
    """Tests for _validate_query_is_read_only defense-in-depth validation."""

    # --- Valid queries that should pass ---

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 1",
            "SELECT * FROM users",
            "select count(*) from analytics.daily_active_users",
            "SELECT id, name FROM users WHERE id = 1",
            "WITH cte AS (SELECT 1) SELECT * FROM cte",
            "with recursive tree as (select 1) select * from tree",
            "SELECT * FROM users LIMIT 100",
            "SELECT * FROM users;",
            "  SELECT 1  ",
        ],
    )
    def test_valid_select_queries(self, query: str):
        assert _validate_query_is_read_only(query) is None

    # --- Column names that look like keywords should NOT be blocked ---

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT comment FROM posts",
            "SELECT analyze, lock_count FROM metrics",
            "SELECT * FROM cluster_stats",
            "SELECT reindex_time FROM maintenance_log",
            "SELECT vacuum_count FROM pg_stat_tables",
        ],
    )
    def test_ambiguous_column_names_allowed(self, query: str):
        assert _validate_query_is_read_only(query) is None

    # --- Invalid queries that should be rejected ---

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO users VALUES (1, 'test')",
            "UPDATE users SET name = 'test'",
            "DELETE FROM users WHERE id = 1",
            "DROP TABLE users",
            "ALTER TABLE users ADD COLUMN age INT",
            "CREATE TABLE new_table (id INT)",
            "TRUNCATE users",
            "GRANT SELECT ON users TO public",
            "REVOKE SELECT ON users FROM public",
            "COPY users TO '/tmp/out.csv'",
        ],
    )
    def test_disallowed_statements_rejected(self, query: str):
        result = _validate_query_is_read_only(query)
        assert result is not None
        assert "Only SELECT queries are allowed" in result or "Disallowed" in result

    # --- Defense-in-depth: disallowed keywords inside SELECT ---

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM users; DELETE FROM users",
            "SELECT * FROM users; DROP TABLE users",
            "SELECT 1; INSERT INTO users VALUES (1)",
            "WITH cte AS (SELECT 1) INSERT INTO users SELECT * FROM cte",
        ],
    )
    def test_multi_statement_and_cte_injection_rejected(self, query: str):
        result = _validate_query_is_read_only(query)
        assert result is not None

    # --- Quoted comment injection / multi-statement bypass ---

    @pytest.mark.parametrize(
        "query",
        [
            # Timeout bypass via quoted comment injection (CVE-like)
            "SELECT '--'; SET LOCAL statement_timeout = 0; SELECT pg_sleep(1000)",
            # Multi-statement with hidden SET
            "SELECT 1; SET search_path TO public",
            # RESET after SELECT
            "SELECT 1; RESET ALL",
        ],
    )
    def test_multi_statement_injection_rejected(self, query: str):
        result = _validate_query_is_read_only(query)
        assert result is not None

    # --- String literals containing keywords should NOT be rejected ---

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT '--not-a-comment' AS label FROM t",
            "SELECT * FROM users WHERE action = 'DELETE'",
            "SELECT * FROM users WHERE status = 'GRANT'",
        ],
    )
    def test_keywords_in_string_literals_allowed(self, query: str):
        assert _validate_query_is_read_only(query) is None

    # --- Comment-wrapped attacks ---

    @pytest.mark.parametrize(
        "query",
        [
            "SELECT 1 /* DROP TABLE users */",
            "SELECT 1 -- DROP TABLE users",
        ],
    )
    def test_comments_stripped_before_validation(self, query: str):
        # After comment stripping, these should be valid SELECT queries
        assert _validate_query_is_read_only(query) is None

    def test_dangerous_query_hidden_in_comment_block(self):
        # The real dangerous part is after the comment
        query = "/* SELECT 1 */ DROP TABLE users"
        result = _validate_query_is_read_only(query)
        assert result is not None

    # --- Edge cases ---

    def test_empty_query(self):
        assert _validate_query_is_read_only("") == "Query is empty."

    def test_whitespace_only_query(self):
        assert _validate_query_is_read_only("   ") == "Query is empty."

    def test_comment_only_query(self):
        result = _validate_query_is_read_only("-- just a comment")
        assert result is not None  # Either "empty" or rejected

    def test_semicolon_only_query(self):
        assert _validate_query_is_read_only(";") == "Query is empty."


class TestValidateConnectionUrl:
    """Tests for _validate_connection_url database type matching."""

    def test_postgres_url_matches_postgres_type(self):
        url = "postgresql://user:pass@host:5432/db"
        assert _validate_connection_url(url, DatabaseType.POSTGRES) is None

    def test_postgres_with_driver_matches(self):
        url = "postgresql+psycopg2://user:pass@host:5432/db"
        assert _validate_connection_url(url, DatabaseType.POSTGRES) is None

    def test_mysql_url_matches_mysql_type(self):
        url = "mysql://user:pass@host:3306/db"
        assert _validate_connection_url(url, DatabaseType.MYSQL) is None

    def test_mysql_with_driver_matches(self):
        url = "mysql+pymysql://user:pass@host:3306/db"
        assert _validate_connection_url(url, DatabaseType.MYSQL) is None

    def test_sqlite_url_matches_sqlite_type(self):
        url = "sqlite:///path/to/db.sqlite"
        assert _validate_connection_url(url, DatabaseType.SQLITE) is None

    def test_mssql_url_matches_mssql_type(self):
        url = "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17"
        assert _validate_connection_url(url, DatabaseType.MSSQL) is None

    def test_postgres_url_rejects_mysql_type(self):
        url = "postgresql://user:pass@host:5432/db"
        result = _validate_connection_url(url, DatabaseType.MYSQL)
        assert result is not None
        assert "does not match" in result

    def test_mysql_url_rejects_postgres_type(self):
        url = "mysql://user:pass@host:3306/db"
        result = _validate_connection_url(url, DatabaseType.POSTGRES)
        assert result is not None
        assert "does not match" in result

    def test_invalid_url_returns_error(self):
        result = _validate_connection_url("not a url at all", DatabaseType.POSTGRES)
        assert result is not None
        assert "Invalid" in result


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
    """Tests for _sanitize_error credential scrubbing."""

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
