import re
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any

import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL


class DatabaseType(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
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
    # MySQL file exfiltration: LOAD DATA LOCAL INFILE reads server/client files
    "LOAD",
    # MySQL REPLACE is INSERT-or-UPDATE; data modification
    "REPLACE",
    # ANSI MERGE (UPSERT) modifies data
    "MERGE",
    # MSSQL BULK INSERT loads external files into tables
    "BULK",
    # MSSQL EXEC / EXEC sp_name runs stored procedures (arbitrary code)
    "EXEC",
}

# Map DatabaseType enum values to the expected SQLAlchemy driver prefix.
_DATABASE_TYPE_TO_DRIVER = {
    DatabaseType.POSTGRES: "postgresql",
    DatabaseType.MYSQL: "mysql+pymysql",
    DatabaseType.MSSQL: "mssql+pymssql",
}

# Connection timeout in seconds passed to the DBAPI driver (connect_timeout /
# login_timeout).  This bounds how long the driver waits to establish a TCP
# connection to the database server.  It is separate from the per-statement
# timeout configured via SET commands inside _configure_session().
_CONNECT_TIMEOUT_SECONDS = 10

# Default ports for each database type.
_DATABASE_TYPE_DEFAULT_PORT = {
    DatabaseType.POSTGRES: 5432,
    DatabaseType.MYSQL: 3306,
    DatabaseType.MSSQL: 1433,
}


def _sanitize_error(
    error_msg: str,
    connection_string: str,
    *,
    host: str = "",
    original_host: str = "",
    username: str = "",
    port: int = 0,
    database: str = "",
) -> str:
    """Remove connection string, credentials, and infrastructure details
    from error messages so they are safe to expose to the LLM.

    Scrubs:
    - The full connection string
    - URL-embedded credentials (``://user:pass@``)
    - ``password=<value>`` key-value pairs
    - The database hostname / IP used for the connection
    - The original (pre-resolution) hostname provided by the user
    - Any IPv4 addresses that appear in the message
    - Any bracketed IPv6 addresses (e.g. ``[::1]``, ``[fe80::1%eth0]``)
    - The database username
    - The database port number
    - The database name
    """
    sanitized = error_msg.replace(connection_string, "<connection_string>")
    sanitized = re.sub(r"password=[^\s&]+", "password=***", sanitized)
    sanitized = re.sub(r"://[^@]+@", "://***:***@", sanitized)

    # Replace the known host (may be an IP already) before the generic IP pass.
    # Also replace the original (pre-DNS-resolution) hostname if it differs.
    if original_host and original_host != host:
        sanitized = sanitized.replace(original_host, "<host>")
    if host:
        sanitized = sanitized.replace(host, "<host>")

    # Replace any remaining IPv4 addresses (e.g. resolved IPs the driver logs)
    sanitized = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "<ip>", sanitized)

    # Replace bracketed IPv6 addresses (e.g. "[::1]", "[fe80::1%eth0]")
    sanitized = re.sub(r"\[[0-9a-fA-F:]+(?:%[^\]]+)?\]", "<ip>", sanitized)

    # Replace the database username (handles double-quoted, single-quoted,
    # and unquoted formats across PostgreSQL, MySQL, and MSSQL error messages).
    if username:
        sanitized = re.sub(
            r"""for user ["']?""" + re.escape(username) + r"""["']?""",
            "for user <user>",
            sanitized,
        )
        # Catch remaining bare occurrences in various quote styles:
        # - PostgreSQL: "FATAL:  role "myuser" does not exist"
        # - MySQL: "Access denied for user 'myuser'@'host'"
        # - MSSQL: "Login failed for user 'myuser'"
        sanitized = sanitized.replace(f'"{username}"', "<user>")
        sanitized = sanitized.replace(f"'{username}'", "<user>")

    # Replace the port number (handles "port 5432" and ":5432" formats)
    if port:
        port_str = re.escape(str(port))
        sanitized = re.sub(
            r"(?:port |:)" + port_str + r"(?![0-9])",
            lambda m: ("port " if m.group().startswith("p") else ":") + "<port>",
            sanitized,
        )

    # Replace the database name to avoid leaking internal infrastructure names.
    # Use word-boundary regex to prevent mangling when the database name is a
    # common substring (e.g. "test", "data", "on").
    if database:
        sanitized = re.sub(r"\b" + re.escape(database) + r"\b", "<database>", sanitized)

    return sanitized


def _extract_keyword_tokens(parsed: sqlparse.sql.Statement) -> list[str]:
    """Extract keyword tokens from a parsed SQL statement.

    Uses sqlparse token type classification to collect Keyword/DML/DDL/DCL
    tokens. String literals and identifiers have different token types, so
    they are naturally excluded from the result.
    """
    return [
        token.normalized.upper()
        for token in parsed.flatten()
        if token.ttype
        in (
            sqlparse.tokens.Keyword,
            sqlparse.tokens.Keyword.DML,
            sqlparse.tokens.Keyword.DDL,
            sqlparse.tokens.Keyword.DCL,
        )
    ]


def _has_disallowed_into(stmt: sqlparse.sql.Statement) -> bool:
    """Check if a statement contains a disallowed ``INTO`` clause.

    ``SELECT ... INTO @variable`` is a valid read-only MySQL syntax that stores
    a query result into a session-scoped user variable.  All other forms of
    ``INTO`` are data-modifying or file-writing and must be blocked:

    * ``SELECT ... INTO new_table``  (PostgreSQL / MSSQL – creates a table)
    * ``SELECT ... INTO OUTFILE``    (MySQL – writes to the filesystem)
    * ``SELECT ... INTO DUMPFILE``   (MySQL – writes to the filesystem)
    * ``INSERT INTO ...``            (already blocked by INSERT being in the
      disallowed set, but we reject INTO as well for defense-in-depth)

    Returns ``True`` if the statement contains a disallowed ``INTO``.
    """
    flat = list(stmt.flatten())
    for i, token in enumerate(flat):
        if not (
            token.ttype in (sqlparse.tokens.Keyword,)
            and token.normalized.upper() == "INTO"
        ):
            continue

        # Look at the first non-whitespace token after INTO.
        j = i + 1
        while j < len(flat) and flat[j].ttype is sqlparse.tokens.Text.Whitespace:
            j += 1

        if j >= len(flat):
            # INTO at the very end – malformed, block it.
            return True

        next_token = flat[j]
        # MySQL user variable: either a single Name starting with "@"
        # (e.g. ``@total``) or a bare ``@`` Operator token followed by a Name.
        if next_token.ttype is sqlparse.tokens.Name and next_token.value.startswith(
            "@"
        ):
            continue
        if next_token.ttype is sqlparse.tokens.Operator and next_token.value == "@":
            continue

        # Everything else (table name, OUTFILE, DUMPFILE, etc.) is disallowed.
        return True

    return False


def _validate_query_is_read_only(stmt: sqlparse.sql.Statement) -> str | None:
    """Validate that a parsed SQL statement is read-only (SELECT/WITH only).

    Accepts an already-parsed statement from ``_validate_single_statement``
    to avoid re-parsing. Checks:
    1. Statement type must be SELECT (sqlparse classifies WITH...SELECT as SELECT)
    2. No disallowed keywords (INSERT, UPDATE, DELETE, DROP, etc.)
    3. No disallowed INTO clauses (allows MySQL ``SELECT ... INTO @variable``)

    Returns an error message if the query is not read-only, None otherwise.
    """
    # sqlparse returns 'SELECT' for SELECT and WITH...SELECT queries
    if stmt.get_type() != "SELECT":
        return "Only SELECT queries are allowed."

    # Defense-in-depth: check parsed keyword tokens for disallowed keywords
    for kw in _extract_keyword_tokens(stmt):
        # Normalize multi-word tokens (e.g. "SET LOCAL" -> "SET")
        base_kw = kw.split()[0] if " " in kw else kw
        if base_kw in _DISALLOWED_KEYWORDS:
            return f"Disallowed SQL keyword: {kw}"

    # Contextual check for INTO: allow MySQL @variable syntax, block everything else
    if _has_disallowed_into(stmt):
        return "Disallowed SQL keyword: INTO"

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
        # NaN / Infinity are not valid JSON numbers; serialize as strings.
        if value.is_nan() or value.is_infinite():
            return str(value)
        # Use int for whole numbers; use str for fractional to preserve exact
        # precision (float would silently round high-precision analytics values).
        if value == value.to_integral_value():
            return int(value)
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, memoryview):
        return bytes(value).hex()
    if isinstance(value, bytes):
        return value.hex()
    return value


def _configure_session(
    conn: Any,
    dialect_name: str,
    timeout_ms: str,
    read_only: bool,
) -> None:
    """Set session-level timeout and read-only mode for the given dialect.

    Timeout limitations by database:

    * **PostgreSQL** – ``statement_timeout`` reliably cancels any running
      statement (SELECT or DML) after the configured duration.
    * **MySQL** – ``MAX_EXECUTION_TIME`` only applies to **read-only SELECT**
      statements.  DML (INSERT/UPDATE/DELETE) and DDL are *not* bounded by
      this hint; they rely on the server's ``wait_timeout`` /
      ``interactive_timeout`` instead.  There is no session-level setting in
      MySQL that reliably cancels long-running writes.
    * **MSSQL** – ``SET LOCK_TIMEOUT`` only limits how long the server waits
      to acquire a **lock**.  CPU-bound queries (e.g. large scans, hash
      joins) that do not block on locks will *not* be cancelled.  MSSQL has
      no session-level ``statement_timeout`` equivalent; the closest
      mechanism is Resource Governor (requires sysadmin configuration) or
      ``CONTEXT_INFO``-based external monitoring.

    Note: SQLite is not supported by this block.  The ``_configure_session``
    function is a no-op for unrecognised dialect names, so an SQLite engine
    would skip all SET commands silently.  The block's ``DatabaseType`` enum
    intentionally excludes SQLite.
    """
    if dialect_name == "postgresql":
        conn.execute(text("SET statement_timeout = " + timeout_ms))
        if read_only:
            conn.execute(text("SET default_transaction_read_only = ON"))
    elif dialect_name == "mysql":
        # NOTE: MAX_EXECUTION_TIME only applies to SELECT statements.
        # Write queries (INSERT/UPDATE/DELETE) are not bounded by this
        # setting; they rely on the database's wait_timeout instead.
        # See docstring above for full limitations.
        conn.execute(text("SET SESSION MAX_EXECUTION_TIME = " + timeout_ms))
        if read_only:
            conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
    elif dialect_name == "mssql":
        # MSSQL: SET LOCK_TIMEOUT limits lock-wait time (ms) only.
        # CPU-bound queries without lock contention are NOT cancelled.
        # See docstring above for full limitations.
        conn.execute(text("SET LOCK_TIMEOUT " + timeout_ms))
        # MSSQL lacks a session-level read-only mode like
        # PostgreSQL/MySQL.  Read-only enforcement is handled by
        # the SQL validation layer (_validate_query_is_read_only)
        # and the ROLLBACK in the finally block.


def _run_in_transaction(
    conn: Any,
    dialect_name: str,
    query: str,
    max_rows: int,
    read_only: bool,
) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    """Execute a query inside an explicit transaction, returning results.

    Returns ``(rows, columns, affected_rows, truncated)`` where *truncated*
    is ``True`` when ``fetchmany`` returned exactly ``max_rows`` rows,
    indicating that additional rows may exist in the result set.
    """
    # MSSQL uses T-SQL "BEGIN TRANSACTION"; others use "BEGIN".
    begin_stmt = "BEGIN TRANSACTION" if dialect_name == "mssql" else "BEGIN"
    conn.execute(text(begin_stmt))
    try:
        result = conn.execute(text(query))
        affected = result.rowcount if not result.returns_rows else -1
        columns = list(result.keys()) if result.returns_rows else []
        rows = result.fetchmany(max_rows) if result.returns_rows else []
        truncated = len(rows) == max_rows
        results = [
            {col: _serialize_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
    except Exception:
        try:
            conn.execute(text("ROLLBACK"))
        except Exception:
            pass
        raise
    else:
        conn.execute(text("ROLLBACK" if read_only else "COMMIT"))
    return results, columns, affected, truncated


def _execute_query(
    connection_url: URL | str,
    query: str,
    timeout: int,
    max_rows: int,
    read_only: bool = True,
    database_type: DatabaseType = DatabaseType.POSTGRES,
) -> tuple[list[dict[str, Any]], list[str], int, bool]:
    """Execute a SQL query and return (rows, columns, affected_rows, truncated).

    Uses SQLAlchemy to connect to any supported database.
    For SELECT queries, rows are limited to ``max_rows`` via DBAPI fetchmany.
    ``truncated`` is ``True`` when the result set was capped by ``max_rows``.
    For write queries, affected_rows contains the rowcount from the driver.
    When ``read_only`` is True, the database session is set to read-only
    mode and the transaction is always rolled back.
    """
    # Determine driver-specific connection timeout argument.
    # pymssql uses "login_timeout", while PostgreSQL/MySQL use "connect_timeout".
    timeout_key = (
        "login_timeout" if database_type == DatabaseType.MSSQL else "connect_timeout"
    )
    engine = create_engine(
        connection_url, connect_args={timeout_key: _CONNECT_TIMEOUT_SECONDS}
    )
    try:
        with engine.connect() as conn:
            # Use AUTOCOMMIT so SET commands take effect immediately.
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")

            # Compute timeout in milliseconds. The value is Pydantic-validated
            # (ge=1, le=120), but we use int() as defense-in-depth.
            # NOTE: SET commands do not support bind parameters in most
            # databases, so we use str(int(...)) for safe interpolation.
            timeout_ms = str(int(timeout * 1000))

            _configure_session(conn, engine.dialect.name, timeout_ms, read_only)
            return _run_in_transaction(
                conn, engine.dialect.name, query, max_rows, read_only
            )
    finally:
        engine.dispose()
