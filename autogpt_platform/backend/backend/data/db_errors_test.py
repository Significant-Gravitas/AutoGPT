"""A query that failed because the connection was lost, refused or timed out
raises ``DatabaseConnectionError``, which the database manager's client
retries; any other failed query, a statement timeout included, raises its own
class, answered at once, as before. Only the query engine's HTTP answers are
recorded; the client, the database manager and its RPC client are the real
ones."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import pytest
from prisma.errors import DataError, PrismaError, RawQueryError, UniqueViolationError

from backend.data import db
from backend.data.db_errors import DatabaseConnectionError
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient
from backend.util import retry as retry_module
from backend.util import service as service_module
from backend.util.service import HTTPClientError, get_service_client
from backend.util.service_auth import internal_auth_headers


def _engine_error(code: str | None, message: str, meta: object = None) -> dict:
    """An engine answer with one error, as the engine writes it: a known error
    has a code and a meta, null for an error with no fields; an unknown error
    has neither."""
    user_facing: dict[str, Any] = {"is_panic": False, "message": message}
    if code is None:
        user_facing["backtrace"] = None
    else:
        user_facing |= {"meta": meta, "error_code": code}
    return {"errors": [{"error": message, "user_facing_error": user_facing}]}


def _raw_query_failed(sqlstate: str, message: str) -> dict:
    return _engine_error(
        "P2010",
        f"Raw query failed. Code: `{sqlstate}`. Message: `{message}`",
        {"code": sqlstate, "message": message},
    )


def _query_failed(code: str, message: str, severity: str) -> dict:
    """A Postgres error outside a raw query, which the engine sends with no
    code, quoting it in full as Rust prints it."""
    return _engine_error(
        None,
        "Error occurred during query execution:\nConnectorError(ConnectorError { "
        "user_facing_error: None, kind: QueryError(PostgresError { "
        f"code: {_quoted(code)}, message: {_quoted(message)}, "
        f"severity: {_quoted(severity)}, detail: None, column: None, hint: None }}), "
        "transient: false })",
    )


def _quoted(text: str) -> str:
    """*text* quoted as Rust prints a string, its own quotes escaped."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# As the query engine (5.17) sent them against Postgres 15: CLOSED for a pooled
# connection whose backend was terminated while idle, the terminated-backend
# answers for a backend terminated mid-query (a raw query and a model query,
# from a server writing English and one writing German), the raw statement
# timeout for a pg_sleep past it, and the two answers quoting a query value
# written to pass for a terminated backend. The other statement timeout is
# AUTOGPT-SERVER-AFC's message.
_SERVER = {"database_host": "db.internal", "database_port": 5432}
_TERMINATED = "terminating connection due to administrator command"
_TERMINATED_IN_GERMAN = (
    "Verbindung wird abgebrochen aufgrund von Anweisung des Administrators"
)
_TIMED_OUT = "canceling statement due to statement timeout"
_LOOKS_TERMINATED = f'PostgresError {{ code: "57P01", message: "{_TERMINATED}"'
CLOSED = _engine_error("P1017", "Server has closed the connection.")
# The engine can send its answer as a JSON string, which prisma decodes.
CLOSED_AS_A_STRING = json.dumps(CLOSED)
TERMINATED_RAW = _raw_query_failed("57P01", f"FATAL: {_TERMINATED}")
TERMINATED_IN_GERMAN = _query_failed("57P01", _TERMINATED_IN_GERMAN, "FATAL")
STATEMENT_TIMED_OUT = _query_failed("57014", _TIMED_OUT, "ERROR")
# SELECT '57P01 terminating connection due to administrator command'::uuid
QUOTES_TERMINATED_RAW = _raw_query_failed(
    "22P02", f'ERROR: invalid input syntax for type uuid: "57P01 {_TERMINATED}"'
)
# The lookalike written to a column whose check reads it as a time zone, as
# User.timezone's does.
QUOTES_TERMINATED = _query_failed(
    "22023", f'time zone "{_LOOKS_TERMINATED}" not recognized', "ERROR"
)
DUPLICATE = _engine_error("P2002", "Unique constraint failed on the fields: (`id`)")
NO_ROW = {"data": {"result": None}}


@pytest.mark.parametrize(
    "answer, raised",
    [
        pytest.param(CLOSED, DatabaseConnectionError, id="P1017-null-meta"),
        pytest.param(
            CLOSED_AS_A_STRING, DatabaseConnectionError, id="P1017-string-encoded"
        ),
        pytest.param(
            _engine_error(
                "P1001", "Can't reach database server at `db.internal:5432`", _SERVER
            ),
            DatabaseConnectionError,
            id="P1001",
        ),
        pytest.param(
            _engine_error(
                "P1002",
                "The database server at `db.internal:5432` was reached but timed out.",
                {**_SERVER, "context": "Connecting"},
            ),
            DatabaseConnectionError,
            id="P1002",
        ),
        pytest.param(
            _query_failed("57P01", _TERMINATED, "FATAL"),
            DatabaseConnectionError,
            id="backend-terminated",
        ),
        pytest.param(
            TERMINATED_RAW, DatabaseConnectionError, id="backend-terminated-raw"
        ),
        # Told by the SQLSTATE alone, so a server writing another language
        # (lc_messages) is retried too.
        pytest.param(
            TERMINATED_IN_GERMAN,
            DatabaseConnectionError,
            id="backend-terminated-german",
        ),
        pytest.param(
            _raw_query_failed("57P01", f"FATAL: {_TERMINATED_IN_GERMAN}"),
            DatabaseConnectionError,
            id="backend-terminated-raw-german",
        ),
        pytest.param(
            _raw_query_failed(
                "22P02", 'ERROR: invalid input syntax for type uuid: "57P01"'
            ),
            RawQueryError,
            id="quotes-57P01-raw",
        ),
        pytest.param(QUOTES_TERMINATED_RAW, RawQueryError, id="quotes-terminated-raw"),
        pytest.param(QUOTES_TERMINATED, DataError, id="quotes-terminated"),
        # A timeout is answered as before, not retried: it mostly means an
        # overloaded database, which re-running the statement would load more.
        pytest.param(
            _engine_error("P1008", "Operations timed out after `10s`", {"time": "10s"}),
            DataError,
            id="P1008",
        ),
        pytest.param(STATEMENT_TIMED_OUT, DataError, id="statement-timeout"),
        pytest.param(
            _raw_query_failed("57014", f"ERROR: {_TIMED_OUT}"),
            RawQueryError,
            id="statement-timeout-raw",
        ),
        pytest.param(DUPLICATE, UniqueViolationError, id="P2002-null-meta"),
    ],
)
async def test_a_query_raises_what_the_engine_answered(answer, raised):
    async with _engine_answering(answer) as engine:
        client = _client_on(engine)
        with pytest.raises(PrismaError) as caught:
            await client.query_raw("SELECT 1")

    assert type(caught.value) is raised


async def test_an_answer_without_errors_is_returned_as_before():
    rows = {"columns": ["health_check"], "types": ["int"], "rows": [[1]]}
    async with _engine_answering({"data": {"result": rows}}) as engine:
        result = await _client_on(engine).query_raw("SELECT 1 as health_check")

    assert result == [{"health_check": 1}]


@pytest.mark.parametrize(
    "dropped",
    [
        pytest.param([CLOSED], id="P1017"),
        pytest.param([CLOSED_AS_A_STRING], id="P1017-string-encoded"),
        # As seen live: the query after a backend terminated mid-query gets its
        # dead pooled connection back (P1017), and the next one a new connection.
        pytest.param([TERMINATED_RAW, CLOSED], id="backend-terminated-raw"),
        pytest.param([TERMINATED_IN_GERMAN, CLOSED], id="backend-terminated-german"),
    ],
)
async def test_the_database_manager_client_retries_a_query_the_database_dropped(
    monkeypatch, no_retry_wait, dropped
):
    async with _database_manager_answering(monkeypatch, *dropped, NO_ROW) as (
        client,
        statuses,
    ):
        active = await client.is_auth_session_active("session-1", "user-1")

    assert active is False
    assert statuses == [503] * len(dropped) + [200]


@pytest.mark.parametrize(
    "answer, raised",
    [
        pytest.param(DUPLICATE, UniqueViolationError, id="P2002"),
        pytest.param(STATEMENT_TIMED_OUT, DataError, id="statement-timeout"),
        # The client rebuilds only the classes it knows, and RawQueryError
        # isn't one, so it raises the 400 as itself.
        pytest.param(
            QUOTES_TERMINATED_RAW, HTTPClientError, id="quotes-terminated-raw"
        ),
        pytest.param(QUOTES_TERMINATED, DataError, id="quotes-terminated"),
    ],
)
async def test_the_database_manager_client_answers_other_query_errors_at_once(
    monkeypatch, no_retry_wait, answer, raised
):
    async with _database_manager_answering(monkeypatch, answer, NO_ROW) as (
        client,
        statuses,
    ):
        with pytest.raises(raised):
            await client.is_auth_session_active("session-1", "user-1")

    assert statuses == [400]


@pytest.fixture
def no_retry_wait(monkeypatch):
    async def skip(seconds: float) -> None:
        pass

    monkeypatch.setattr(retry_module, "_interruptible_async_sleep", skip)


@asynccontextmanager
async def _engine_answering(*answers: object) -> AsyncIterator[Any]:
    """The query engine of the backend's client, answering each request with
    the next of *answers* instead of a query engine process."""
    pending = iter(answers)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=next(pending)))
    owner = type(db.prisma)(auto_register=False, http={"transport": transport})
    engine = owner._create_engine()
    # Spawning the engine process sets its URL; the transport answers instead.
    engine.url = "http://query-engine"
    try:
        yield engine
    finally:
        await engine.aclose()


def _client_on(engine: Any) -> Any:
    client = type(db.prisma)(auto_register=False)
    client._internal_engine = engine
    return client


@asynccontextmanager
async def _database_manager_answering(
    monkeypatch: pytest.MonkeyPatch, *answers: object
) -> AsyncIterator[tuple[DatabaseManagerAsyncClient, list[int]]]:
    """The database manager's retrying client, served in process by the
    database manager's own app, whose queries get *answers* in turn. Yields
    the client and the status of each RPC answer, in order."""
    monkeypatch.setattr(service_module, "instrument_fastapi", lambda *_a, **_k: None)
    statuses: list[int] = []

    async def record(response: httpx.Response) -> None:
        statuses.append(response.status_code)

    client = get_service_client(DatabaseManagerAsyncClient, request_retry=True)
    inner = cast(Any, client)
    inner._async_clients[asyncio.get_running_loop()] = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=DatabaseManager().build_fastapi_app(), raise_app_exceptions=False
        ),
        base_url=inner.base_url,
        headers=internal_auth_headers(),
        event_hooks={"response": [record]},
    )
    # The manager's queries go through the registered client. It is put back
    # here, not by monkeypatch, whose undo runs after the session's teardown
    # check has queried it.
    previous = db.prisma._internal_engine
    async with _engine_answering(*answers) as engine:
        db.prisma._internal_engine = engine
        try:
            yield client, statuses
        finally:
            db.prisma._internal_engine = previous
            await inner.aclose()
