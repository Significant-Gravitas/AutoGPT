"""A query the database never got to run raises ``DatabaseConnectionError``,
which the database manager's client retries; a query the database rejected
raises its own class, answered at once, as before. Only the query engine's
HTTP answers are recorded; the client, the database manager and its RPC
client are the real ones."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
import pytest
from prisma.errors import PrismaError, RawQueryError, UniqueViolationError

from backend.data import db
from backend.data.db_errors import DatabaseConnectionError
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient
from backend.util import retry as retry_module
from backend.util import service as service_module
from backend.util.service import get_service_client
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


def _raw_query_failed(sqlstate: str, reason: str) -> dict:
    return _engine_error(
        "P2010",
        f"Raw query failed. Code: `{sqlstate}`. Message: `ERROR: {reason}`",
        {"code": sqlstate, "message": f"ERROR: {reason}"},
    )


# CLOSED and the raw statement timeout are as the query engine (5.17) sent them
# against Postgres 15, for a terminated backend and a pg_sleep past the timeout.
_SERVER = {"database_host": "db.internal", "database_port": 5432}
# AUTOGPT-SERVER-AFC: a statement timeout outside a raw query.
_STATEMENT_TIMED_OUT = (
    "Error occurred during query execution:\nConnectorError(ConnectorError { "
    'user_facing_error: None, kind: QueryError(PostgresError { code: "57014", '
    'message: "canceling statement due to statement timeout", severity: '
    '"ERROR", detail: None, column: None, hint: None }), transient: false })'
)
CLOSED = _engine_error("P1017", "Server has closed the connection.")
DUPLICATE = _engine_error("P2002", "Unique constraint failed on the fields: (`id`)")
NO_ROW = {"data": {"result": None}}


@pytest.mark.parametrize(
    "answer, raised",
    [
        pytest.param(CLOSED, DatabaseConnectionError, id="P1017-null-meta"),
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
            _engine_error("P1008", "Operations timed out after `10s`", {"time": "10s"}),
            DatabaseConnectionError,
            id="P1008",
        ),
        pytest.param(
            _engine_error(None, _STATEMENT_TIMED_OUT),
            DatabaseConnectionError,
            id="statement-timeout",
        ),
        pytest.param(
            _raw_query_failed("57014", "canceling statement due to statement timeout"),
            DatabaseConnectionError,
            id="statement-timeout-raw",
        ),
        pytest.param(
            _raw_query_failed("57014", "canceling statement due to user request"),
            RawQueryError,
            id="cancel-on-request-raw",
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


async def test_the_database_manager_client_retries_a_query_the_database_dropped(
    monkeypatch, no_retry_wait
):
    async with _database_manager_answering(monkeypatch, CLOSED, NO_ROW) as (
        client,
        statuses,
    ):
        active = await client.is_auth_session_active("session-1", "user-1")

    assert active is False
    assert statuses == [503, 200]


async def test_the_database_manager_client_answers_a_rejected_query_at_once(
    monkeypatch, no_retry_wait
):
    async with _database_manager_answering(monkeypatch, DUPLICATE, NO_ROW) as (
        client,
        statuses,
    ):
        with pytest.raises(UniqueViolationError):
            await client.is_auth_session_active("session-1", "user-1")

    assert statuses == [400]


@pytest.fixture
def no_retry_wait(monkeypatch):
    async def skip(seconds: float) -> None:
        pass

    monkeypatch.setattr(retry_module, "_interruptible_async_sleep", skip)


@asynccontextmanager
async def _engine_answering(*answers: dict) -> AsyncIterator[Any]:
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
    monkeypatch: pytest.MonkeyPatch, *answers: dict
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
