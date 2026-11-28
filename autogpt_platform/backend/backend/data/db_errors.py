"""The database errors a retry can cure, told apart where the query engine
answers, so every query raises them, in a transaction or not.

Prisma raises the same ``DataError`` for a query the database rejected and for
one that failed because the connection was lost, refused or timed out, and the
database manager answers a ``DataError`` with a 400, which its RPC clients
never retry. ``RetryablePrisma`` raises ``DatabaseConnectionError``, still a
``DataError``, for the second kind; the database manager answers it with a 503,
which they retry. Nothing else is needed to recover: the query engine drops a
dead pooled connection when a query on it fails with P1017, and the next retry
opens a new one. A backend terminated mid-query takes two retries, because its
connection's next query gets that P1017.

A statement timeout (SQLSTATE 57014) and an operation timeout (P1008) are left
as they were, on purpose: they mostly mean the database is overloaded, and the
database manager's client, retrying about a hundred times up to 30 s apart,
would keep re-running the statement on it.
"""

import json
import re
from pathlib import Path
from typing import Any

import httpx
from prisma import Prisma
from prisma.engine import AsyncAbstractEngine, AsyncQueryEngine
from prisma.errors import DataError
from prisma.http_abstract import AbstractResponse
from prisma.utils import is_dict

# The server can't be reached (P1001), was reached but timed out (P1002), the
# server closed the connection (P1017).
CONNECTION_ERROR_CODES = frozenset({"P1001", "P1002", "P1017"})
# Postgres terminated the backend running the query (SQLSTATE 57P01), as
# pg_terminate_backend, a failover or a shutdown does. Only the SQLSTATE tells
# it: the message is in the server's language (lc_messages), and an error's
# message can quote the query's own values.
BACKEND_TERMINATED = "57P01"
# A raw query reports a Postgres error as P2010, with the SQLSTATE in its meta.
# Any other query reports it as an error with no code, whose message embeds the
# error as Rust prints it. Every string in it is quoted, with its own quotes
# escaped (\"), so a value the error quotes can't fake the code field:
#   ...kind: QueryError(PostgresError { code: "57P01", message: "...", ... })...
_POSTGRES_ERROR = re.compile(r'\bPostgresError \{ code: "(?P<sqlstate>[0-9A-Z]{5})"')


class DatabaseConnectionError(DataError):
    """The connection to the database was lost, refused or timed out, rather
    than the database rejecting the query, so the same query can succeed when
    it is retried."""


class RetryablePrisma(Prisma):
    """A Prisma client whose queries raise ``DatabaseConnectionError`` for the
    errors a retry can cure."""

    def _create_engine(self, dml_path: Path | None = None) -> AsyncAbstractEngine:
        return _QueryEngine(
            dml_path=dml_path or self._packaged_schema_path,
            log_queries=self._log_queries,
            http_config=self._http_config,
        )


class _QueryEngine(AsyncQueryEngine):
    def _process_response_data(
        self, *, data: object, response: AbstractResponse[httpx.Response]
    ) -> Any:
        # The engine can send its answer as a JSON string, which the base
        # method decodes before reading it; decoded here first, such an answer
        # gets the same treatment.
        if isinstance(data, str):
            data = json.loads(data)
        errors: Any = data.get("errors") if is_dict(data) else None
        for error in errors or ():
            user_facing = error.get("user_facing_error") or {}
            # The engine sends a null meta for an error with no fields, P1017's
            # among them, and prisma 0.15.0 then fails on meta.get() with an
            # AttributeError instead of raising the error's own class.
            if user_facing.get("meta", {}) is None:
                user_facing["meta"] = {}
            if _cured_by_a_retry(user_facing):
                raise DatabaseConnectionError(error)
        return super()._process_response_data(data=data, response=response)


def _cured_by_a_retry(user_facing: dict[str, Any]) -> bool:
    return (
        user_facing.get("error_code") in CONNECTION_ERROR_CODES
        or _sqlstate(user_facing) == BACKEND_TERMINATED
    )


def _sqlstate(user_facing: dict[str, Any]) -> str | None:
    """The SQLSTATE of the Postgres error that failed the query, if the engine
    reports one."""
    code = user_facing.get("error_code")
    if code == "P2010":
        return user_facing.get("meta", {}).get("code")
    if code is None:
        found = _POSTGRES_ERROR.search(user_facing.get("message") or "")
        return found["sqlstate"] if found else None
    return None
