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
# pg_terminate_backend, a failover or a shutdown does. A raw query reports it
# as P2010 and any other as an error with no code, and both quote it.
BACKEND_TERMINATED = ("57P01", "terminating connection due to administrator command")


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
    message = user_facing.get("message") or ""
    return user_facing.get("error_code") in CONNECTION_ERROR_CODES or all(
        part in message for part in BACKEND_TERMINATED
    )
