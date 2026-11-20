"""The database errors a retry can cure, told apart where the query engine
answers, so every query raises them, in a transaction or not.

Prisma raises the same ``DataError`` for a query the database rejected and for
one that never ran because the connection was lost, refused or timed out, and
the database manager answers a ``DataError`` with a 400, which its RPC clients
never retry. ``RetryablePrisma`` raises ``DatabaseConnectionError``, still a
``DataError``, for the second kind; the database manager answers it with a 503,
which they retry. Nothing else is needed to recover: the query engine drops a
dead pooled connection, and the retried query opens a new one.
"""

from pathlib import Path
from typing import Any

import httpx
from prisma import Prisma
from prisma.engine import AsyncAbstractEngine, AsyncQueryEngine
from prisma.errors import DataError
from prisma.http_abstract import AbstractResponse
from prisma.utils import is_dict

# The server can't be reached (P1001), was reached but timed out (P1002), an
# operation timed out (P1008), the server closed the connection (P1017).
CONNECTION_ERROR_CODES = frozenset({"P1001", "P1002", "P1008", "P1017"})
# Postgres cancelled the statement at statement_timeout (SQLSTATE 57014). A raw
# query reports it as P2010 and any other as an error with no code, and both
# quote it. A cancel someone asked for has the same SQLSTATE and another reason.
STATEMENT_TIMEOUT = ("57014", "canceling statement due to statement timeout")


class DatabaseConnectionError(DataError):
    """The database dropped, refused or timed out the query rather than
    rejecting it, so the same query can succeed when it is retried."""


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
        part in message for part in STATEMENT_TIMEOUT
    )
