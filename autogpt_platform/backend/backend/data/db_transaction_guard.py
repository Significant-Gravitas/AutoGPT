"""Find database transactions a test left open, so the test can be failed.

A connection that goes back to the pool inside an open transaction runs every
later statement inside it: their writes are never committed, and the first one
that fails aborts the rest (Postgres ``25P02``), in whatever test comes next.
"""

import asyncio
from datetime import datetime, timezone

from prisma import Prisma
from prisma.errors import PrismaError
from pydantic import BaseModel

from backend.data.db import DISCONNECT_TIMEOUT

_UNKNOWN_START = datetime.min.replace(tzinfo=timezone.utc)

# The guard's own query shows its connection as active, not idle. It was
# already inside a transaction if that transaction has written something (only
# writes take a transaction id; this SELECT does not), or if it began before
# this statement: a clean query sees statement_timestamp() within a fraction of
# a millisecond of now().
_OWN_CONNECTION_GAP_SECONDS = 0.002
# A transaction still open this long after the first look was left open, not
# caught mid-flight.
_CONFIRM_AFTER_SECONDS = 0.5

_OPEN_TRANSACTIONS_SQL = """
SELECT a.pid,
       a.pid = pg_backend_pid() AS is_own,
       a.state,
       a.xact_start,
       a.backend_xid IS NOT NULL AS has_written,
       EXTRACT(EPOCH FROM (statement_timestamp() - now())) AS own_gap,
       left(a.query, 300) AS last_query
FROM pg_stat_activity a
WHERE a.datname = current_database() AND a.xact_start IS NOT NULL
"""


class LeakedTransaction(BaseModel):
    pid: int
    xact_start: datetime
    state: str
    last_query: str


async def find_leaked_transactions(client: Prisma) -> list[LeakedTransaction]:
    """Transactions on the client's database still open after two looks."""
    first = await _open_transactions(client)
    if not first:
        return []
    await asyncio.sleep(_CONFIRM_AFTER_SECONDS)
    second = {
        (leak.pid, leak.xact_start): leak for leak in await _open_transactions(client)
    }
    return [
        second[(leak.pid, leak.xact_start)]
        for leak in first
        if (leak.pid, leak.xact_start) in second
    ]


async def release_leaked_transactions(
    client: Prisma, leaks: list[LeakedTransaction]
) -> None:
    """End the leaked transactions and start the client on a fresh pool.

    Postgres rolls back a transaction whose connection closes: other backends
    are terminated, and reconnecting closes this client's own.
    """
    for leak in leaks:
        try:
            await client.query_raw(
                "SELECT pg_terminate_backend($1::int) AS terminated"
                " WHERE $1::int <> pg_backend_pid()",
                leak.pid,
            )
        except PrismaError:
            pass  # This connection is the aborted one; the reconnect ends it.
    await client.disconnect(DISCONNECT_TIMEOUT)
    await client.connect()


async def _open_transactions(client: Prisma) -> list[LeakedTransaction]:
    try:
        rows = await client.query_raw(_OPEN_TRANSACTIONS_SQL)
    except PrismaError as error:
        # Only a connection already inside an aborted transaction refuses a
        # plain SELECT like this one.
        if not _is_aborted_transaction(error):
            raise
        return [
            LeakedTransaction(
                pid=0,
                xact_start=_UNKNOWN_START,
                state="idle in transaction (aborted), on the guard's connection",
                last_query=str(error)[:300],
            )
        ]
    return [
        LeakedTransaction(
            pid=row["pid"],
            xact_start=row["xact_start"],
            state=(
                "idle in transaction, on the guard's connection"
                if row["is_own"]
                else row["state"] or ""
            ),
            last_query="" if row["is_own"] else row["last_query"] or "",
        )
        for row in rows
        if _is_left_open(row)
    ]


def _is_aborted_transaction(error: PrismaError) -> bool:
    # A raw query reports the message, a model query the SQLSTATE code.
    message = str(error)
    return "25P02" in message or "current transaction is aborted" in message


def _is_left_open(row: dict) -> bool:
    if row["is_own"]:
        return (
            bool(row["has_written"])
            or float(row["own_gap"]) > _OWN_CONNECTION_GAP_SECONDS
        )
    return str(row["state"]).startswith("idle in transaction")
