"""The leaked-transaction guard, on the real database it guards.

conftest.py runs it after every test; these make sure it still sees a leak,
and still lets a transaction that is merely in flight finish.
"""

import asyncio

import pytest
from prisma import Prisma
from prisma.errors import PrismaError

from backend.data import db
from backend.data.db_transaction_guard import (
    end_leaked_transactions,
    find_leaked_transactions,
    release_leaked_transactions,
)
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_transaction_left_open_is_found_and_ended(server: SpinTestServer):
    await db.prisma.execute_raw("BEGIN")
    # The guard looks after the test has finished, never in the same instant.
    await asyncio.sleep(0.01)

    leaks = await find_leaked_transactions(db.prisma)
    assert leaks, "a BEGIN left on a pooled connection went unnoticed"

    await release_leaked_transactions(db.prisma, leaks)
    assert await find_leaked_transactions(db.prisma) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_uncommitted_write_is_found_however_soon_the_guard_looks(
    server: SpinTestServer,
):
    await db.prisma.execute_raw("BEGIN")
    await db.prisma.execute_raw("CREATE TEMP TABLE leaked_write (id int)")

    leaks = await find_leaked_transactions(db.prisma)
    assert leaks, "an uncommitted write went unnoticed"

    await release_leaked_transactions(db.prisma, leaks)
    assert await find_leaked_transactions(db.prisma) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_aborted_transaction_left_open_is_found_and_ended(
    server: SpinTestServer,
):
    """What the next test would otherwise inherit: every statement refused
    with 25P02, the guard's own query included."""
    await db.prisma.execute_raw("BEGIN")
    with pytest.raises(PrismaError):
        await db.prisma.execute_raw("SELECT 1 / 0")

    leaks = await find_leaked_transactions(db.prisma)
    assert leaks, "an aborted transaction went unnoticed"

    await release_leaked_transactions(db.prisma, leaks)
    assert await find_leaked_transactions(db.prisma) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_leak_on_another_connection_of_this_session_is_terminated(
    server: SpinTestServer,
):
    """The service processes' connections carry the session's name too."""
    other = Prisma(datasource={"url": db.DATABASE_URL})
    await other.connect()
    try:
        await other.execute_raw("BEGIN")

        leaks = await find_leaked_transactions(db.prisma)
        assert leaks, "an open transaction on another session connection went unnoticed"

        assert await release_leaked_transactions(db.prisma, leaks) == []
        assert await find_leaked_transactions(db.prisma) == []
    finally:
        await other.disconnect()


@pytest.mark.asyncio(loop_scope="session")
async def test_an_aborted_guard_connection_does_not_hide_another_leak(
    server: SpinTestServer,
):
    """The aborted connection refuses the query that finds leaks, so a leak on
    another connection stays hidden until it is gone. The next test must not
    inherit that one, or be blamed for it."""
    other = Prisma(datasource={"url": db.DATABASE_URL})
    await other.connect()
    try:
        await other.execute_raw("BEGIN")
        await other.execute_raw("CREATE TEMP TABLE hidden_leak (id int)")
        await db.prisma.execute_raw("BEGIN")
        with pytest.raises(PrismaError):
            await db.prisma.execute_raw("SELECT 1 / 0")

        report = await end_leaked_transactions(db.prisma)

        assert len(report.leaks) == 2, report.leaks
        assert report.survivors == []
        assert await find_leaked_transactions(db.prisma) == []
    finally:
        await other.disconnect()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_transaction_held_by_another_client_is_left_alone(
    server: SpinTestServer,
):
    """Someone else's session on the same database is not this test's leak."""
    url = db.add_param(db.DATABASE_URL, "application_name", "someone-else")
    outsider = Prisma(datasource={"url": url})
    await outsider.connect()
    try:
        await outsider.execute_raw("BEGIN")

        assert await find_leaked_transactions(db.prisma) == []
    finally:
        await outsider.execute_raw("ROLLBACK")
        await outsider.disconnect()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_transaction_still_in_flight_is_not_reported(server: SpinTestServer):
    inside = asyncio.Event()

    async def hold_briefly():
        async with db.transaction() as tx:
            await tx.execute_raw("SELECT 1")
            inside.set()
            await asyncio.sleep(0.1)

    holder = asyncio.create_task(hold_briefly())
    await asyncio.wait_for(inside.wait(), timeout=5)

    leaks = await find_leaked_transactions(db.prisma)
    await holder

    assert leaks == []
