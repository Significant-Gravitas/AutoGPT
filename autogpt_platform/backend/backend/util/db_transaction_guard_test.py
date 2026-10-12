"""The leaked-transaction guard, on the real database it guards.

conftest.py runs it after every test; these make sure it still sees a leak,
and still lets a transaction that is merely in flight finish.
"""

import asyncio

import pytest
from prisma.errors import PrismaError

from backend.data import db
from backend.util.db_transaction_guard import (
    find_leaked_transactions,
    release_leaked_transactions,
)
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_transaction_left_open_is_found_and_ended(server: SpinTestServer):
    await db.prisma.execute_raw("BEGIN")
    # The guard looks after the test has finished, never in the same instant.
    await asyncio.sleep(0.01)

    leaks = await find_leaked_transactions()
    assert leaks, "a BEGIN left on a pooled connection went unnoticed"

    await release_leaked_transactions(leaks)
    assert await find_leaked_transactions() == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_uncommitted_write_is_found_however_soon_the_guard_looks(
    server: SpinTestServer,
):
    await db.prisma.execute_raw("BEGIN")
    await db.prisma.execute_raw("CREATE TEMP TABLE leaked_write (id int)")

    leaks = await find_leaked_transactions()
    assert leaks, "an uncommitted write went unnoticed"

    await release_leaked_transactions(leaks)
    assert await find_leaked_transactions() == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_aborted_transaction_left_open_is_found_and_ended(
    server: SpinTestServer,
):
    """What the next test would otherwise inherit: every statement refused
    with 25P02, the guard's own query included."""
    await db.prisma.execute_raw("BEGIN")
    with pytest.raises(PrismaError):
        await db.prisma.execute_raw("SELECT 1 / 0")

    leaks = await find_leaked_transactions()
    assert leaks, "an aborted transaction went unnoticed"

    await release_leaked_transactions(leaks)
    assert await find_leaked_transactions() == []


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

    leaks = await find_leaked_transactions()
    await holder

    assert leaks == []
