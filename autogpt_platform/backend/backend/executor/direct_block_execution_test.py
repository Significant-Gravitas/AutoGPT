import asyncio
import threading

import pytest
import pytest_mock
from prisma import Prisma

from backend.data import db, db_accessors
from backend.executor.direct_block_execution import (
    DIRECT_BLOCK_EXECUTION_WORKERS,
    run_block_off_loop,
)

RPC_CLIENT = object()


@pytest.fixture
def prisma_connected_here(mocker: pytest_mock.MockFixture) -> None:
    """The server's Prisma client, connected on this test's thread."""
    mocker.patch.object(Prisma, "is_connected", return_value=True)
    mocker.patch.object(db.prisma, "_connected_thread", threading.get_ident())
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client",
        return_value=RPC_CLIENT,
    )


@pytest.mark.asyncio
async def test_collects_every_output_by_name():
    async def outputs():
        yield "a", 1
        yield "b", 2
        yield "a", 3

    assert await run_block_off_loop(outputs()) == {"a": [1, 3], "b": [2]}


@pytest.mark.asyncio
async def test_block_errors_reach_the_caller():
    async def outputs():
        raise ValueError("bad input")
        yield "never", None

    with pytest.raises(ValueError, match="bad input"):
        await run_block_off_loop(outputs())


@pytest.mark.asyncio
@pytest.mark.usefixtures("prisma_connected_here")
async def test_server_thread_keeps_the_direct_prisma_path():
    assert db.is_connected()
    assert db.prisma.is_connected()
    assert db_accessors.chat_db() is not RPC_CLIENT


@pytest.mark.asyncio
@pytest.mark.usefixtures("prisma_connected_here")
async def test_block_on_a_worker_loop_takes_the_database_manager_path():
    seen: dict[str, object] = {}

    async def outputs():
        seen["prisma_connected"] = db.prisma.is_connected()
        seen["chat_db"] = db_accessors.chat_db()
        yield "done", True

    assert await run_block_off_loop(outputs()) == {"done": [True]}
    assert seen == {"prisma_connected": False, "chat_db": RPC_CLIENT}
    assert db_accessors.chat_db() is not RPC_CLIENT


@pytest.mark.asyncio
async def test_worker_keeps_one_event_loop_across_runs():
    async def outputs():
        yield "loop", asyncio.get_running_loop()

    loops = {id((await run_block_off_loop(outputs()))["loop"][0]) for _ in range(20)}
    # One loop per worker thread at most, never one per run.
    assert len(loops) <= DIRECT_BLOCK_EXECUTION_WORKERS
