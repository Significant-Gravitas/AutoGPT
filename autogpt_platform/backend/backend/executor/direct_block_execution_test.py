import asyncio
import threading
import time

import pytest
import pytest_mock
from prisma import Prisma

from backend.data import db, db_accessors
from backend.executor import direct_block_execution
from backend.executor.direct_block_execution import (
    DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER,
    DIRECT_BLOCK_EXECUTION_WORKERS,
    DirectBlockExecutionLimitError,
    DirectBlockExecutionTimeoutError,
    claim_direct_block_execution_slot,
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


@pytest.mark.asyncio
async def test_waiting_runs_share_the_worker_loops():
    runs = 3 * DIRECT_BLOCK_EXECUTION_WORKERS
    started = 0
    all_started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()

    async def outputs():
        nonlocal started
        with lock:
            started += 1
            if started == runs:
                all_started.set()
        while not release.is_set():
            await asyncio.sleep(0.01)
        yield "done", True

    pending = [asyncio.create_task(run_block_off_loop(outputs())) for _ in range(runs)]
    try:
        # More runs than worker threads are in flight at once: none holds a
        # thread while it waits.
        assert await asyncio.to_thread(all_started.wait, 10)
    finally:
        release.set()
    results = await asyncio.wait_for(asyncio.gather(*pending), timeout=10)
    assert results == [{"done": [True]}] * runs


@pytest.mark.asyncio
async def test_time_budget_starts_when_the_run_starts(
    mocker: pytest_mock.MockFixture,
):
    budget = 3.0
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS", budget
    )
    worker = direct_block_execution._WorkerLoop(index=99)
    mocker.patch.object(
        direct_block_execution, "_least_busy_worker", return_value=worker
    )
    loop_held = threading.Event()

    async def hold_the_loop():
        loop_held.set()
        # Synchronous wait: the run below cannot start until it ends.
        time.sleep(budget * 0.8)

    asyncio.run_coroutine_threadsafe(hold_the_loop(), worker.loop)
    assert await asyncio.to_thread(loop_held.wait, 10)

    async def outputs():
        await asyncio.sleep(budget * 0.6)
        yield "done", True

    # Queued 0.8 budgets, then ran 0.6: 1.4 budgets after it was queued, but
    # only 0.6 after it started.
    result = await asyncio.wait_for(run_block_off_loop(outputs()), timeout=30)
    assert result == {"done": [True]}
    worker.loop.call_soon_threadsafe(worker.loop.stop)


@pytest.mark.asyncio
async def test_run_past_the_budget_is_cancelled(mocker: pytest_mock.MockFixture):
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS", 0.5
    )
    cancelled = threading.Event()

    async def outputs():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        yield "never", None

    with pytest.raises(DirectBlockExecutionTimeoutError):
        await asyncio.wait_for(run_block_off_loop(outputs()), timeout=10)
    assert await asyncio.to_thread(cancelled.wait, 10)


def test_each_user_has_a_cap_on_runs_in_flight():
    limit = DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER
    slots = [claim_direct_block_execution_slot("user-a") for _ in range(limit)]
    try:
        with pytest.raises(DirectBlockExecutionLimitError, match=f"{limit} blocks"):
            claim_direct_block_execution_slot("user-a")
        # Another user is not affected.
        claim_direct_block_execution_slot("user-b").release()
        slots.pop().release()
        slots.append(claim_direct_block_execution_slot("user-a"))
    finally:
        for slot in slots:
            slot.release()
    assert "user-a" not in direct_block_execution._runs_per_user


@pytest.mark.asyncio
async def test_slot_is_held_until_the_block_stops(mocker: pytest_mock.MockFixture):
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER", 1
    )
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS", 0.5
    )
    stop = threading.Event()
    stopped = threading.Event()

    async def ignores_cancel():
        # Synchronous wait: the block keeps running after its run times out.
        stop.wait(timeout=10)
        stopped.set()
        yield "done", True

    with claim_direct_block_execution_slot("user-a") as slot:
        with pytest.raises(DirectBlockExecutionTimeoutError):
            await asyncio.wait_for(
                run_block_off_loop(ignores_cancel(), slot), timeout=10
            )

    try:
        with pytest.raises(DirectBlockExecutionLimitError):
            claim_direct_block_execution_slot("user-a")
    finally:
        stop.set()
    assert await asyncio.to_thread(stopped.wait, 10)

    async def slot_free() -> None:
        while "user-a" in direct_block_execution._runs_per_user:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(slot_free(), timeout=10)
    claim_direct_block_execution_slot("user-a").release()
