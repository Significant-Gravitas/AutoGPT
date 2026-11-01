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
    try:
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

        # Queued 0.8 budgets, then ran 0.6: 1.4 budgets after it was queued,
        # but only 0.6 after it started.
        result = await asyncio.wait_for(run_block_off_loop(outputs()), timeout=30)
        assert result == {"done": [True]}
    finally:
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


@pytest.mark.asyncio
async def test_timeout_raised_by_the_block_reaches_the_caller():
    async def outputs():
        raise TimeoutError("upstream request timed out")
        yield "never", None

    with pytest.raises(TimeoutError, match="upstream request timed out"):
        await asyncio.wait_for(run_block_off_loop(outputs()), timeout=10)


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


@pytest.fixture
def short_budget_pool(mocker: pytest_mock.MockFixture) -> None:
    """A worker pool of the test's own, with a short budget and grace."""
    mocker.patch.object(direct_block_execution, "_pool", [], create=True)
    mocker.patch.object(direct_block_execution, "_retired_workers", set(), create=True)
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_TIMEOUT_SECONDS", 0.3
    )
    mocker.patch.object(
        direct_block_execution,
        "DIRECT_BLOCK_EXECUTION_CANCEL_GRACE_SECONDS",
        0.2,
        create=True,
    )


class _BlocksItsThread:
    """Block runs that hold their worker's thread until released."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.threads: list[threading.Thread] = []

    async def outputs(self):
        self.threads.append(threading.current_thread())
        # Synchronous wait: cancelling the run cannot interrupt it.
        self.release.wait(timeout=30)
        yield "done", True

    @property
    def thread_names(self) -> set[str]:
        return {thread.name for thread in self.threads}


async def _eventually(check, timeout: float = 10) -> None:
    async def poll() -> None:
        while not check():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _retired_worker_count() -> int:
    return len(getattr(direct_block_execution, "_retired_workers", ()))


async def _thread_name():
    yield "thread", threading.current_thread().name


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_budget_pool")
async def test_worker_held_past_the_budget_is_replaced():
    blocked = _BlocksItsThread()
    try:
        runs = [
            asyncio.create_task(run_block_off_loop(blocked.outputs()))
            for _ in range(DIRECT_BLOCK_EXECUTION_WORKERS)
        ]
        for run in runs:
            with pytest.raises(DirectBlockExecutionTimeoutError):
                await asyncio.wait_for(run, timeout=10)
        assert len(blocked.thread_names) == DIRECT_BLOCK_EXECUTION_WORKERS

        # Every original worker is still held, yet a new run starts promptly.
        result = await asyncio.wait_for(run_block_off_loop(_thread_name()), 5)
        assert result["thread"][0] not in blocked.thread_names

        await _eventually(
            lambda: _retired_worker_count() == DIRECT_BLOCK_EXECUTION_WORKERS
        )
        # Full capacity: as many runs as workers each hold a thread at once.
        barrier = threading.Barrier(DIRECT_BLOCK_EXECUTION_WORKERS)

        async def holds_a_thread():
            barrier.wait(timeout=5)
            yield "thread", threading.current_thread().name

        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    run_block_off_loop(holds_a_thread())
                    for _ in range(DIRECT_BLOCK_EXECUTION_WORKERS)
                )
            ),
            timeout=10,
        )
        threads = {result["thread"][0] for result in results}
        assert len(threads) == DIRECT_BLOCK_EXECUTION_WORKERS
        assert not threads & blocked.thread_names
    finally:
        blocked.release.set()

    # A retired worker's thread ends once its run stops.
    for thread in blocked.threads:
        await asyncio.to_thread(thread.join, 10)
        assert not thread.is_alive()
    await _eventually(lambda: _retired_worker_count() == 0)


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_budget_pool")
async def test_slot_on_a_retired_worker_is_released_when_the_block_stops(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch.object(
        direct_block_execution, "DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER", 1
    )
    blocked = _BlocksItsThread()
    try:
        with claim_direct_block_execution_slot("user-a") as slot:
            with pytest.raises(DirectBlockExecutionTimeoutError):
                await asyncio.wait_for(
                    run_block_off_loop(blocked.outputs(), slot), timeout=10
                )
        await _eventually(lambda: _retired_worker_count() == 1)

        # The block still runs, so the user's slot is still taken.
        with pytest.raises(DirectBlockExecutionLimitError, match="1 blocks"):
            claim_direct_block_execution_slot("user-a")
    finally:
        blocked.release.set()

    await _eventually(lambda: "user-a" not in direct_block_execution._runs_per_user)
    await asyncio.to_thread(blocked.threads[0].join, 10)
    assert not blocked.threads[0].is_alive()
    with claim_direct_block_execution_slot("user-a") as slot:
        result = await asyncio.wait_for(run_block_off_loop(_thread_name(), slot), 5)
    assert result["thread"][0] not in blocked.thread_names
    assert "user-a" not in direct_block_execution._runs_per_user


def test_one_user_cannot_fill_the_retired_worker_cap():
    assert (
        direct_block_execution.DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS
        >= 2 * DIRECT_BLOCK_EXECUTION_MAX_RUNS_PER_USER
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("short_budget_pool")
async def test_runs_are_refused_while_retired_workers_are_at_the_cap(
    mocker: pytest_mock.MockFixture, caplog: pytest.LogCaptureFixture
):
    mocker.patch.object(
        direct_block_execution,
        "DIRECT_BLOCK_EXECUTION_MAX_RETIRED_WORKERS",
        1,
        create=True,
    )
    threads_before = set(threading.enumerate())
    blocked = _BlocksItsThread()
    try:
        runs = [
            asyncio.create_task(run_block_off_loop(blocked.outputs())) for _ in range(2)
        ]
        for run in runs:
            with pytest.raises(DirectBlockExecutionTimeoutError):
                await asyncio.wait_for(run, timeout=10)
        await _eventually(lambda: "retired workers are still running" in caplog.text)

        with pytest.raises(DirectBlockExecutionLimitError, match="right now"):
            claim_direct_block_execution_slot("user-a")
        # One worker was replaced; the second held worker was kept.
        new_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in threads_before
            and thread.name.startswith("direct-block-execution-")
        ]
        assert len(new_threads) == DIRECT_BLOCK_EXECUTION_WORKERS + 1
    finally:
        blocked.release.set()

    def accepts_runs() -> bool:
        try:
            claim_direct_block_execution_slot("user-a").release()
        except DirectBlockExecutionLimitError:
            return False
        return True

    await _eventually(accepts_runs)
