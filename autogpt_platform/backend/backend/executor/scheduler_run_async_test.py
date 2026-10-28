"""``run_async``, the scheduler's bridge from its sync job bodies onto the
shared event loop: what a timeout does to the coroutine it gave up on.

Only the memory-registry bridges ask for cancellation (see
``TestMemoryScopeGateBridge`` in ``scheduler_test.py``); every other
caller's coroutine runs on, because graph dispatch is not safe to stop
midway.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.executor.scheduler import GraphExecutionJobArgs, execute_graph, run_async

_SCHEDULER_PATH = "backend.executor.scheduler"
_JOB = GraphExecutionJobArgs(
    schedule_id="sched-1",
    user_id="user-1",
    graph_id="graph-1",
    graph_version=1,
    cron="* * * * *",
    input_data={},
    input_credentials={},
)


@pytest.fixture
def scheduler_loop():
    """A real event loop on its own thread, installed as the scheduler's
    shared loop so ``run_async`` bridges to it as in production."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        with patch(f"{_SCHEDULER_PATH}._event_loop", loop):
            yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_run_async_cancels_the_coroutine_it_gives_up_on_when_asked(scheduler_loop):
    """The memory bridges ask for it: a timed-out gate must not stay on the
    shared loop with nobody waiting for it."""
    cancelled = threading.Event()

    async def hang() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError):
        run_async(hang(), timeout=0.05, cancel_on_timeout=True)
    assert cancelled.wait(timeout=2)


def test_run_async_leaves_the_coroutine_running_by_default(scheduler_loop):
    """Every other caller keeps the old behaviour: the coroutine runs to its
    end after the timeout, since not every one is safe to stop midway."""
    finished = threading.Event()

    async def slow() -> None:
        await asyncio.sleep(0.2)
        finished.set()

    with pytest.raises(TimeoutError):
        run_async(slow(), timeout=0.05)
    assert finished.wait(timeout=2)


def test_a_timed_out_graph_dispatch_still_publishes_its_execution(scheduler_loop):
    """Codex's round-2 reproduction at the durable boundary: graph dispatch
    creates its execution before its remaining lookups, so the bridge timing
    out there must not cancel it, or the execution stays INCOMPLETE and is
    never published. Once the slow lookup answers, dispatch completes."""
    dispatch = _SlowDispatch()

    def scaled(coro, timeout=None, **kwargs):
        return run_async(coro, timeout=0.05, **kwargs)

    db = MagicMock(increment_onboarding_runs=AsyncMock())
    with (
        patch(f"{_SCHEDULER_PATH}.run_async", new=scaled),
        patch(
            f"{_SCHEDULER_PATH}.execution_utils.add_graph_execution",
            new=dispatch.add,
        ),
        patch(f"{_SCHEDULER_PATH}.get_database_manager_async_client", return_value=db),
        patch(f"{_SCHEDULER_PATH}.product_analytics.track_schedule_fired"),
    ):
        with pytest.raises(TimeoutError):
            execute_graph(**_JOB.model_dump(mode="json"))
        dispatch.lookup_answered.set()
        dispatch.wait_until_published()

    assert dispatch.events == ["created", "published"]


class _SlowDispatch:
    """An ``add_graph_execution`` that creates its execution at once and
    publishes it only after a slow lookup answers."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.lookup_answered = threading.Event()

    async def add(self, **kwargs) -> MagicMock:
        self.events.append("created")
        while not self.lookup_answered.is_set():
            await asyncio.sleep(0.01)
        self.events.append("published")
        return MagicMock(id="exec-1")

    def wait_until_published(self, seconds: float = 2.0) -> None:
        until = time.monotonic() + seconds
        while "published" not in self.events and time.monotonic() < until:
            time.sleep(0.01)
