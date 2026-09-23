"""Guards on the two scheduled passes.

The scheduler fires these on a fixed interval and the RPC returns as soon as
the task is spawned, so a pass that outlives its tick would otherwise overlap
the next one. Both passes queue their email *before* marking the rows that
suppress a resend, so an overlap is a duplicate email, not just wasted work.
"""

import asyncio
from collections.abc import Iterator

import pytest

from backend.notifications.notifications import NotificationManager


@pytest.fixture(scope="session")
def server() -> None:
    return None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup() -> Iterator[None]:
    yield


def _manager() -> NotificationManager:
    """A manager with no service wiring: `_spawn_pass` touches only `_passes`."""
    manager = NotificationManager.__new__(NotificationManager)
    manager._passes = {}
    return manager


@pytest.mark.asyncio
async def test_a_slow_pass_is_not_started_twice():
    manager = _manager()
    started = 0
    release = asyncio.Event()

    async def slow_pass() -> None:
        nonlocal started
        started += 1
        await release.wait()

    manager._spawn_pass("flush", slow_pass)
    await asyncio.sleep(0)
    manager._spawn_pass("flush", slow_pass)  # the next tick, while still running
    await asyncio.sleep(0)

    assert started == 1, "the second tick must not start an overlapping pass"

    release.set()
    await asyncio.gather(*list(manager._passes.values()))


@pytest.mark.asyncio
async def test_the_next_tick_runs_once_the_previous_pass_finishes():
    manager = _manager()
    started = 0

    async def quick_pass() -> None:
        nonlocal started
        started += 1

    manager._spawn_pass("flush", quick_pass)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    manager._spawn_pass("flush", quick_pass)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert started == 2
    assert not manager._passes, "a finished pass must not stay registered"


@pytest.mark.asyncio
async def test_the_two_passes_do_not_block_each_other():
    manager = _manager()
    ran: list[str] = []
    release = asyncio.Event()

    async def briefings() -> None:
        ran.append("briefings")
        await release.wait()

    async def alerts() -> None:
        ran.append("alerts")

    manager._spawn_pass("send_due_briefings", briefings)
    await asyncio.sleep(0)
    manager._spawn_pass("flush_matured_alerts", alerts)
    await asyncio.sleep(0)

    assert ran == ["briefings", "alerts"]

    release.set()
    await asyncio.gather(*list(manager._passes.values()))


@pytest.mark.asyncio
async def test_a_failing_pass_does_not_wedge_the_schedule():
    """A pass that raises must still clear its slot, or the guard would latch
    on and silently stop every later tick."""
    manager = _manager()

    async def boom() -> None:
        raise RuntimeError("boom")

    manager._spawn_pass("flush", boom)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not manager._passes

    ran = False

    async def after() -> None:
        nonlocal ran
        ran = True

    manager._spawn_pass("flush", after)
    await asyncio.sleep(0)
    assert ran


@pytest.mark.asyncio
async def test_a_late_done_callback_does_not_clear_a_running_successor():
    """Done callbacks fire via `call_soon`, so a task that finished just before
    the next tick can have its callback run after the successor registers.
    Clearing by name alone would drop the guard while the successor still runs.
    """
    manager = _manager()
    release = asyncio.Event()
    started = 0

    async def first() -> None:
        return None

    async def second() -> None:
        nonlocal started
        started += 1
        await release.wait()

    manager._spawn_pass("flush", first)
    first_task = manager._passes["flush"]
    await asyncio.sleep(0)
    assert first_task.done()

    # Successor registers before the finished task's callback is dispatched.
    manager._spawn_pass("flush", second)
    second_task = manager._passes["flush"]
    assert second_task is not first_task

    await asyncio.sleep(0)
    assert manager._passes.get("flush") is second_task, "successor was evicted"

    manager._spawn_pass("flush", second)
    await asyncio.sleep(0)
    assert started == 1, "guard must still block while the successor runs"

    release.set()
    await second_task
