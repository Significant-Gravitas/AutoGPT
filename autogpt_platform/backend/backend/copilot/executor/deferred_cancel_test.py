"""A cancel that reaches a turn's cleanup waits for each step (#17286).

``processor_cancel_cleanup_test.py`` runs the Stop that hits this through the
real processor; these pin the helper's other cases.
"""

import asyncio

import pytest

from backend.copilot.executor.deferred_cancel import DeferredCancel

# The longest any one wait here may take. The tests wait on events, never on
# time, so this only ends a run that hangs.
_HANG_SECONDS = 30


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_a_step_on_its_own_task_finishes(cancelled: bool):
    deferred = DeferredCancel()
    started, release = asyncio.Event(), asyncio.Event()

    async def step() -> str:
        started.set()
        await release.wait()
        return "done"

    cleanup = asyncio.create_task(deferred.finish(step()))
    async with asyncio.timeout(_HANG_SECONDS):
        await started.wait()
        if cancelled:
            cleanup.cancel()
        release.set()
        assert await cleanup == "done"

    assert (deferred.held is not None) is cancelled


@pytest.mark.asyncio
async def test_a_steps_error_still_reaches_the_cleanup_after_a_cancel():
    deferred = DeferredCancel()
    started, release = asyncio.Event(), asyncio.Event()

    async def step() -> None:
        started.set()
        await release.wait()
        raise RuntimeError("redis down")

    cleanup = asyncio.create_task(deferred.finish(step()))
    async with asyncio.timeout(_HANG_SECONDS):
        await started.wait()
        cleanup.cancel()
        release.set()
        with pytest.raises(RuntimeError, match="redis down"):
            await cleanup

    assert isinstance(deferred.held, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_step_cancelled_on_its_own_task_is_held_as_a_cancel():
    deferred = DeferredCancel()

    async def step() -> str:
        raise asyncio.CancelledError

    assert await deferred.finish(step()) is None
    assert isinstance(deferred.held, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_cancel_cuts_a_step_on_this_task_short_but_not_the_next():
    deferred = DeferredCancel()
    started = asyncio.Event()
    ran: list[str] = []

    async def step() -> None:
        started.set()
        await asyncio.Event().wait()
        ran.append("rest of the step")

    async def cleanup() -> None:
        await deferred.run_here(step())
        ran.append("next step")

    task = asyncio.create_task(cleanup())
    async with asyncio.timeout(_HANG_SECONDS):
        await started.wait()
        task.cancel()
        await task

    assert ran == ["next step"]
    assert isinstance(deferred.held, asyncio.CancelledError)
