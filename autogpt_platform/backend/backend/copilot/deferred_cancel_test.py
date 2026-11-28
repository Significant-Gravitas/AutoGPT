"""A cancel that reaches a turn's cleanup waits for each step (#17286).

``executor/processor_cancel_cleanup_test.py`` runs the Stop that hits this
through the real processor; these pin the helper's other cases.
"""

import asyncio
import logging

import pytest

from backend.copilot import deferred_cancel
from backend.copilot.deferred_cancel import DeferredCancel

# The longest any one wait here may take. The tests wait on events, never on
# time, so this only ends a run that hangs. They wait through asyncio.shield:
# a task running DeferredCancel would hold back the timeout's own cancel.
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
        assert await asyncio.shield(cleanup) == "done"

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
            await asyncio.shield(cleanup)

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
        await asyncio.shield(task)

    assert ran == ["next step"]
    assert isinstance(deferred.held, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_step_that_outlasts_a_held_cancel_is_given_up(monkeypatch, caplog):
    """A hung step can't hold the turn's task for good once a cancel is held."""
    monkeypatch.setattr(deferred_cancel, "HELD_CANCEL_LIMIT_SECONDS", 0.05)
    deferred = DeferredCancel()
    started, step_cancelled = asyncio.Event(), asyncio.Event()

    async def hung_step() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            step_cancelled.set()
            raise
        return "never"

    cleanup = asyncio.create_task(deferred.finish(hung_step()))
    with caplog.at_level(logging.WARNING, logger=deferred_cancel.__name__):
        async with asyncio.timeout(_HANG_SECONDS):
            await started.wait()
            cleanup.cancel()
            assert await asyncio.shield(cleanup) is None
            await step_cancelled.wait()

    assert isinstance(deferred.held, asyncio.CancelledError)
    assert "still running 0.05s after a cancel was held" in caplog.text


@pytest.mark.asyncio
async def test_the_whole_held_cleanup_gets_one_limit(monkeypatch):
    """Steps after the one a cancel was held in share its deadline: on their
    own tasks, on the cleanup's task, and in a DeferredCancel made inside a
    step, as a credit-billed turn's charge is. Each would finish in half a
    fresh limit; none is given one."""
    limit = 0.1
    monkeypatch.setattr(deferred_cancel, "HELD_CANCEL_LIMIT_SECONDS", limit)
    deferred = DeferredCancel()
    started = asyncio.Event()
    finished: list[str] = []

    async def hangs() -> None:
        started.set()
        await asyncio.Event().wait()

    async def half_a_limit(name: str) -> None:
        await asyncio.sleep(limit / 2)
        finished.append(name)

    async def nested_cleanup() -> None:
        nested = DeferredCancel()
        try:
            await nested.finish(half_a_limit("nested step"))
        finally:
            if nested.held is not None:
                raise nested.held

    async def cleanup() -> None:
        await deferred.finish(hangs())
        await deferred.finish(half_a_limit("next step"))
        await deferred.run_here(half_a_limit("step on this task"))
        await deferred.run_here(nested_cleanup())

    task = asyncio.create_task(cleanup())
    async with asyncio.timeout(_HANG_SECONDS):
        await started.wait()
        task.cancel()
        await asyncio.shield(task)

    assert finished == []
    assert isinstance(deferred.held, asyncio.CancelledError)
