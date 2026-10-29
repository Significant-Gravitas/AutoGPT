import asyncio
import logging

import pytest

from backend.util.background import _background_tasks, spawn_background_task


@pytest.mark.asyncio
async def test_spawn_background_task_holds_a_strong_reference_until_done():
    """asyncio only weak-refs running tasks, so the module set is what keeps a
    detached task from being collected mid-flight."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()

    task = spawn_background_task(work(), name="held-task")
    await started.wait()
    assert task in _background_tasks

    release.set()
    await task
    assert task not in _background_tasks


@pytest.mark.asyncio
async def test_spawn_background_task_logs_failures_instead_of_swallowing_them(
    caplog: pytest.LogCaptureFixture,
):
    """A detached task has no awaiter, so a raised exception would otherwise go
    unobserved."""

    async def boom() -> None:
        raise ValueError("kaboom")

    with caplog.at_level(logging.WARNING, logger="backend.util.background"):
        task = spawn_background_task(boom(), name="failing-task")
        with pytest.raises(ValueError):
            await task
        # The done-callback runs on the next loop iteration.
        await asyncio.sleep(0)

    assert task not in _background_tasks
    assert any(
        "failing-task" in record.getMessage() and record.exc_info
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_spawn_background_task_does_not_log_cancellation():
    started = asyncio.Event()

    async def work() -> None:
        started.set()
        await asyncio.sleep(60)

    task = spawn_background_task(work(), name="cancelled-task")
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert task not in _background_tasks
