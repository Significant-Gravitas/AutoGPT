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


@pytest.mark.asyncio
async def test_spawn_background_task_drops_inherited_live_tenancy_scopes():
    from backend.data import db_accessors, tenancy

    live_token = tenancy._active_live_scopes.set(frozenset({("user", "org", "team")}))
    actor_token = tenancy._active_actor_scopes.set(frozenset({("user", "org")}))
    graph_token = tenancy._active_graph_scopes.set(frozenset({"graph"}))
    remote_token = db_accessors._active_live_resource_leases.set(("stale",))  # type: ignore[arg-type]
    seen: tuple[frozenset, frozenset, frozenset, tuple] | None = None

    async def work() -> None:
        nonlocal seen
        seen = (
            tenancy._active_live_scopes.get(),
            tenancy._active_actor_scopes.get(),
            tenancy._active_graph_scopes.get(),
            db_accessors._active_live_resource_leases.get(),
        )

    try:
        await spawn_background_task(work(), name="scope-free-task")
    finally:
        db_accessors._active_live_resource_leases.reset(remote_token)
        tenancy._active_graph_scopes.reset(graph_token)
        tenancy._active_actor_scopes.reset(actor_token)
        tenancy._active_live_scopes.reset(live_token)

    assert seen == (frozenset(), frozenset(), frozenset(), ())
