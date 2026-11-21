"""How a turn's expert check reads: no database, so none of experts_db_test's
seeding or cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db

_WATCHDOG_SECONDS = 30


def _reading_once_both_started(started: dict[str, asyncio.Event], own: str, result):
    """A read that returns *result* only once the other read has started too."""

    async def read(*args, **kwargs):
        started[own].set()
        await asyncio.wait_for(
            asyncio.gather(*(e.wait() for e in started.values())), _WATCHDOG_SECONDS
        )
        return result

    return read


@pytest.mark.asyncio(loop_scope="session")
async def test_check_private_expert_reads_the_hire_beside_its_owners_scope():
    """The expert's row and its owner's scope don't depend on each other, so
    a turn's check reads them side by side rather than one after the other."""
    started = {"row": asyncio.Event(), "scope": asyncio.Event()}
    row = SimpleNamespace(id="expert-1")
    manager = SimpleNamespace(
        find_first=AsyncMock(
            side_effect=_reading_once_both_started(started, "row", row)
        )
    )
    owner_scope = AsyncMock(
        side_effect=_reading_once_both_started(
            started, "scope", ("personal-org", "personal-team")
        )
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(experts_db, "get_user_default_team", new=owner_scope),
        patch.object(experts_db, "_to_model", return_value=row),
        patch.object(experts_db, "CheckedExpert", side_effect=SimpleNamespace),
    ):
        check = await experts_db.check_private_expert("owner-1", "expert-1")

    assert check.expert is row
    assert (check.organization_id, check.team_id) == ("personal-org", "personal-team")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_hire_is_the_error_when_its_owner_has_no_scope_either():
    """The scope's failure comes back first here, but a turn for an expert that
    is gone must say so, not that its tenancy is temporarily missing."""
    expert_lookup = asyncio.Event()

    async def no_row(*args, **kwargs):
        await asyncio.wait_for(expert_lookup.wait(), _WATCHDOG_SECONDS)

    async def no_scope(user_id: str):
        expert_lookup.set()
        return None, None

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=SimpleNamespace(find_first=AsyncMock(side_effect=no_row)),
        ),
        patch.object(
            experts_db, "get_user_default_team", new=AsyncMock(side_effect=no_scope)
        ),
        pytest.raises(experts_db.ExpertNotFoundError),
    ):
        await experts_db.check_private_expert("owner-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_hire_leaves_its_owners_scope_read_finished():
    """On a user's first touch the scope read creates their personal
    organization, so a missing expert waits for it instead of cancelling it
    midway or leaving it running after the check has answered."""
    finished: list[bool] = []
    scope_started = asyncio.Event()
    scope_release = asyncio.Event()
    row_finished = asyncio.Event()

    async def no_row(*args, **kwargs):
        row_task = asyncio.current_task()
        assert row_task is not None
        row_task.add_done_callback(lambda _: row_finished.set())
        await asyncio.wait_for(scope_started.wait(), _WATCHDOG_SECONDS)

    async def bootstrapping_scope(user_id: str):
        scope_started.set()
        await scope_release.wait()
        finished.append(True)
        return "personal-org", "personal-team"

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=SimpleNamespace(find_first=AsyncMock(side_effect=no_row)),
        ),
        patch.object(
            experts_db,
            "get_user_default_team",
            new=AsyncMock(side_effect=bootstrapping_scope),
        ),
    ):
        check = asyncio.create_task(
            experts_db.check_private_expert("owner-1", "expert-1")
        )
        try:
            await asyncio.wait_for(row_finished.wait(), _WATCHDOG_SECONDS)
            assert not check.done()
            assert finished == []
            scope_release.set()
            with pytest.raises(experts_db.ExpertNotFoundError):
                await asyncio.wait_for(check, _WATCHDOG_SECONDS)
        finally:
            await _clean_up_check(check, scope_release)

    assert finished == [True]


async def _clean_up_check(task: asyncio.Task, release: asyncio.Event) -> None:
    release.set()
    done, _ = await asyncio.wait({task}, timeout=_WATCHDOG_SECONDS)
    if task not in done:
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio(loop_scope="session")
async def test_cancellation_after_the_row_finishes_still_finishes_the_scope_read():
    row = SimpleNamespace(id="expert-1")
    row_finished = asyncio.Event()
    scope_started = asyncio.Event()
    scope_release = asyncio.Event()
    scope_finished = asyncio.Event()
    scope_tasks: list[asyncio.Task] = []

    async def read_row(user_id: str, expert_id: str):
        await asyncio.wait_for(scope_started.wait(), _WATCHDOG_SECONDS)
        row_finished.set()
        return row

    async def read_scope(user_id: str, expert_id: str):
        scope_task = asyncio.current_task()
        assert scope_task is not None
        scope_tasks.append(scope_task)
        scope_started.set()
        await scope_release.wait()
        scope_finished.set()
        return "personal-org", "personal-team"

    with (
        patch.object(experts_db, "_active_private_expert_row", new=read_row),
        patch.object(experts_db, "_private_expert_owner_scope", new=read_scope),
        patch.object(experts_db, "_to_model", return_value=row),
        patch.object(experts_db, "CheckedExpert", side_effect=SimpleNamespace),
    ):
        check = asyncio.create_task(
            experts_db.check_private_expert("owner-1", "expert-1")
        )
        try:
            await asyncio.wait_for(row_finished.wait(), _WATCHDOG_SECONDS)
            await asyncio.sleep(0)
            check.cancel("after-row")
            await asyncio.sleep(0)

            assert not check.done()
            scope_release.set()
            done, _ = await asyncio.wait({check}, timeout=_WATCHDOG_SECONDS)
            assert done == {check}
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await check

            assert cancelled.value.args == ("after-row",)
            assert scope_finished.is_set()
            assert scope_tasks and all(task.done() for task in scope_tasks)
            assert [task for task in scope_tasks if not task.done()] == []
        finally:
            await _clean_up_check(check, scope_release)


@pytest.mark.asyncio(loop_scope="session")
async def test_repeated_cancellation_during_row_failure_cleanup_drains_scope():
    row_failed = asyncio.Event()
    scope_started = asyncio.Event()
    scope_release = asyncio.Event()
    scope_finished = asyncio.Event()
    scope_failure = RuntimeError("scope failed after bootstrap")
    scope_tasks: list[asyncio.Task] = []

    async def read_row(user_id: str, expert_id: str):
        await asyncio.wait_for(scope_started.wait(), _WATCHDOG_SECONDS)
        row_failed.set()
        raise experts_db.ExpertNotFoundError(expert_id)

    async def read_scope(user_id: str, expert_id: str):
        scope_task = asyncio.current_task()
        assert scope_task is not None
        scope_tasks.append(scope_task)
        scope_started.set()
        await scope_release.wait()
        scope_finished.set()
        raise scope_failure

    with (
        patch.object(experts_db, "_active_private_expert_row", new=read_row),
        patch.object(experts_db, "_private_expert_owner_scope", new=read_scope),
    ):
        check = asyncio.create_task(
            experts_db.check_private_expert("owner-1", "expert-1")
        )
        try:
            await asyncio.wait_for(row_failed.wait(), _WATCHDOG_SECONDS)
            await asyncio.sleep(0)
            check.cancel("first-cancellation")
            await asyncio.sleep(0)
            assert not check.done()

            check.cancel("second-cancellation")
            await asyncio.sleep(0)
            assert not check.done()

            scope_release.set()
            done, _ = await asyncio.wait({check}, timeout=_WATCHDOG_SECONDS)
            assert done == {check}
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await check

            assert cancelled.value.args == ("first-cancellation",)
            assert scope_finished.is_set()
            assert scope_tasks and all(task.done() for task in scope_tasks)
            assert scope_tasks[0].exception() is scope_failure
            assert [task for task in scope_tasks if not task.done()] == []
        finally:
            await _clean_up_check(check, scope_release)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_directly_cancelled_scope_task_is_observed_without_pending_work():
    row = SimpleNamespace(id="expert-1")
    scope_started = asyncio.Event()
    scope_tasks: list[asyncio.Task] = []

    async def read_row(user_id: str, expert_id: str):
        await asyncio.wait_for(scope_started.wait(), _WATCHDOG_SECONDS)
        return row

    async def cancel_scope(user_id: str, expert_id: str):
        scope_task = asyncio.current_task()
        assert scope_task is not None
        scope_tasks.append(scope_task)
        scope_started.set()
        raise asyncio.CancelledError("scope-cancelled")

    with (
        patch.object(experts_db, "_active_private_expert_row", new=read_row),
        patch.object(experts_db, "_private_expert_owner_scope", new=cancel_scope),
        pytest.raises(asyncio.CancelledError),
    ):
        await experts_db.check_private_expert("owner-1", "expert-1")

    assert len(scope_tasks) == 1
    assert scope_tasks[0].cancelled()
    assert [task for task in scope_tasks if not task.done()] == []
