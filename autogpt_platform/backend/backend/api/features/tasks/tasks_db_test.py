"""Real-Prisma tests for the task spine.

Two things need a live database to be proved rather than asserted against a
mock: cross-user isolation (the ``userId`` predicate has to be *in the query*)
and the cancel cascade (which walks the tree with several statements).
"""

import uuid
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.models
import pytest

from backend.api.features.tasks import tasks_db
from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"task-seed-{suffix}@example.com",
            "name": "Task Owner",
        }
    )


async def _seed_task(
    user_id: str,
    *,
    title: str = "Draft the weekly report",
    status: prisma.enums.DelegatedTaskStatus = (
        prisma.enums.DelegatedTaskStatus.WORKING
    ),
    parent_task_id: str | None = None,
    owner_id: str | None = None,
) -> prisma.models.DelegatedTask:
    return await prisma.models.DelegatedTask.prisma().create(
        data={
            "userId": user_id,
            "ownerId": owner_id,
            "parentTaskId": parent_task_id,
            "createdByType": prisma.enums.TaskCreatedByType.USER,
            "title": title,
            "spec": "spec",
            "status": status,
        }
    )


# ─── tenancy ───────────────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_list_tasks_never_returns_another_users_tasks(server: SpinTestServer):
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    await _seed_task(owner.id, title="Owner's private task")

    assert await tasks_db.list_tasks(intruder.id) == []
    assert [task.title for task in await tasks_db.list_tasks(owner.id)] == [
        "Owner's private task"
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_get_task_is_none_for_another_user(server: SpinTestServer):
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    task = await _seed_task(owner.id)

    assert await tasks_db.get_task(intruder.id, task.id) is None
    detail = await tasks_db.get_task(owner.id, task.id)
    assert detail is not None and detail.task.id == task.id


@pytest.mark.asyncio(loop_scope="session")
async def test_get_task_children_exclude_another_users_rows(server: SpinTestServer):
    """A child row can only be reached through its parent, so its own
    ``userId`` filter is the last thing standing between a shared parent id
    and another tenant's data."""
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    parent = await _seed_task(owner.id, title="Parent")
    await _seed_task(owner.id, title="Owner child", parent_task_id=parent.id)
    await _seed_task(intruder.id, title="Intruder child", parent_task_id=parent.id)

    detail = await tasks_db.get_task(owner.id, parent.id)

    assert detail is not None
    assert [child.title for child in detail.children] == ["Owner child"]


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_task_rejects_another_users_task(server: SpinTestServer):
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    task = await _seed_task(owner.id)

    with pytest.raises(DelegatedTaskNotFoundError):
        await tasks_db.cancel_task(intruder.id, task.id)

    unchanged = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": task.id}
    )
    assert unchanged is not None
    assert unchanged.status == prisma.enums.DelegatedTaskStatus.WORKING


@pytest.mark.asyncio(loop_scope="session")
async def test_close_task_ignores_another_users_task(server: SpinTestServer):
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    task = await _seed_task(owner.id)

    assert (
        await tasks_db.close_task(
            intruder.id, task.id, succeeded=True, outcome_summary="stolen"
        )
        is None
    )
    row = await prisma.models.DelegatedTask.prisma().find_unique(where={"id": task.id})
    assert row is not None
    assert row.status == prisma.enums.DelegatedTaskStatus.WORKING
    assert row.outcomeSummary is None


# ─── create / close lifecycle ──────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_create_task_stamps_itself_as_its_own_root(server: SpinTestServer):
    owner = await _create_seed_user()

    task = await tasks_db.create_task(
        owner.id, title="Weekly Report", spec="Run it", origin_session_id=None
    )

    assert task.root_task_id == task.id
    assert task.status == "QUEUED"
    assert task.owner is None


@pytest.mark.asyncio(loop_scope="session")
async def test_close_task_returns_the_origin_session_once(server: SpinTestServer):
    """The origin session is the caller's signal to post exactly one outcome
    message; a re-fired completion must come back empty-handed."""
    owner = await _create_seed_user()
    session = await prisma.models.ChatSession.prisma().create(data={"userId": owner.id})
    task = await _seed_task(owner.id)
    await prisma.models.DelegatedTask.prisma().update(
        where={"id": task.id}, data={"originSessionId": session.id}
    )

    first = await tasks_db.close_task(
        owner.id, task.id, succeeded=True, outcome_summary="All done.", spend=120
    )
    second = await tasks_db.close_task(
        owner.id, task.id, succeeded=True, outcome_summary="All done again."
    )

    assert first == session.id
    assert second is None
    row = await prisma.models.DelegatedTask.prisma().find_unique(where={"id": task.id})
    assert row is not None
    assert row.status == prisma.enums.DelegatedTaskStatus.DONE
    assert row.outcomeSummary == "All done."
    assert row.spendTotal == 120


@pytest.mark.asyncio(loop_scope="session")
async def test_close_task_does_not_resurrect_a_cancelled_task(server: SpinTestServer):
    owner = await _create_seed_user()
    task = await _seed_task(owner.id, status=prisma.enums.DelegatedTaskStatus.CANCELLED)

    assert (
        await tasks_db.close_task(
            owner.id, task.id, succeeded=True, outcome_summary="late"
        )
        is None
    )
    row = await prisma.models.DelegatedTask.prisma().find_unique(where={"id": task.id})
    assert row is not None
    assert row.status == prisma.enums.DelegatedTaskStatus.CANCELLED


# ─── cancel cascade ────────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_cascades_to_open_descendants_only(server: SpinTestServer):
    owner = await _create_seed_user()
    root = await _seed_task(owner.id, title="Root")
    child = await _seed_task(owner.id, title="Child", parent_task_id=root.id)
    grandchild = await _seed_task(owner.id, title="Grandchild", parent_task_id=child.id)
    finished = await _seed_task(
        owner.id,
        title="Already done",
        parent_task_id=root.id,
        status=prisma.enums.DelegatedTaskStatus.DONE,
    )

    with patch.object(
        tasks_db.execution_utils, "stop_graph_execution", new_callable=AsyncMock
    ):
        await tasks_db.cancel_task(owner.id, root.id)

    statuses = {
        row.id: row.status
        for row in await prisma.models.DelegatedTask.prisma().find_many(
            where={"userId": owner.id}
        )
    }
    assert statuses[root.id] == prisma.enums.DelegatedTaskStatus.CANCELLED
    assert statuses[child.id] == prisma.enums.DelegatedTaskStatus.CANCELLED
    assert statuses[grandchild.id] == prisma.enums.DelegatedTaskStatus.CANCELLED
    # A finished task keeps its outcome: cancelling the parent must not
    # rewrite work that already landed.
    assert statuses[finished.id] == prisma.enums.DelegatedTaskStatus.DONE


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_does_not_touch_a_sibling_branch(server: SpinTestServer):
    """Descendants are found by walking ``parentTaskId``, not by reading
    ``rootTaskId`` — cancelling one branch must leave its siblings alone."""
    owner = await _create_seed_user()
    root = await _seed_task(owner.id, title="Root")
    branch = await _seed_task(owner.id, title="Branch", parent_task_id=root.id)
    sibling = await _seed_task(owner.id, title="Sibling", parent_task_id=root.id)

    with patch.object(
        tasks_db.execution_utils, "stop_graph_execution", new_callable=AsyncMock
    ):
        await tasks_db.cancel_task(owner.id, branch.id)

    rows = {
        row.id: row.status
        for row in await prisma.models.DelegatedTask.prisma().find_many(
            where={"userId": owner.id}
        )
    }
    assert rows[branch.id] == prisma.enums.DelegatedTaskStatus.CANCELLED
    assert rows[root.id] == prisma.enums.DelegatedTaskStatus.WORKING
    assert rows[sibling.id] == prisma.enums.DelegatedTaskStatus.WORKING


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_stops_only_the_callers_running_executions(
    server: SpinTestServer,
):
    """The stop call is what actually reaches into another tenant's compute,
    so the execution query has to be user-scoped too, not just task-scoped."""
    owner = await _create_seed_user()
    task = await _seed_task(owner.id)

    stop = AsyncMock()
    with patch.object(tasks_db.execution_utils, "stop_graph_execution", stop):
        await tasks_db.cancel_task(owner.id, task.id)

    for call in stop.await_args_list:
        assert call.kwargs["user_id"] == owner.id


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_survives_an_execution_that_refuses_to_stop(
    server: SpinTestServer,
):
    """The task is already CANCELLED by the time executions are stopped, so a
    stop failure must not roll the cancel back or surface as a 500."""
    owner = await _create_seed_user()
    task = await _seed_task(owner.id)

    with patch.object(
        tasks_db.execution_utils,
        "stop_graph_execution",
        new_callable=AsyncMock,
        side_effect=RuntimeError("executor unreachable"),
    ):
        detail = await tasks_db.cancel_task(owner.id, task.id)

    assert detail.task.status == "CANCELLED"
