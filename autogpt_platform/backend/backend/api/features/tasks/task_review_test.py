"""Real-Prisma tests for the phase-3 writers.

Like ``tasks_db_test``, these need a live database: the revision-cap and
acceptance flips are optimistic-concurrency writes whose predicates have to
be *in the query*, and orphan reassignment appends per-row Json amendments.
"""

import uuid
from datetime import UTC, datetime, timedelta

import prisma.enums
import prisma.models
import pytest

from backend.api.features.tasks import overseer_db, task_review
from backend.data.user import get_or_create_user
from backend.util.exceptions import TaskDelegationRefusedError
from backend.util.test import SpinTestServer


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"task-review-{suffix}@example.com",
            "name": "Task Owner",
        }
    )


async def _seed_task(
    user_id: str,
    *,
    title: str = "Draft the weekly report",
    status: prisma.enums.DelegatedTaskStatus = prisma.enums.DelegatedTaskStatus.DONE,
    owner_id: str | None = None,
    revision_count: int = 0,
    outcome_summary: str | None = "Report drafted.",
) -> prisma.models.DelegatedTask:
    return await prisma.models.DelegatedTask.prisma().create(
        data={
            "userId": user_id,
            "ownerId": owner_id,
            "createdByType": prisma.enums.TaskCreatedByType.USER,
            "title": title,
            "spec": "spec",
            "status": status,
            "revisionCount": revision_count,
            "outcomeSummary": outcome_summary,
        }
    )


async def _seed_expert(user_id: str) -> prisma.models.Expert:
    return await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Maria",
            "role": "Marketing",
            "identity": "Marketing strategist",
            "visibility": prisma.enums.ResourceVisibility.PRIVATE,
        }
    )


# ─── accept / reject ───────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_accept_stamps_acceptance(server: SpinTestServer):
    user = await _create_seed_user()
    row = await _seed_task(user.id)

    task = await task_review.accept_delegated_task(user.id, row.id)

    assert task.acceptance == "ACCEPTED"


@pytest.mark.asyncio(loop_scope="session")
async def test_accept_refuses_an_open_task(server: SpinTestServer):
    user = await _create_seed_user()
    row = await _seed_task(user.id, status=prisma.enums.DelegatedTaskStatus.WORKING)

    with pytest.raises(TaskDelegationRefusedError):
        await task_review.accept_delegated_task(user.id, row.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_reject_reopens_the_task_for_the_same_owner(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id)
    row = await _seed_task(user.id, owner_id=expert.id)

    task, reopened = await task_review.reject_delegated_task(
        user.id, row.id, note="Wrong quarter — use Q3 numbers."
    )

    assert reopened is True
    # The task itself goes back to work — acceptance resets so the next
    # report_task presents a fresh outcome for review.
    assert task.status == "WORKING"
    assert task.acceptance == "PENDING"
    assert task.revision_count == 1
    assert task.amendments[-1].kind == "revision"
    assert "Q3" in task.amendments[-1].note

    # No revision subtask is spawned anymore.
    children = await prisma.models.DelegatedTask.prisma().find_many(
        where={"parentTaskId": row.id}
    )
    assert children == []


@pytest.mark.asyncio(loop_scope="session")
async def test_reject_at_the_cap_escalates_instead_of_looping(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    row = await _seed_task(user.id, revision_count=2)

    task, reopened = await task_review.reject_delegated_task(
        user.id, row.id, note="Still wrong."
    )

    assert reopened is False
    assert task.status == "DONE"
    assert task.acceptance == "REJECTED"
    assert task.revision_count == 2
    children = await prisma.models.DelegatedTask.prisma().find_many(
        where={"parentTaskId": row.id}
    )
    assert children == []


# ─── amendments ────────────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_append_amendment_records_a_user_note_on_an_open_task(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    row = await _seed_task(user.id, status=prisma.enums.DelegatedTaskStatus.WORKING)

    task = await task_review.append_task_amendment(
        user.id, row.id, note="Also include churn numbers.", by="user"
    )

    assert task is not None
    assert task.amendments[-1].kind == "note"
    assert task.amendments[-1].by == "user"
    assert task.amendments[-1].note == "Also include churn numbers."


@pytest.mark.asyncio(loop_scope="session")
async def test_append_amendment_skips_a_closed_task(server: SpinTestServer):
    user = await _create_seed_user()
    row = await _seed_task(user.id, status=prisma.enums.DelegatedTaskStatus.DONE)

    assert (
        await task_review.append_task_amendment(
            user.id, row.id, note="too late", by="user"
        )
        is None
    )


# ─── overseer writes ───────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_mark_task_stale_only_touches_waiting_rows(server: SpinTestServer):
    user = await _create_seed_user()
    waiting = await _seed_task(
        user.id, status=prisma.enums.DelegatedTaskStatus.WAITING_USER
    )
    working = await _seed_task(user.id, status=prisma.enums.DelegatedTaskStatus.WORKING)
    stamp = datetime.now(UTC)

    assert await overseer_db.mark_task_stale(user.id, waiting.id, stamp) is True
    assert await overseer_db.mark_task_stale(user.id, waiting.id, stamp) is False
    assert await overseer_db.mark_task_stale(user.id, working.id, stamp) is False

    refreshed = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": waiting.id}
    )
    assert refreshed is not None and refreshed.staleAt is not None
    assert refreshed.status == prisma.enums.DelegatedTaskStatus.WAITING_USER


@pytest.mark.asyncio(loop_scope="session")
async def test_reassign_open_tasks_moves_them_to_autopilot(server: SpinTestServer):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id)
    open_row = await _seed_task(
        user.id,
        owner_id=expert.id,
        status=prisma.enums.DelegatedTaskStatus.WORKING,
    )
    done_row = await _seed_task(user.id, owner_id=expert.id)

    assert await overseer_db.count_open_tasks_for_expert(user.id, expert.id) == 1
    reassigned = await overseer_db.reassign_open_tasks_to_autopilot(user.id, expert.id)
    assert reassigned == 1

    moved = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": open_row.id}
    )
    kept = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": done_row.id}
    )
    assert moved is not None and moved.ownerId is None
    assert isinstance(moved.amendments, list)
    assert moved.amendments[-1]["kind"] == "handoff"
    assert kept is not None and kept.ownerId == expert.id


@pytest.mark.asyncio(loop_scope="session")
async def test_failed_task_counts_group_by_expert(server: SpinTestServer):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id)
    for _ in range(3):
        await _seed_task(
            user.id,
            owner_id=expert.id,
            status=prisma.enums.DelegatedTaskStatus.FAILED,
        )

    counts = await overseer_db.count_recent_failed_tasks_by_expert(
        user.id, since=datetime.now(UTC) - timedelta(days=7)
    )

    assert counts == {expert.id: 3}
