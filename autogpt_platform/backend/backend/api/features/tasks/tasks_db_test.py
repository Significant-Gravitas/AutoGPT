"""Real-Prisma tests for the task spine.

Two things need a live database to be proved rather than asserted against a
mock: cross-user isolation (the ``userId`` predicate has to be *in the query*)
and the cancel cascade (which walks the tree with several statements).
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import fastapi
import httpx
import prisma.enums
import prisma.models
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.tasks import task_actions, tasks_db
from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.api.features.tasks.routes import router as tasks_router
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError
from backend.util.test import SpinTestServer, get_or_create_user_with_retry


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user_with_retry(
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


async def _seed_running_execution(
    user_id: str, task_id: str
) -> prisma.models.AgentGraphExecution:
    graph = await prisma.models.AgentGraph.prisma().create(data={"userId": user_id})
    return await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "userId": user_id,
            "executionStatus": prisma.enums.AgentExecutionStatus.RUNNING,
            "delegatedTaskId": task_id,
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
        await tasks_db.close_delegated_task(
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

    task = await tasks_db.create_delegated_task(
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

    first = await tasks_db.close_delegated_task(
        owner.id, task.id, succeeded=True, outcome_summary="All done.", spend=120
    )
    second = await tasks_db.close_delegated_task(
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
        await tasks_db.close_delegated_task(
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
    intruder = await _create_seed_user()
    task = await _seed_task(owner.id)
    owner_execution = await _seed_running_execution(owner.id, task.id)
    await _seed_running_execution(intruder.id, task.id)

    stop = AsyncMock()
    with patch.object(tasks_db.execution_utils, "stop_graph_execution", stop):
        await tasks_db.cancel_task(owner.id, task.id)

    stop.assert_awaited_once_with(graph_exec_id=owner_execution.id, user_id=owner.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_cancel_survives_an_execution_that_refuses_to_stop(
    server: SpinTestServer,
):
    """The task is already CANCELLED by the time executions are stopped, so a
    stop failure must not roll the cancel back or surface as a 500."""
    owner = await _create_seed_user()
    task = await _seed_task(owner.id)
    await _seed_running_execution(owner.id, task.id)

    with patch.object(
        tasks_db.execution_utils,
        "stop_graph_execution",
        new_callable=AsyncMock,
        side_effect=RuntimeError("executor unreachable"),
    ) as stop:
        detail = await tasks_db.cancel_task(owner.id, task.id)

    stop.assert_awaited()
    assert detail.task.status == "CANCELLED"


# ─── phase 2: delegation policy, handoff, escalate, report ─────────────


async def _seed_expert(user_id: str, name: str) -> prisma.models.Expert:
    return await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": f"{name} {uuid.uuid4().hex[:8]}",
            "role": f"{name}'s role",
            "identity": f"You are {name}.",
        }
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_subtask_delegation_back_to_an_ancestor_is_refused(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    root = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    with pytest.raises(TaskDelegationRefusedError, match="loop"):
        await tasks_db.create_delegated_task(
            owner.id,
            title="Back to Alice",
            spec="spec",
            owner_id=alice.id,
            parent_task_id=root.id,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_subtask_tree_depth_is_capped(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    cara = await _seed_expert(owner.id, "Cara")
    dave = await _seed_expert(owner.id, "Dave")

    root = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )
    child = await tasks_db.create_delegated_task(
        owner.id, title="Child", spec="spec", owner_id=bob.id, parent_task_id=root.id
    )
    grandchild = await tasks_db.create_delegated_task(
        owner.id,
        title="Grandchild",
        spec="spec",
        owner_id=cara.id,
        parent_task_id=child.id,
    )

    assert child.root_task_id == root.id
    assert grandchild.root_task_id == root.id
    assert grandchild.ancestor_expert_ids == [alice.id, bob.id, cara.id]
    with pytest.raises(TaskDelegationRefusedError, match="levels deep"):
        await tasks_db.create_delegated_task(
            owner.id,
            title="Too deep",
            spec="spec",
            owner_id=dave.id,
            parent_task_id=grandchild.id,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_handoff_swaps_owner_and_records_the_hop(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    created = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    task = await task_actions.handoff_delegated_task(
        owner.id,
        created.id,
        to_expert_id=bob.id,
        note="Needs Bob's integrations.",
        expected_updated_at=created.updated_at,
    )

    assert task.owner is not None and task.owner.id == bob.id
    assert task.handoff_count == 1
    assert task.ancestor_expert_ids == [alice.id, bob.id]
    assert [a.kind for a in task.amendments] == ["handoff"]
    assert task.amendments[0].from_expert_id == alice.id
    assert task.amendments[0].to_expert_id == bob.id


@pytest.mark.asyncio(loop_scope="session")
async def test_handoff_is_refused_after_the_cap(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    created = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )
    await prisma.models.DelegatedTask.prisma().update(
        where={"id": created.id}, data={"handoffCount": 5}
    )
    row = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": created.id}
    )
    assert row is not None

    with pytest.raises(TaskDelegationRefusedError, match="changed hands"):
        await task_actions.handoff_delegated_task(
            owner.id,
            created.id,
            to_expert_id=bob.id,
            note="One hop too many.",
            expected_updated_at=row.updatedAt,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_handoff_with_a_stale_read_is_a_retryable_conflict(
    server: SpinTestServer,
):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    created = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    stale = created.updated_at - timedelta(minutes=1)
    with pytest.raises(TaskUpdateConflictError):
        await task_actions.handoff_delegated_task(
            owner.id,
            created.id,
            to_expert_id=bob.id,
            note="Racing hop.",
            expected_updated_at=stale,
        )

    row = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": created.id}
    )
    assert row is not None
    assert row.ownerId == alice.id
    assert row.handoffCount == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_report_is_refused_while_subtasks_are_open(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    root = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )
    child = await tasks_db.create_delegated_task(
        owner.id, title="Child", spec="spec", owner_id=bob.id, parent_task_id=root.id
    )

    with pytest.raises(TaskDelegationRefusedError, match="open subtask"):
        await task_actions.report_delegated_task(
            owner.id, root.id, outcome_summary="All done."
        )

    await prisma.models.DelegatedTask.prisma().update(
        where={"id": child.id},
        data={"status": prisma.enums.DelegatedTaskStatus.CANCELLED},
    )
    task = await task_actions.report_delegated_task(
        owner.id, root.id, outcome_summary="All done."
    )
    assert task.status == "DONE"
    assert task.outcome_summary == "All done."


@pytest.mark.asyncio(loop_scope="session")
async def test_escalate_then_answer_round_trip(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    created = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    escalated = await task_actions.escalate_delegated_task(
        owner.id,
        created.id,
        question="Ship to staging or prod?",
        options=["Staging", "Prod"],
        session_id="worker-session-1",
    )
    assert escalated.status == "WAITING_USER"
    assert escalated.amendments[-1].kind == "escalation"
    assert escalated.amendments[-1].options == ["Staging", "Prod"]

    task, worker_session_id = await task_actions.answer_delegated_task(
        owner.id, created.id, answer="Staging"
    )
    assert task.status == "WORKING"
    assert worker_session_id == "worker-session-1"
    assert task.amendments[-1].kind == "answer"
    assert task.amendments[-1].note == "Staging"


@pytest.mark.asyncio(loop_scope="session")
async def test_answer_after_handoff_does_not_resume_the_old_owner_session(
    server: SpinTestServer,
):
    """A handoff while WAITING_USER moves ownership; the answer must not be
    delivered into the previous owner's escalating session."""
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    created = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    escalated = await task_actions.escalate_delegated_task(
        owner.id,
        created.id,
        question="Ship to staging or prod?",
        session_id="alice-session",
    )
    assert escalated.status == "WAITING_USER"

    handed = await task_actions.handoff_delegated_task(
        owner.id,
        created.id,
        to_expert_id=bob.id,
        note="Bob owns deploys.",
        expected_updated_at=escalated.updated_at,
    )
    assert handed.status == "WAITING_USER"

    task, worker_session_id = await task_actions.answer_delegated_task(
        owner.id, created.id, answer="Staging"
    )
    assert task.status == "WORKING"
    assert worker_session_id is None


@pytest.mark.asyncio(loop_scope="session")
async def test_manager_escalation_does_not_park_on_the_user(server: SpinTestServer):
    """target="manager" records the question for the delegator without
    changing status — WAITING_USER is reserved for the user."""
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    bob = await _seed_expert(owner.id, "Bob")
    root = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )
    child = await tasks_db.create_delegated_task(
        owner.id, title="Child", spec="spec", owner_id=bob.id, parent_task_id=root.id
    )

    escalated = await task_actions.escalate_delegated_task(
        owner.id,
        child.id,
        question="Which repo does this belong in?",
        session_id="worker-session-2",
        target="manager",
    )

    assert escalated.status == "QUEUED"
    entry = escalated.amendments[-1]
    assert entry.kind == "escalation"
    assert entry.target == "manager"
    assert entry.session_id == "worker-session-2"


@pytest.mark.asyncio(loop_scope="session")
async def test_manager_escalation_is_refused_on_a_root_task(server: SpinTestServer):
    owner = await _create_seed_user()
    alice = await _seed_expert(owner.id, "Alice")
    root = await tasks_db.create_delegated_task(
        owner.id, title="Root", spec="spec", owner_id=alice.id
    )

    with pytest.raises(TaskDelegationRefusedError, match="no delegator"):
        await task_actions.escalate_delegated_task(
            owner.id,
            root.id,
            question="Who decides?",
            target="manager",
        )

    row = await prisma.models.DelegatedTask.prisma().find_unique(where={"id": root.id})
    assert row is not None
    assert row.status == prisma.enums.DelegatedTaskStatus.QUEUED
    assert row.amendments in (None, [])


# ─── events feed ───────────────────────────────────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_task_events_never_include_another_users_tasks(server: SpinTestServer):
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    task = await _seed_task(owner.id)
    await _seed_task(intruder.id, title="Intruder task")

    events = await tasks_db.list_task_events(owner.id)

    assert [e.task_id for e in events] == [task.id]
    assert events[0].event == "working"
    assert all(
        e.task_id != task.id for e in await tasks_db.list_task_events(intruder.id)
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_task_events_since_filters_older_rows(server: SpinTestServer):
    owner = await _create_seed_user()
    await _seed_task(owner.id, title="Older")
    cutoff = datetime.fromisoformat((await tasks_db.list_task_events(owner.id))[-1].ts)
    newer_task = await _seed_task(owner.id, title="Newer")

    events = await tasks_db.list_task_events(owner.id, since=cutoff)

    assert [e.task_id for e in events] == [newer_task.id]


@pytest.mark.asyncio(loop_scope="session")
async def test_events_endpoint_only_returns_the_callers_rows(server: SpinTestServer):
    """End-to-end tenancy proof for GET /tasks/events: user A's poll must
    never surface user B's task events, with the real query underneath."""
    owner = await _create_seed_user()
    intruder = await _create_seed_user()
    owner_task = await _seed_task(owner.id, title="Owner task")
    await _seed_task(intruder.id, title="Intruder task")

    app = fastapi.FastAPI()
    app.include_router(tasks_router)
    app.dependency_overrides[get_jwt_payload] = lambda: {
        "sub": owner.id,
        "role": "user",
        "email": owner.email,
    }
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/tasks/events")

    assert response.status_code == 200
    events = response.json()["events"]
    assert [e["task_id"] for e in events] == [owner_task.id]
    assert events[0] == {
        "task_id": owner_task.id,
        "expert_id": None,
        "event": "working",
        "ts": events[0]["ts"],
    }
