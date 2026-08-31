"""DB-level tests for the dream proactive pass.

Real-postgres tests via the session ``server`` fixture. The two invariants
that matter: a SUGGEST expert's proposal never becomes running work, and a
DREAM-created task at the budget cap is stopped dead.
"""

import uuid

import prisma.enums
import prisma.models
import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.dream.proactive import enforce_task_budget_caps, run_proactive_pass
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

# DB tests share the session loop so the Prisma pool stays bound to one loop
# (mirrors experts_db_test.py / task_review_test.py).
pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_seed_user():
    suffix = uuid.uuid4().hex[:8]
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"proactive-{suffix}@example.com",
            "name": "Proactive Owner",
        }
    )


async def _seed_expert(
    user_id: str, autonomy: prisma.enums.ExpertAutonomyLevel
) -> prisma.models.Expert:
    return await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": f"Expert {uuid.uuid4().hex[:8]}",
            "role": "Marketing",
            "identity": "You are a marketing expert.",
            "autonomyLevel": autonomy,
        }
    )


async def _seed_dream_task(
    user_id: str,
    expert_id: str,
    *,
    spend: int,
    status: prisma.enums.DelegatedTaskStatus = prisma.enums.DelegatedTaskStatus.WORKING,
) -> prisma.models.DelegatedTask:
    row = await prisma.models.DelegatedTask.prisma().create(
        data={
            "userId": user_id,
            "ownerId": expert_id,
            "createdByType": prisma.enums.TaskCreatedByType.DREAM,
            "createdById": "pass-prior",
            "title": "Prior dream task",
            "spec": "Keep working.",
            "status": status,
            "spendTotal": spend,
        }
    )
    stamped = await prisma.models.DelegatedTask.prisma().update(
        where={"id": row.id}, data={"rootTaskId": row.id}
    )
    return stamped or row


async def test_budget_cap_stops_over_budget_dream_task(server: SpinTestServer):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id, prisma.enums.ExpertAutonomyLevel.AUTONOMOUS)
    over = await _seed_dream_task(user.id, expert.id, spend=30)
    under = await _seed_dream_task(user.id, expert.id, spend=5)

    stopped = await enforce_task_budget_caps(user.id, 25)

    assert stopped == 1
    over_row = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": over.id}
    )
    assert over_row is not None
    assert over_row.status == prisma.enums.DelegatedTaskStatus.FAILED
    assert over_row.outcomeSummary is not None
    assert "25-credit cap" in over_row.outcomeSummary

    under_row = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": under.id}
    )
    assert under_row is not None
    assert under_row.status == prisma.enums.DelegatedTaskStatus.WORKING


async def test_budget_cap_at_exact_cap_is_enforced(server: SpinTestServer):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id, prisma.enums.ExpertAutonomyLevel.ASK_FIRST)
    task = await _seed_dream_task(user.id, expert.id, spend=25)

    assert await enforce_task_budget_caps(user.id, 25) == 1
    row = await prisma.models.DelegatedTask.prisma().find_unique(where={"id": task.id})
    assert row is not None
    assert row.status == prisma.enums.DelegatedTaskStatus.FAILED


async def test_suggest_expert_gets_proposal_that_never_executes(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id, prisma.enums.ExpertAutonomyLevel.SUGGEST)

    result = await run_proactive_pass(
        user.id, dream_pass_id="pass-suggest", config=ChatConfig()
    )

    assert len(result.created) == 1
    created = result.created[0]
    assert created.expert_id == expert.id
    assert created.proposal_only is True

    task = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": created.task_id}
    )
    assert task is not None
    assert task.status == prisma.enums.DelegatedTaskStatus.QUEUED
    assert task.acceptance == prisma.enums.DelegatedTaskAcceptance.PENDING
    assert task.createdByType == prisma.enums.TaskCreatedByType.DREAM
    assert task.createdById == "pass-suggest"
    assert task.originSessionId is None
    assert "do not start any work" in task.spec

    # The pass proposed only — nothing was executed for this user.
    assert (
        await prisma.models.AgentGraphExecution.prisma().count(
            where={"userId": user.id}
        )
        == 0
    )


async def test_ask_first_expert_gets_accepted_task(server: SpinTestServer):
    user = await _create_seed_user()
    await _seed_expert(user.id, prisma.enums.ExpertAutonomyLevel.ASK_FIRST)

    result = await run_proactive_pass(
        user.id, dream_pass_id="pass-ask", config=ChatConfig()
    )

    assert len(result.created) == 1
    assert result.created[0].proposal_only is False
    task = await prisma.models.DelegatedTask.prisma().find_unique(
        where={"id": result.created[0].task_id}
    )
    assert task is not None
    assert task.status == prisma.enums.DelegatedTaskStatus.QUEUED
    assert task.acceptance == prisma.enums.DelegatedTaskAcceptance.ACCEPTED


async def test_busy_expert_is_skipped(server: SpinTestServer):
    user = await _create_seed_user()
    expert = await _seed_expert(user.id, prisma.enums.ExpertAutonomyLevel.AUTONOMOUS)
    await _seed_dream_task(user.id, expert.id, spend=0)

    result = await run_proactive_pass(
        user.id, dream_pass_id="pass-busy", config=ChatConfig()
    )
    assert result.created == []
