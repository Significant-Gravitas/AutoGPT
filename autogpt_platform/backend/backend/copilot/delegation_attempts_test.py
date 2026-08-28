from datetime import datetime, timezone
from uuid import uuid4

import prisma.models
import pytest

from backend.api.features.experts import work_items
from backend.copilot.model import ChatSession, create_chat_session
from backend.data.db_accessors import experts_db

from .delegation_attempts import DelegationWorkSpec, persist_delegation_attempt


async def _attempt(test_user_id: str):
    suffix = uuid4().hex[:8]
    expert = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": test_user_id,
            "name": f"Delegation Expert {suffix}",
            "role": "Researcher",
            "identity": "You research delegated work.",
        }
    )
    parent = await create_chat_session(test_user_id, dry_run=False)
    organization_id, team_id = await experts_db().resolve_private_expert_tenancy(
        test_user_id, expert.id
    )
    inner = ChatSession.new(
        test_user_id,
        dry_run=False,
        expert_id=expert.id,
        delegated_by_session_id=parent.session_id,
        organization_id=organization_id,
        team_id=team_id,
    )
    work = DelegationWorkSpec(
        work_item_id=str(uuid4()),
        user_id=test_user_id,
        expert_id=expert.id,
        manager_session_id=parent.session_id,
        delegated_session_id=inner.session_id,
        project_phase="Discovery",
        task_title="Research competitors",
        expected_deliverable="A verified competitor brief",
        deliverable_mode="message",
        success_criteria=[],
        dependencies=[],
        source_artifacts=[],
        constraints=[],
        approval_boundaries=[],
        estimate_minutes=30,
        manager_wait_expires_at=datetime.now(timezone.utc),
    )
    return inner, work


@pytest.mark.asyncio(loop_scope="session")
async def test_real_delegation_create_is_atomic_and_idempotent(
    setup_test_user, test_user_id
):
    inner, work = await _attempt(test_user_id)

    persisted_session, item, created = await persist_delegation_attempt(
        session=inner,
        work=work,
        create_session=True,
    )
    retried_session, retried_item, retried = await persist_delegation_attempt(
        session=inner,
        work=work,
        create_session=True,
    )

    assert created is True
    assert retried is False
    assert persisted_session.session_id == retried_session.session_id
    assert item.id == retried_item.id
    assert (
        await prisma.models.ExpertWorkItem.prisma().count(
            where={"id": work.work_item_id}
        )
        == 1
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_work_item_failure_rolls_back_delegated_session(
    setup_test_user, test_user_id, monkeypatch
):
    inner, work = await _attempt(test_user_id)

    async def fail_work_item(**kwargs):
        raise RuntimeError("simulated work item write failure")

    monkeypatch.setattr(work_items, "create_work_item", fail_work_item)

    with pytest.raises(RuntimeError, match="simulated work item"):
        await persist_delegation_attempt(
            session=inner,
            work=work,
            create_session=True,
        )

    assert (
        await prisma.models.ChatSession.prisma().find_unique(
            where={"id": inner.session_id}
        )
        is None
    )
