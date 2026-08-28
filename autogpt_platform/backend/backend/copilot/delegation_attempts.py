import logging
from datetime import datetime
from typing import Literal

from prisma import Prisma
from prisma.errors import UniqueViolationError
from pydantic import BaseModel

from backend.api.features.experts import work_items
from backend.api.features.experts.models import (
    ExpertWorkArtifact,
    ExpertWorkCriterion,
    ExpertWorkItem,
)
from backend.copilot import db as copilot_db
from backend.copilot.model import ChatSession, cache_chat_session, get_chat_session
from backend.data.db import transaction

logger = logging.getLogger(__name__)


class DelegationPersistenceError(RuntimeError):
    pass


class DelegationWorkSpec(BaseModel):
    work_item_id: str
    user_id: str
    expert_id: str
    manager_session_id: str
    delegated_session_id: str
    project_phase: str
    task_title: str
    expected_deliverable: str
    deliverable_mode: Literal["message", "workspace_files"]
    success_criteria: list[ExpertWorkCriterion]
    dependencies: list[str]
    source_artifacts: list[ExpertWorkArtifact]
    constraints: list[str]
    approval_boundaries: list[str]
    estimate_minutes: int | None
    manager_wait_expires_at: datetime | None


async def persist_delegation_attempt(
    *,
    session: ChatSession,
    work: DelegationWorkSpec,
    create_session: bool,
) -> tuple[ChatSession, ExpertWorkItem, bool]:
    """Persist the delegated session and work item as one retry-safe unit."""
    try:
        if create_session:
            async with transaction() as tx:
                persisted = await copilot_db.create_chat_session_in_transaction(
                    tx,
                    session.session_id,
                    work.user_id,
                    organization_id=session.organization_id,
                    team_id=session.team_id,
                    metadata=session.metadata,
                    expert_id=work.expert_id,
                )
                item = await _create_work_item(work, tx)
                session.expert_id = persisted.expert_id
            await _cache_session(session)
            return session, item, True

        item = await _create_work_item(work)
        return session, item, True
    except UniqueViolationError:
        return await _load_existing_attempt(work)


async def _create_work_item(
    work: DelegationWorkSpec, client: Prisma | None = None
) -> ExpertWorkItem:
    return await work_items.create_work_item(
        user_id=work.user_id,
        expert_id=work.expert_id,
        manager_session_id=work.manager_session_id,
        delegated_session_id=work.delegated_session_id,
        project_phase=work.project_phase,
        task_title=work.task_title,
        expected_deliverable=work.expected_deliverable,
        deliverable_mode=work.deliverable_mode,
        success_criteria=work.success_criteria,
        dependencies=work.dependencies,
        source_artifacts=work.source_artifacts,
        constraints=work.constraints,
        approval_boundaries=work.approval_boundaries,
        estimate_minutes=work.estimate_minutes,
        manager_wait_expires_at=work.manager_wait_expires_at,
        work_item_id=work.work_item_id,
        client=client,
    )


async def _load_existing_attempt(
    work: DelegationWorkSpec,
) -> tuple[ChatSession, ExpertWorkItem, bool]:
    item = await work_items.get_work_item(work.work_item_id, work.user_id)
    session = await get_chat_session(work.delegated_session_id, work.user_id)
    if item is None or session is None:
        raise DelegationPersistenceError(
            "A delegation retry found incomplete persistence state."
        )
    if (
        item.manager_session_id != work.manager_session_id
        or item.delegated_session_id != work.delegated_session_id
        or item.expert_id != work.expert_id
    ):
        raise DelegationPersistenceError(
            "A delegation retry did not match the original assignment."
        )
    return session, item, False


async def _cache_session(session: ChatSession) -> None:
    try:
        await cache_chat_session(session)
    except Exception:
        logger.warning(
            "Failed to cache delegated session %s",
            session.session_id,
            exc_info=True,
        )
