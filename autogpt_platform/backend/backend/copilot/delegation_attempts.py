import logging
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from backend.api.features.experts.models import (
    ExpertWorkArtifact,
    ExpertWorkCriterion,
    ExpertWorkItem,
)
from backend.copilot.model import ChatSession, cache_chat_session
from backend.data.db_accessors import experts_db

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
        result = await experts_db().persist_delegation_attempt(
            session_id=session.session_id,
            user_id=work.user_id,
            organization_id=session.organization_id,
            team_id=session.team_id,
            metadata=session.metadata,
            expert_id=work.expert_id,
            manager_session_id=work.manager_session_id,
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
            create_session=create_session,
        )
    except RuntimeError as exc:
        raise DelegationPersistenceError(str(exc)) from exc

    session.expert_id = result.session.expert_id
    if create_session:
        await _cache_session(session)
    return session, result.work_item, result.created


async def _cache_session(session: ChatSession) -> None:
    try:
        await cache_chat_session(session)
    except Exception:
        logger.warning(
            "Failed to cache delegated session %s",
            session.session_id,
            exc_info=True,
        )
