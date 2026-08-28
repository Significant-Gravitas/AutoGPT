from __future__ import annotations

from typing import cast
from uuid import uuid4

import prisma.enums
import prisma.models
from pydantic import TypeAdapter

from backend.api.features.experts.models import (
    ExpertWorkStatus,
    ProjectContext,
    ProjectContextArtifact,
    ProjectWorkOwner,
)
from backend.data.db import transaction
from backend.util.json import SafeJson

_ARTIFACTS = TypeAdapter(list[ProjectContextArtifact])
_ACTIVE_WORK_STATUSES = [
    prisma.enums.ExpertWorkItemStatus.QUEUED,
    prisma.enums.ExpertWorkItemStatus.RUNNING,
    prisma.enums.ExpertWorkItemStatus.BLOCKED_MANAGER,
]
_WORK_STATUS = {
    prisma.enums.ExpertWorkItemStatus.QUEUED: "queued",
    prisma.enums.ExpertWorkItemStatus.RUNNING: "running",
    prisma.enums.ExpertWorkItemStatus.DELIVERED: "delivered",
    prisma.enums.ExpertWorkItemStatus.PARTIAL: "partial",
    prisma.enums.ExpertWorkItemStatus.BLOCKED_MANAGER: "blocked_manager",
    prisma.enums.ExpertWorkItemStatus.FAILED: "failed",
}


async def upsert_project_context(
    *,
    user_id: str,
    manager_session_id: str,
    title: str,
    summary: str,
    phase: str,
    decisions: list[str],
    constraints: list[str],
    artifacts: list[ProjectContextArtifact],
    activate: bool = True,
) -> ProjectContext:
    async with transaction() as tx:
        manager = await tx.chatsession.find_first(
            where={"id": manager_session_id, "userId": user_id, "expertId": None}
        )
        if manager is None:
            raise ValueError("Project context can only belong to an AutoPilot session")

        if activate:
            await tx.projectcontext.update_many(
                where={"ownerUserId": user_id, "active": True},
                data={"active": False},
            )

        row = await tx.projectcontext.upsert(
            where={"managerSessionId": manager_session_id},
            data={
                "create": {
                    "id": str(uuid4()),
                    "ownerUserId": user_id,
                    "managerSessionId": manager_session_id,
                    "title": title,
                    "summary": summary,
                    "phase": phase,
                    "decisions": decisions,
                    "constraints": constraints,
                    "artifacts": SafeJson(artifacts),
                    "active": activate,
                },
                "update": {
                    "title": title,
                    "summary": summary,
                    "phase": phase,
                    "decisions": decisions,
                    "constraints": constraints,
                    "artifacts": SafeJson(artifacts),
                    "active": activate,
                },
            },
        )
    return await _with_current_work(row)


async def get_manager_project_context(
    user_id: str, manager_session_id: str
) -> ProjectContext | None:
    row = await prisma.models.ProjectContext.prisma().find_first(
        where={"ownerUserId": user_id, "managerSessionId": manager_session_id}
    )
    return await _with_current_work(row) if row else None


async def get_project_context_for_session(
    *, user_id: str, session_id: str, expert_id: str | None
) -> ProjectContext | None:
    manager_session_id = session_id if expert_id is None else None
    if expert_id is not None:
        delegated_work = await prisma.models.ExpertWorkItem.prisma().find_first(
            where={
                "ownerUserId": user_id,
                "delegatedSessionId": session_id,
                "expertId": expert_id,
            },
            order={"createdAt": "desc"},
        )
        if delegated_work is not None:
            manager_session_id = delegated_work.managerSessionId

    row = None
    if manager_session_id is not None:
        row = await prisma.models.ProjectContext.prisma().find_first(
            where={
                "ownerUserId": user_id,
                "managerSessionId": manager_session_id,
            }
        )
    if row is None:
        row = await prisma.models.ProjectContext.prisma().find_first(
            where={"ownerUserId": user_id, "active": True},
            order={"updatedAt": "desc"},
        )
    return await _with_current_work(row) if row else None


async def _with_current_work(row: prisma.models.ProjectContext) -> ProjectContext:
    work_rows = await prisma.models.ExpertWorkItem.prisma().find_many(
        where={
            "ownerUserId": row.ownerUserId,
            "managerSessionId": row.managerSessionId,
            "status": {"in": _ACTIVE_WORK_STATUSES},
        },
        include={"Expert": True},
        order={"updatedAt": "desc"},
        take=50,
    )
    current_work = [
        ProjectWorkOwner(
            expert_name=work.Expert.name if work.Expert else "Expert",
            expert_role=work.Expert.role if work.Expert else "Team member",
            task_title=work.taskTitle,
            project_phase=work.projectPhase,
            status=cast(ExpertWorkStatus, _WORK_STATUS[work.status]),
        )
        for work in work_rows
    ]
    return ProjectContext(
        id=row.id,
        manager_session_id=row.managerSessionId,
        title=row.title,
        summary=row.summary,
        phase=row.phase,
        decisions=row.decisions,
        constraints=row.constraints,
        artifacts=_ARTIFACTS.validate_python(row.artifacts),
        active=row.active,
        updated_at=row.updatedAt,
        current_work=current_work,
    )
