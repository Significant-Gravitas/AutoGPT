from datetime import datetime, timezone
from typing import Literal, cast
from uuid import uuid4

import prisma.enums
import prisma.models
from prisma import Prisma
from pydantic import TypeAdapter

from backend.api.features.experts.models import (
    ExpertWorkArtifact,
    ExpertWorkConfidence,
    ExpertWorkCriterion,
    ExpertWorkItem,
    ExpertWorkStatus,
)
from backend.util.json import SafeJson

_TERMINAL_STATUSES = {
    prisma.enums.ExpertWorkItemStatus.DELIVERED,
    prisma.enums.ExpertWorkItemStatus.PARTIAL,
    prisma.enums.ExpertWorkItemStatus.BLOCKED_MANAGER,
    prisma.enums.ExpertWorkItemStatus.FAILED,
}
_STATUS_TO_DB = {
    "queued": prisma.enums.ExpertWorkItemStatus.QUEUED,
    "running": prisma.enums.ExpertWorkItemStatus.RUNNING,
    "delivered": prisma.enums.ExpertWorkItemStatus.DELIVERED,
    "partial": prisma.enums.ExpertWorkItemStatus.PARTIAL,
    "blocked_manager": prisma.enums.ExpertWorkItemStatus.BLOCKED_MANAGER,
    "failed": prisma.enums.ExpertWorkItemStatus.FAILED,
}
_DB_TO_STATUS = {value: key for key, value in _STATUS_TO_DB.items()}
_CONFIDENCE_TO_DB = {
    "verified": prisma.enums.ExpertWorkConfidence.VERIFIED,
    "likely": prisma.enums.ExpertWorkConfidence.LIKELY,
    "unknown": prisma.enums.ExpertWorkConfidence.UNKNOWN,
    "disqualified": prisma.enums.ExpertWorkConfidence.DISQUALIFIED,
}
_DB_TO_CONFIDENCE = {value: key for key, value in _CONFIDENCE_TO_DB.items()}
_CRITERIA_ADAPTER = TypeAdapter(list[ExpertWorkCriterion])
_ARTIFACT_ADAPTER = TypeAdapter(list[ExpertWorkArtifact])


async def create_work_item(
    *,
    user_id: str,
    expert_id: str,
    manager_session_id: str,
    delegated_session_id: str,
    project_phase: str,
    task_title: str,
    expected_deliverable: str,
    deliverable_mode: Literal["message", "workspace_files"],
    success_criteria: list[ExpertWorkCriterion],
    dependencies: list[str],
    source_artifacts: list[ExpertWorkArtifact],
    constraints: list[str],
    approval_boundaries: list[str],
    estimate_minutes: int | None,
    manager_wait_expires_at: datetime | None,
    work_item_id: str | None = None,
    client: Prisma | None = None,
) -> ExpertWorkItem:
    row = await prisma.models.ExpertWorkItem.prisma(client).create(
        data={
            "id": work_item_id or str(uuid4()),
            "ownerUserId": user_id,
            "expertId": expert_id,
            "managerSessionId": manager_session_id,
            "delegatedSessionId": delegated_session_id,
            "projectPhase": project_phase,
            "taskTitle": task_title,
            "expectedDeliverable": expected_deliverable,
            "deliverableMode": deliverable_mode,
            "successCriteria": SafeJson(success_criteria),
            "dependencies": dependencies,
            "sourceArtifacts": SafeJson(source_artifacts),
            "constraints": constraints,
            "approvalBoundaries": approval_boundaries,
            "estimateMinutes": estimate_minutes,
            "managerWaitExpiresAt": manager_wait_expires_at,
        }
    )
    return _to_model(row)


async def mark_work_started(work_item_id: str, user_id: str) -> ExpertWorkItem | None:
    now = datetime.now(timezone.utc)
    await prisma.models.ExpertWorkItem.prisma().update_many(
        where={
            "id": work_item_id,
            "ownerUserId": user_id,
            "status": prisma.enums.ExpertWorkItemStatus.QUEUED,
        },
        data={
            "status": prisma.enums.ExpertWorkItemStatus.RUNNING,
            "startedAt": now,
        },
    )
    return await get_work_item(work_item_id, user_id)


async def report_work_item(
    *,
    work_item_id: str,
    user_id: str,
    delegated_session_id: str,
    expert_id: str,
    status: Literal["delivered", "partial", "blocked_manager", "failed"],
    result: str,
    blocker: str | None,
    progress: int,
    confidence: ExpertWorkConfidence,
    success_criteria: list[ExpertWorkCriterion],
    artifacts: list[ExpertWorkArtifact],
) -> tuple[ExpertWorkItem | None, bool]:
    now = datetime.now(timezone.utc)
    data = {
        "status": _STATUS_TO_DB[status],
        "result": result,
        "blocker": blocker,
        "progress": max(0, min(progress, 100)),
        "confidence": _CONFIDENCE_TO_DB[confidence],
        "successCriteria": SafeJson(success_criteria),
        "artifacts": SafeJson(artifacts),
        "completedAt": now,
    }
    updated = await prisma.models.ExpertWorkItem.prisma().update_many(
        where={
            "id": work_item_id,
            "ownerUserId": user_id,
            "delegatedSessionId": delegated_session_id,
            "expertId": expert_id,
            "status": {
                "in": [
                    prisma.enums.ExpertWorkItemStatus.QUEUED,
                    prisma.enums.ExpertWorkItemStatus.RUNNING,
                ]
            },
        },
        data=data,
    )
    return await get_work_item(work_item_id, user_id), updated > 0


async def record_delegation_outcome(
    *,
    work_item_id: str,
    user_id: str,
    status: ExpertWorkStatus,
    result: str | None = None,
    blocker: str | None = None,
    progress: int | None = None,
    confidence: ExpertWorkConfidence = "unknown",
    artifacts: list[ExpertWorkArtifact] | None = None,
    parent_seen: bool = False,
) -> ExpertWorkItem | None:
    now = datetime.now(timezone.utc)
    db_status = _STATUS_TO_DB[status]
    data: dict = {
        "status": db_status,
        "confidence": _CONFIDENCE_TO_DB[confidence],
    }
    if status == "running":
        data["startedAt"] = now
    if db_status in _TERMINAL_STATUSES:
        data["completedAt"] = now
    if result is not None:
        data["result"] = result
    if blocker is not None:
        data["blocker"] = blocker
    if progress is not None:
        data["progress"] = max(0, min(progress, 100))
    if artifacts is not None:
        data["artifacts"] = SafeJson(artifacts)
    if parent_seen:
        data["parentWokenAt"] = now
    await prisma.models.ExpertWorkItem.prisma().update_many(
        where={
            "id": work_item_id,
            "ownerUserId": user_id,
            "status": {
                "in": [
                    prisma.enums.ExpertWorkItemStatus.QUEUED,
                    prisma.enums.ExpertWorkItemStatus.RUNNING,
                ]
            },
        },
        data=data,
    )
    return await get_work_item(work_item_id, user_id)


async def claim_parent_wake(work_item_id: str, user_id: str) -> bool:
    claimed = await prisma.models.ExpertWorkItem.prisma().update_many(
        where={
            "id": work_item_id,
            "ownerUserId": user_id,
            "status": {"in": list(_TERMINAL_STATUSES)},
            "parentWokenAt": None,
        },
        data={"parentWokenAt": datetime.now(timezone.utc)},
    )
    return claimed > 0


async def get_work_item(work_item_id: str, user_id: str) -> ExpertWorkItem | None:
    row = await prisma.models.ExpertWorkItem.prisma().find_first(
        where={"id": work_item_id, "ownerUserId": user_id}
    )
    return _to_model(row) if row else None


async def get_active_work_for_session(
    *, user_id: str, delegated_session_id: str, expert_id: str
) -> ExpertWorkItem | None:
    row = await prisma.models.ExpertWorkItem.prisma().find_first(
        where={
            "ownerUserId": user_id,
            "delegatedSessionId": delegated_session_id,
            "expertId": expert_id,
            "status": {
                "in": [
                    prisma.enums.ExpertWorkItemStatus.QUEUED,
                    prisma.enums.ExpertWorkItemStatus.RUNNING,
                ]
            },
        },
        order={"createdAt": "desc"},
    )
    return _to_model(row) if row else None


async def get_latest_work_for_manager(
    *, user_id: str, delegated_session_id: str, manager_session_id: str
) -> ExpertWorkItem | None:
    row = await prisma.models.ExpertWorkItem.prisma().find_first(
        where={
            "ownerUserId": user_id,
            "delegatedSessionId": delegated_session_id,
            "managerSessionId": manager_session_id,
        },
        order={"createdAt": "desc"},
    )
    return _to_model(row) if row else None


async def list_expert_work(user_id: str, expert_id: str) -> list[ExpertWorkItem]:
    owned_expert = await prisma.models.Expert.prisma().count(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False}
    )
    if not owned_expert:
        return []
    rows = await prisma.models.ExpertWorkItem.prisma().find_many(
        where={"ownerUserId": user_id, "expertId": expert_id},
        order={"createdAt": "desc"},
        take=100,
    )
    return [_to_model(row) for row in rows]


async def list_user_work(user_id: str) -> list[ExpertWorkItem]:
    rows = await prisma.models.ExpertWorkItem.prisma().find_many(
        where={"ownerUserId": user_id}, order={"createdAt": "desc"}, take=300
    )
    return [_to_model(row) for row in rows]


async def should_enqueue_parent_wake(work_item_id: str, user_id: str) -> bool:
    """Atomically claim the single manager continuation for terminal work.

    The expert reports before its final assistant message is persisted. Even
    when AutoPilot is currently waiting inline, that wait can expire between
    the report and the final message. Always enqueue the claimed continuation
    so a result cannot be stranded in that race; the active manager turn and
    the queued notice are serialized by the normal session queue.
    """
    return await claim_parent_wake(work_item_id, user_id)


def _to_model(row: prisma.models.ExpertWorkItem) -> ExpertWorkItem:
    return ExpertWorkItem(
        id=row.id,
        expert_id=row.expertId,
        manager_session_id=row.managerSessionId,
        delegated_session_id=row.delegatedSessionId,
        project_phase=row.projectPhase,
        task_title=row.taskTitle,
        expected_deliverable=row.expectedDeliverable,
        deliverable_mode=(
            "workspace_files" if row.deliverableMode == "workspace_files" else "message"
        ),
        success_criteria=_CRITERIA_ADAPTER.validate_python(row.successCriteria),
        dependencies=row.dependencies,
        source_artifacts=_ARTIFACT_ADAPTER.validate_python(row.sourceArtifacts),
        constraints=row.constraints,
        approval_boundaries=row.approvalBoundaries,
        estimate_minutes=row.estimateMinutes,
        progress=row.progress,
        status=cast(ExpertWorkStatus, _DB_TO_STATUS[row.status]),
        result=row.result,
        blocker=row.blocker,
        confidence=cast(ExpertWorkConfidence, _DB_TO_CONFIDENCE[row.confidence]),
        artifacts=_ARTIFACT_ADAPTER.validate_python(row.artifacts),
        created_at=row.createdAt,
        updated_at=row.updatedAt,
        started_at=row.startedAt,
        completed_at=row.completedAt,
        link=f"/copilot?sessionId={row.delegatedSessionId}&workItemId={row.id}",
    )
