from datetime import datetime, timezone
from typing import Any, Literal, cast

import prisma.enums
import prisma.models
from prisma.enums import ResourceVisibility
from pydantic import BaseModel

from backend.data.sharing.workspace_refs import extract_workspace_file_ids
from backend.util.json import SafeJson

WorkflowDeliveryTarget = Literal["message", "workspace_files"]
WorkflowDeliveryStatus = Literal[
    "queued", "running", "delivered", "partial", "blocked", "failed"
]
WorkflowTerminalDeliveryStatus = Literal["delivered", "partial", "blocked", "failed"]


class WorkflowValidationEvidence(BaseModel):
    id: str
    graph_version: int
    test_execution_id: str
    artifacts: list[dict[str, str]]


_TARGET_TO_DB = {
    "message": prisma.enums.ExpertWorkflowDeliveryTarget.MESSAGE,
    "workspace_files": prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES,
}
_STATUS_TO_DB = {
    "queued": prisma.enums.ExpertWorkflowDeliveryStatus.QUEUED,
    "running": prisma.enums.ExpertWorkflowDeliveryStatus.RUNNING,
    "delivered": prisma.enums.ExpertWorkflowDeliveryStatus.DELIVERED,
    "partial": prisma.enums.ExpertWorkflowDeliveryStatus.PARTIAL,
    "blocked": prisma.enums.ExpertWorkflowDeliveryStatus.BLOCKED,
    "failed": prisma.enums.ExpertWorkflowDeliveryStatus.FAILED,
}
_DB_TO_STATUS: dict[
    prisma.enums.ExpertWorkflowDeliveryStatus, WorkflowDeliveryStatus
] = {
    prisma.enums.ExpertWorkflowDeliveryStatus.QUEUED: "queued",
    prisma.enums.ExpertWorkflowDeliveryStatus.RUNNING: "running",
    prisma.enums.ExpertWorkflowDeliveryStatus.DELIVERED: "delivered",
    prisma.enums.ExpertWorkflowDeliveryStatus.PARTIAL: "partial",
    prisma.enums.ExpertWorkflowDeliveryStatus.BLOCKED: "blocked",
    prisma.enums.ExpertWorkflowDeliveryStatus.FAILED: "failed",
}


def artifact_manifest(
    outputs: dict[str, list[Any]], artifact_output_names: list[str]
) -> list[dict[str, str]]:
    names = artifact_output_names or list(outputs)
    artifacts: list[dict[str, str]] = []
    for output_name in names:
        for file_id in sorted(extract_workspace_file_ids(outputs.get(output_name))):
            artifacts.append(
                {
                    "output_name": output_name,
                    "file_id": file_id,
                    "uri": f"workspace://{file_id}",
                }
            )
    return artifacts


async def record_workflow_validation(
    *,
    user_id: str,
    library_agent_id: str,
    graph_id: str,
    graph_version: int,
    test_execution_id: str,
    session_id: str,
    transport_succeeded: bool,
    node_error_count: int,
    node_failures: list[object],
    delivery_target: WorkflowDeliveryTarget,
    artifact_output_names: list[str],
    outputs: dict[str, list[Any]],
) -> bool:
    library_agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={
            "id": library_agent_id,
            "userId": user_id,
            "agentGraphId": graph_id,
            "agentGraphVersion": graph_version,
            "isArchived": False,
            "isDeleted": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if library_agent is None:
        return False

    artifacts = artifact_manifest(outputs, artifact_output_names)
    required_artifacts_present = delivery_target == "message" or bool(artifacts)
    passed = (
        transport_succeeded
        and node_error_count == 0
        and not node_failures
        and required_artifacts_present
    )
    await prisma.models.ExpertWorkflowValidation.prisma().upsert(
        where={"testExecutionId": test_execution_id},
        data={
            "create": {
                "userId": user_id,
                "libraryAgentId": library_agent_id,
                "testExecutionId": test_execution_id,
                "sessionId": session_id,
                "graphId": graph_id,
                "graphVersion": graph_version,
                "status": (
                    prisma.enums.ExpertWorkflowValidationStatus.PASSED
                    if passed
                    else prisma.enums.ExpertWorkflowValidationStatus.FAILED
                ),
                "deliveryTarget": _TARGET_TO_DB[delivery_target],
                "artifactOutputNames": artifact_output_names,
                "artifacts": SafeJson(artifacts),
                "requiredArtifactsPresent": required_artifacts_present,
                "nodeErrorCount": max(0, node_error_count),
                "nodeFailures": SafeJson(node_failures),
            },
            "update": {
                "status": (
                    prisma.enums.ExpertWorkflowValidationStatus.PASSED
                    if passed
                    else prisma.enums.ExpertWorkflowValidationStatus.FAILED
                ),
                "deliveryTarget": _TARGET_TO_DB[delivery_target],
                "artifactOutputNames": artifact_output_names,
                "artifacts": SafeJson(artifacts),
                "requiredArtifactsPresent": required_artifacts_present,
                "nodeErrorCount": max(0, node_error_count),
                "nodeFailures": SafeJson(node_failures),
            },
        },
    )
    return passed


async def has_passed_workflow_validation(
    *,
    user_id: str,
    library_agent_id: str,
    delivery_target: WorkflowDeliveryTarget = "message",
    artifact_output_names: list[str] | None = None,
) -> bool:
    return (
        await get_passed_workflow_validation(
            user_id=user_id,
            library_agent_id=library_agent_id,
            delivery_target=delivery_target,
            artifact_output_names=artifact_output_names or [],
        )
        is not None
    )


async def get_workflow_delivery_statuses(
    *, user_id: str, execution_ids: list[str]
) -> dict[str, WorkflowTerminalDeliveryStatus]:
    if not execution_ids:
        return {}
    rows = await prisma.models.ExpertWorkflowRunState.prisma().find_many(
        where={
            "userId": user_id,
            "executionId": {"in": list(dict.fromkeys(execution_ids))},
        }
    )
    statuses: dict[str, WorkflowTerminalDeliveryStatus] = {}
    for row in rows:
        status = _DB_TO_STATUS.get(row.status)
        if status in {"delivered", "partial", "blocked", "failed"}:
            statuses[row.executionId] = cast(WorkflowTerminalDeliveryStatus, status)
    return statuses


async def get_passed_workflow_validation(
    *,
    user_id: str,
    library_agent_id: str,
    delivery_target: WorkflowDeliveryTarget,
    artifact_output_names: list[str],
) -> WorkflowValidationEvidence | None:
    library_agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={
            "id": library_agent_id,
            "userId": user_id,
            "isCreatedByUser": True,
            "isArchived": False,
            "isDeleted": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    if library_agent is None:
        return None
    validation = await prisma.models.ExpertWorkflowValidation.prisma().find_first(
        where={
            "userId": user_id,
            "libraryAgentId": library_agent_id,
            "graphId": library_agent.agentGraphId,
            "graphVersion": library_agent.agentGraphVersion,
            "status": prisma.enums.ExpertWorkflowValidationStatus.PASSED,
            "requiredArtifactsPresent": True,
            "nodeErrorCount": 0,
        },
        order={"createdAt": "desc"},
    )
    if validation is None:
        return None
    stored_artifacts = (
        validation.artifacts if isinstance(validation.artifacts, list) else []
    )
    artifacts = [
        artifact
        for artifact in stored_artifacts
        if isinstance(artifact, dict)
        and isinstance(artifact.get("file_id"), str)
        and isinstance(artifact.get("uri"), str)
    ]
    if delivery_target == "workspace_files":
        found_outputs = {
            artifact.get("output_name")
            for artifact in artifacts
            if isinstance(artifact.get("output_name"), str)
        }
        if not artifacts or (
            artifact_output_names
            and not set(artifact_output_names).issubset(found_outputs)
        ):
            return None
    return WorkflowValidationEvidence(
        id=validation.id,
        graph_version=validation.graphVersion,
        test_execution_id=validation.testExecutionId,
        artifacts=artifacts,
    )


async def record_workflow_run_start(
    *,
    user_id: str,
    expert_id: str,
    graph_id: str,
    graph_version: int,
    execution_id: str,
) -> bool:
    workflow = await _workflow_for_graph(
        user_id=user_id,
        expert_id=expert_id,
        graph_id=graph_id,
        graph_version=graph_version,
    )
    if workflow is None:
        return False
    delivery_target: WorkflowDeliveryTarget = (
        "workspace_files"
        if workflow.deliveryTarget
        == prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES
        else "message"
    )
    artifact_output_names = workflow.artifactOutputNames
    await prisma.models.ExpertWorkflowRunState.prisma().upsert(
        where={"executionId": execution_id},
        data={
            "create": {
                "userId": user_id,
                "expertId": expert_id,
                "workflowId": workflow.id,
                "executionId": execution_id,
                "status": prisma.enums.ExpertWorkflowDeliveryStatus.RUNNING,
                "deliveryTarget": _TARGET_TO_DB[delivery_target],
                "artifactOutputNames": artifact_output_names,
            },
            "update": {
                "status": prisma.enums.ExpertWorkflowDeliveryStatus.RUNNING,
                "deliveryTarget": _TARGET_TO_DB[delivery_target],
                "artifactOutputNames": artifact_output_names,
            },
        },
    )
    return True


async def finalize_workflow_run(
    *,
    user_id: str,
    expert_id: str,
    graph_id: str,
    graph_version: int,
    execution_id: str,
    transport_status: Literal["completed", "failed"],
    node_error_count: int,
    node_failures: list[object],
    outputs: dict[str, list[Any]],
) -> WorkflowDeliveryStatus | None:
    existing = await prisma.models.ExpertWorkflowRunState.prisma().find_unique(
        where={"executionId": execution_id}
    )
    workflow = None
    if existing is None:
        workflow = await _workflow_for_graph(
            user_id=user_id,
            expert_id=expert_id,
            graph_id=graph_id,
            graph_version=graph_version,
        )
        if workflow is None:
            return None

    source = existing or workflow
    if source is None:
        return None
    delivery_target: WorkflowDeliveryTarget = (
        "workspace_files"
        if source.deliveryTarget
        == prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES
        else "message"
    )
    artifact_output_names = source.artifactOutputNames
    artifacts = artifact_manifest(outputs, artifact_output_names)
    required_artifacts_present = delivery_target == "message" or bool(artifacts)
    blocker: str | None = None
    if transport_status == "failed":
        status: WorkflowDeliveryStatus = "failed"
        blocker = "The workflow execution failed."
    elif node_error_count > 0 or node_failures:
        status = "partial"
        blocker = "One or more workflow steps failed."
    elif not required_artifacts_present:
        status = "blocked"
        blocker = "The required workspace deliverable was not produced."
    else:
        status = "delivered"

    now = datetime.now(timezone.utc)
    workflow_id = existing.workflowId if existing else source.id
    create: dict[str, Any] = {
        "userId": user_id,
        "expertId": expert_id,
        "workflowId": workflow_id,
        "executionId": execution_id,
        "status": _STATUS_TO_DB[status],
        "deliveryTarget": _TARGET_TO_DB[delivery_target],
        "artifactOutputNames": artifact_output_names,
        "artifacts": SafeJson(artifacts),
        "requiredArtifactsPresent": required_artifacts_present,
        "nodeErrorCount": max(0, node_error_count),
        "nodeFailures": SafeJson(node_failures),
        "blocker": blocker,
        "completedAt": now,
    }
    await prisma.models.ExpertWorkflowRunState.prisma().upsert(
        where={"executionId": execution_id},
        data={
            "create": create,
            "update": {
                "status": _STATUS_TO_DB[status],
                "artifacts": SafeJson(artifacts),
                "requiredArtifactsPresent": required_artifacts_present,
                "nodeErrorCount": max(0, node_error_count),
                "nodeFailures": SafeJson(node_failures),
                "blocker": blocker,
                "completedAt": now,
            },
        },
    )
    return status


async def _workflow_for_graph(
    *, user_id: str, expert_id: str, graph_id: str, graph_version: int
) -> prisma.models.ExpertWorkflow | None:
    return await prisma.models.ExpertWorkflow.prisma().find_first(
        where={
            "expertId": expert_id,
            "Expert": {
                "is": {
                    "ownerUserId": user_id,
                    "isTemplate": False,
                    "isArchived": False,
                    "visibility": ResourceVisibility.PRIVATE,
                }
            },
            "LibraryAgent": {
                "is": {
                    "userId": user_id,
                    "agentGraphId": graph_id,
                    "agentGraphVersion": graph_version,
                    "isArchived": False,
                    "isDeleted": False,
                    "visibility": ResourceVisibility.PRIVATE,
                }
            },
        }
    )
