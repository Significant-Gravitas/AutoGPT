from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.enums
import prisma.models

from backend.api.features.experts import workflow_state


def test_artifact_manifest_keeps_output_provenance() -> None:
    outputs = {
        "report": ["workspace://11111111-1111-1111-1111-111111111111"],
        "debug": ["workspace://22222222-2222-2222-2222-222222222222"],
    }

    assert workflow_state.artifact_manifest(outputs, ["report"]) == [
        {
            "output_name": "report",
            "file_id": "11111111-1111-1111-1111-111111111111",
            "uri": "workspace://11111111-1111-1111-1111-111111111111",
        }
    ]


async def test_failed_nodes_persist_a_failed_validation() -> None:
    library_manager = SimpleNamespace(
        find_first=AsyncMock(return_value=SimpleNamespace(id="library-1"))
    )
    validation_manager = SimpleNamespace(upsert=AsyncMock())
    with (
        patch.object(
            prisma.models.LibraryAgent, "prisma", return_value=library_manager
        ),
        patch.object(
            prisma.models.ExpertWorkflowValidation,
            "prisma",
            return_value=validation_manager,
        ),
    ):
        passed = await workflow_state.record_workflow_validation(
            user_id="user-1",
            library_agent_id="library-1",
            graph_id="graph-1",
            graph_version=3,
            test_execution_id="run-1",
            session_id="session-1",
            transport_succeeded=True,
            node_error_count=1,
            node_failures=[{"node": "failed"}],
            delivery_target="message",
            artifact_output_names=[],
            outputs={},
        )

    assert passed is False
    create = validation_manager.upsert.await_args.kwargs["data"]["create"]
    assert create["status"] == prisma.enums.ExpertWorkflowValidationStatus.FAILED
    assert create["nodeErrorCount"] == 1


async def test_validation_must_match_current_version_and_required_artifacts() -> None:
    library_manager = SimpleNamespace(
        find_first=AsyncMock(
            return_value=SimpleNamespace(agentGraphId="graph-1", agentGraphVersion=4)
        )
    )
    validation_manager = SimpleNamespace(
        find_first=AsyncMock(
            return_value=SimpleNamespace(
                id="validation-1",
                graphVersion=4,
                testExecutionId="run-1",
                artifacts=[
                    {
                        "output_name": "report",
                        "file_id": "11111111-1111-1111-1111-111111111111",
                        "uri": "workspace://11111111-1111-1111-1111-111111111111",
                    }
                ],
            )
        )
    )
    with (
        patch.object(
            prisma.models.LibraryAgent, "prisma", return_value=library_manager
        ),
        patch.object(
            prisma.models.ExpertWorkflowValidation,
            "prisma",
            return_value=validation_manager,
        ),
    ):
        evidence = await workflow_state.get_passed_workflow_validation(
            user_id="user-1",
            library_agent_id="library-1",
            delivery_target="workspace_files",
            artifact_output_names=["report"],
        )
        missing = await workflow_state.get_passed_workflow_validation(
            user_id="user-1",
            library_agent_id="library-1",
            delivery_target="workspace_files",
            artifact_output_names=["missing"],
        )

    assert evidence is not None
    assert evidence.graph_version == 4
    assert evidence.test_execution_id == "run-1"
    assert missing is None
    where = validation_manager.find_first.await_args_list[0].kwargs["where"]
    assert where["graphVersion"] == 4
    assert where["status"] == prisma.enums.ExpertWorkflowValidationStatus.PASSED


async def test_installed_workflow_contract_drives_run_state() -> None:
    workflow = SimpleNamespace(
        id="workflow-1",
        deliveryTarget=prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES,
        artifactOutputNames=["report"],
    )
    run_manager = SimpleNamespace(upsert=AsyncMock())
    with (
        patch.object(
            workflow_state,
            "_workflow_for_graph",
            AsyncMock(return_value=workflow),
        ),
        patch.object(
            prisma.models.ExpertWorkflowRunState,
            "prisma",
            return_value=run_manager,
        ),
    ):
        recorded = await workflow_state.record_workflow_run_start(
            user_id="user-1",
            expert_id="expert-1",
            graph_id="graph-1",
            graph_version=4,
            execution_id="run-1",
        )

    assert recorded is True
    create = run_manager.upsert.await_args.kwargs["data"]["create"]
    assert create["status"] == prisma.enums.ExpertWorkflowDeliveryStatus.RUNNING
    assert (
        create["deliveryTarget"]
        == prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES
    )
    assert create["artifactOutputNames"] == ["report"]


async def test_completed_transport_with_node_errors_is_partial() -> None:
    existing = SimpleNamespace(
        workflowId="workflow-1",
        deliveryTarget=prisma.enums.ExpertWorkflowDeliveryTarget.MESSAGE,
        artifactOutputNames=[],
    )
    run_manager = SimpleNamespace(
        find_unique=AsyncMock(return_value=existing), upsert=AsyncMock()
    )
    with patch.object(
        prisma.models.ExpertWorkflowRunState,
        "prisma",
        return_value=run_manager,
    ):
        status = await workflow_state.finalize_workflow_run(
            user_id="user-1",
            expert_id="expert-1",
            graph_id="graph-1",
            graph_version=4,
            execution_id="run-1",
            transport_status="completed",
            node_error_count=1,
            node_failures=[],
            outputs={"result": ["usable partial output"]},
        )

    assert status == "partial"
    update = run_manager.upsert.await_args.kwargs["data"]["update"]
    assert update["status"] == prisma.enums.ExpertWorkflowDeliveryStatus.PARTIAL
    assert update["blocker"] == "One or more workflow steps failed."


async def test_missing_required_workspace_artifact_is_blocked() -> None:
    existing = SimpleNamespace(
        workflowId="workflow-1",
        deliveryTarget=prisma.enums.ExpertWorkflowDeliveryTarget.WORKSPACE_FILES,
        artifactOutputNames=["report"],
    )
    run_manager = SimpleNamespace(
        find_unique=AsyncMock(return_value=existing), upsert=AsyncMock()
    )
    with patch.object(
        prisma.models.ExpertWorkflowRunState,
        "prisma",
        return_value=run_manager,
    ):
        status = await workflow_state.finalize_workflow_run(
            user_id="user-1",
            expert_id="expert-1",
            graph_id="graph-1",
            graph_version=4,
            execution_id="run-1",
            transport_status="completed",
            node_error_count=0,
            node_failures=[],
            outputs={"result": ["not a workspace file"]},
        )

    assert status == "blocked"
    update = run_manager.upsert.await_args.kwargs["data"]["update"]
    assert update["requiredArtifactsPresent"] is False
    assert update["status"] == prisma.enums.ExpertWorkflowDeliveryStatus.BLOCKED


async def test_delivery_status_lookup_is_owner_scoped_and_deduplicated() -> None:
    run_manager = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                SimpleNamespace(
                    executionId="run-1",
                    status=prisma.enums.ExpertWorkflowDeliveryStatus.PARTIAL,
                )
            ]
        )
    )
    with patch.object(
        prisma.models.ExpertWorkflowRunState,
        "prisma",
        return_value=run_manager,
    ):
        statuses = await workflow_state.get_workflow_delivery_statuses(
            user_id="user-1", execution_ids=["run-1", "run-1"]
        )

    assert statuses == {"run-1": "partial"}
    where = run_manager.find_many.await_args.kwargs["where"]
    assert where == {"userId": "user-1", "executionId": {"in": ["run-1"]}}
