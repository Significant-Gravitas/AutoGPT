"""A new run reads its user, graph settings and workspace side by side: each
stand-in answers only once all three are being awaited."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.data.execution import ExecutionStatus, GraphExecutionWithNodes
from backend.data.graph import GraphSettings
from backend.data.model import User
from backend.executor.utils import add_graph_execution
from backend.util.testing import Rendezvous


@pytest.fixture(autouse=True)
def _account_state(mocker: MockerFixture) -> None:
    mocker.patch(
        "backend.executor.utils.onboarding_db.increment_onboarding_runs",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.executor.utils._spend_approval_required",
        new=AsyncMock(return_value=None),
    )


def _created_run(mocker: MockerFixture) -> MagicMock:
    run = mocker.MagicMock(spec=GraphExecutionWithNodes)
    run.is_dry_run = False
    run.user_id = "user-1"
    run.graph_id = "graph-1"
    run.graph_version = 1
    run.credential_inputs = None
    run.organization_id = None
    run.team_id = None
    run.expert_id = None
    run.id = "run-1"
    run.node_executions = []
    run.status = ExecutionStatus.QUEUED
    return run


@pytest.mark.asyncio
async def test_a_new_run_reads_its_user_settings_and_workspace_side_by_side(
    mocker: MockerFixture,
) -> None:
    meeting = Rendezvous(3)
    now = datetime.now(UTC)
    user = User(
        id="user-1",
        email="user@example.com",
        name=None,
        created_at=now,
        updated_at=now,
        stripe_customer_id=None,
        top_up_config=None,
        timezone="Europe/Paris",
    )
    run = _created_run(mocker)
    db = mocker.MagicMock()
    db.get_user_by_id = AsyncMock(side_effect=meeting.returning(user))
    db.get_graph_settings = AsyncMock(
        side_effect=meeting.returning(
            GraphSettings(
                human_in_the_loop_safe_mode=False, sensitive_action_safe_mode=True
            )
        )
    )
    db.get_or_create_workspace = AsyncMock(
        side_effect=meeting.returning(MagicMock(id="workspace-1"))
    )
    db.create_graph_execution = AsyncMock(return_value=run)
    db.update_graph_execution_stats = AsyncMock(return_value=run)
    db.update_node_execution_status_batch = AsyncMock()
    db.increment_onboarding_runs = AsyncMock()
    db.resolve_default_tenancy = AsyncMock(return_value=(None, None))
    mocker.patch(
        "backend.executor.utils.validate_and_construct_node_execution_input",
        return_value=(MagicMock(version=1), [], {}, set()),
    )
    mocker.patch("backend.executor.utils.prisma").is_connected.return_value = False
    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client", return_value=db
    )
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue", return_value=AsyncMock()
    )
    event_bus = MagicMock()
    event_bus.publish = AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus", return_value=event_bus
    )

    assert await add_graph_execution(graph_id="graph-1", user_id="user-1") is run

    context = run.to_graph_execution_entry.call_args.kwargs["execution_context"]
    assert context.user_timezone == "Europe/Paris"
    assert context.workspace_id == "workspace-1"
    assert context.human_in_the_loop_safe_mode is False
    assert context.sensitive_action_safe_mode is True
