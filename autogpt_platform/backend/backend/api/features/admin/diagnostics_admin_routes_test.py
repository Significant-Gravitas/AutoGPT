from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import AgentExecutionStatus

import backend.api.features.admin.diagnostics_admin_routes as diagnostics_admin_routes
from backend.data.diagnostics import (
    AgentDiagnosticsSummary,
    ExecutionDiagnosticsSummary,
    FailedExecutionDetail,
    OrphanedScheduleDetail,
    RunningExecutionDetail,
    ScheduleDetail,
    ScheduleHealthMetrics,
)
from backend.data.execution import GraphExecutionMeta

app = fastapi.FastAPI()
app.include_router(diagnostics_admin_routes.router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Setup admin auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_get_execution_diagnostics_success(
    mocker: pytest_mock.MockFixture,
):
    """Test fetching execution diagnostics with invalid state detection"""
    mock_diagnostics = ExecutionDiagnosticsSummary(
        running_count=10,
        queued_db_count=5,
        rabbitmq_queue_depth=3,
        cancel_queue_depth=0,
        orphaned_running=2,
        orphaned_queued=1,
        failed_count_1h=5,
        failed_count_24h=20,
        failure_rate_24h=0.83,
        stuck_running_24h=1,
        stuck_running_1h=3,
        oldest_running_hours=26.5,
        stuck_queued_1h=2,
        queued_never_started=1,
        invalid_queued_with_start=1,  # New invalid state
        invalid_running_without_start=1,  # New invalid state
        completed_1h=50,
        completed_24h=1200,
        throughput_per_hour=50.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=mock_diagnostics,
    )

    response = client.get("/admin/diagnostics/executions")

    assert response.status_code == 200
    data = response.json()

    # Verify new invalid state fields are included
    assert data["invalid_queued_with_start"] == 1
    assert data["invalid_running_without_start"] == 1
    # Verify all expected fields present
    assert "running_executions" in data
    assert "orphaned_running" in data
    assert "failed_count_24h" in data


def test_list_invalid_executions(
    mocker: pytest_mock.MockFixture,
):
    """Test listing executions in invalid states (read-only endpoint)"""
    mock_invalid_executions = [
        RunningExecutionDetail(
            execution_id="exec-invalid-1",
            graph_id="graph-123",
            graph_name="Test Graph",
            graph_version=1,
            user_id="user-123",
            user_email="test@example.com",
            status="QUEUED",
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(
                timezone.utc
            ),  # QUEUED but has startedAt - INVALID!
            queue_status=None,
        ),
        RunningExecutionDetail(
            execution_id="exec-invalid-2",
            graph_id="graph-456",
            graph_name="Another Graph",
            graph_version=2,
            user_id="user-456",
            user_email="user@example.com",
            status="RUNNING",
            created_at=datetime.now(timezone.utc),
            started_at=None,  # RUNNING but no startedAt - INVALID!
            queue_status=None,
        ),
    ]

    mock_diagnostics = ExecutionDiagnosticsSummary(
        running_count=10,
        queued_db_count=5,
        rabbitmq_queue_depth=3,
        cancel_queue_depth=0,
        orphaned_running=0,
        orphaned_queued=0,
        failed_count_1h=0,
        failed_count_24h=0,
        failure_rate_24h=0.0,
        stuck_running_24h=0,
        stuck_running_1h=0,
        oldest_running_hours=None,
        stuck_queued_1h=0,
        queued_never_started=0,
        invalid_queued_with_start=1,
        invalid_running_without_start=1,
        completed_1h=0,
        completed_24h=0,
        throughput_per_hour=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_invalid_executions_details",
        return_value=mock_invalid_executions,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=mock_diagnostics,
    )

    response = client.get("/admin/diagnostics/executions/invalid?limit=100&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2  # Sum of both invalid state types
    assert len(data["executions"]) == 2
    # Verify both types of invalid states are returned
    assert data["executions"][0]["execution_id"] in [
        "exec-invalid-1",
        "exec-invalid-2",
    ]
    assert data["executions"][1]["execution_id"] in [
        "exec-invalid-1",
        "exec-invalid-2",
    ]


def test_requeue_single_execution_with_add_graph_execution(
    mocker: pytest_mock.MockFixture,
    admin_user_id: str,
):
    """Test requeueing uses add_graph_execution in requeue mode"""
    mock_exec_meta = GraphExecutionMeta(
        id="exec-stuck-123",
        user_id="user-123",
        graph_id="graph-456",
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=AgentExecutionStatus.QUEUED,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        stats=None,
    )

    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[mock_exec_meta],
    )

    mock_add_graph_execution = mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.add_graph_execution",
        return_value=AsyncMock(),
    )

    response = client.post(
        "/admin/diagnostics/executions/requeue",
        json={"execution_id": "exec-stuck-123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["requeued_count"] == 1

    # Verify it used add_graph_execution in requeue mode
    mock_add_graph_execution.assert_called_once()
    call_kwargs = mock_add_graph_execution.call_args.kwargs
    assert call_kwargs["graph_exec_id"] == "exec-stuck-123"  # Requeue mode!
    assert call_kwargs["graph_id"] == "graph-456"
    assert call_kwargs["user_id"] == "user-123"


def test_stop_single_execution_with_stop_graph_execution(
    mocker: pytest_mock.MockFixture,
    admin_user_id: str,
):
    """Test stopping uses robust stop_graph_execution"""
    mock_exec_meta = GraphExecutionMeta(
        id="exec-running-123",
        user_id="user-789",
        graph_id="graph-999",
        graph_version=2,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=AgentExecutionStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        ended_at=datetime.now(timezone.utc),
        stats=None,
    )

    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[mock_exec_meta],
    )

    mock_stop_graph_execution = mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.stop_graph_execution",
        return_value=AsyncMock(),
    )

    response = client.post(
        "/admin/diagnostics/executions/stop",
        json={"execution_id": "exec-running-123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 1

    # Verify it used stop_graph_execution with cascade
    mock_stop_graph_execution.assert_called_once()
    call_kwargs = mock_stop_graph_execution.call_args.kwargs
    assert call_kwargs["graph_exec_id"] == "exec-running-123"
    assert call_kwargs["user_id"] == "user-789"
    assert call_kwargs["cascade"] is True  # Stops children too!
    assert call_kwargs["wait_timeout"] == 15.0


def test_requeue_not_queued_execution_fails(
    mocker: pytest_mock.MockFixture,
):
    """Test that requeue fails if execution is not in QUEUED status"""
    # Mock an execution that's RUNNING (not QUEUED)
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[],  # No QUEUED executions found
    )

    response = client.post(
        "/admin/diagnostics/executions/requeue",
        json={"execution_id": "exec-running-123"},
    )

    assert response.status_code == 404
    assert "not found or not in QUEUED status" in response.json()["detail"]


def test_list_invalid_executions_no_bulk_actions(
    mocker: pytest_mock.MockFixture,
):
    """Verify invalid executions endpoint is read-only (no bulk actions)"""
    # This is a documentation test - the endpoint exists but should not
    # have corresponding cleanup/stop/requeue endpoints

    # These endpoints should NOT exist for invalid states:
    invalid_bulk_endpoints = [
        "/admin/diagnostics/executions/cleanup-invalid",
        "/admin/diagnostics/executions/stop-invalid",
        "/admin/diagnostics/executions/requeue-invalid",
    ]

    for endpoint in invalid_bulk_endpoints:
        response = client.post(endpoint, json={"execution_ids": ["test"]})
        assert response.status_code == 404, f"{endpoint} should not exist (read-only)"


def test_execution_ids_filter_efficiency(
    mocker: pytest_mock.MockFixture,
):
    """Test that bulk operations use efficient execution_ids filter"""
    mock_exec_metas = [
        GraphExecutionMeta(
            id=f"exec-{i}",
            user_id=f"user-{i}",
            graph_id="graph-123",
            graph_version=1,
            inputs=None,
            credential_inputs=None,
            nodes_input_masks=None,
            preset_id=None,
            status=AgentExecutionStatus.QUEUED,
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            stats=None,
        )
        for i in range(3)
    ]

    mock_get_graph_executions = mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=mock_exec_metas,
    )

    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.add_graph_execution",
        return_value=AsyncMock(),
    )

    response = client.post(
        "/admin/diagnostics/executions/requeue-bulk",
        json={"execution_ids": ["exec-0", "exec-1", "exec-2"]},
    )

    assert response.status_code == 200

    # Verify it used execution_ids filter (not fetching all queued)
    mock_get_graph_executions.assert_called_once()
    call_kwargs = mock_get_graph_executions.call_args.kwargs
    assert "execution_ids" in call_kwargs
    assert call_kwargs["execution_ids"] == ["exec-0", "exec-1", "exec-2"]
    assert call_kwargs["statuses"] == [AgentExecutionStatus.QUEUED]


# ---------------------------------------------------------------------------
# Helper: reusable mock diagnostics summary
# ---------------------------------------------------------------------------


def _make_mock_diagnostics(**overrides) -> ExecutionDiagnosticsSummary:
    defaults = dict(
        running_count=10,
        queued_db_count=5,
        rabbitmq_queue_depth=3,
        cancel_queue_depth=0,
        orphaned_running=2,
        orphaned_queued=1,
        failed_count_1h=5,
        failed_count_24h=20,
        failure_rate_24h=0.83,
        stuck_running_24h=3,
        stuck_running_1h=5,
        oldest_running_hours=26.5,
        stuck_queued_1h=2,
        queued_never_started=1,
        invalid_queued_with_start=1,
        invalid_running_without_start=1,
        completed_1h=50,
        completed_24h=1200,
        throughput_per_hour=50.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return ExecutionDiagnosticsSummary(**defaults)


_SENTINEL = object()


def _make_mock_execution(
    exec_id: str = "exec-1",
    status: str = "RUNNING",
    started_at: datetime | None | object = _SENTINEL,
) -> RunningExecutionDetail:
    return RunningExecutionDetail(
        execution_id=exec_id,
        graph_id="graph-123",
        graph_name="Test Graph",
        graph_version=1,
        user_id="user-123",
        user_email="test@example.com",
        status=status,
        created_at=datetime.now(timezone.utc),
        started_at=(
            datetime.now(timezone.utc) if started_at is _SENTINEL else started_at
        ),
        queue_status=None,
    )


def _make_mock_failed_execution(
    exec_id: str = "exec-fail-1",
) -> FailedExecutionDetail:
    return FailedExecutionDetail(
        execution_id=exec_id,
        graph_id="graph-123",
        graph_name="Test Graph",
        graph_version=1,
        user_id="user-123",
        user_email="test@example.com",
        status="FAILED",
        created_at=datetime.now(timezone.utc),
        started_at=datetime.now(timezone.utc),
        failed_at=datetime.now(timezone.utc),
        error_message="Something went wrong",
    )


def _make_mock_schedule_health(**overrides) -> ScheduleHealthMetrics:
    defaults = dict(
        total_schedules=15,
        user_schedules=10,
        system_schedules=5,
        orphaned_deleted_graph=2,
        orphaned_no_library_access=1,
        orphaned_invalid_credentials=0,
        orphaned_validation_failed=0,
        total_orphaned=3,
        schedules_next_hour=4,
        schedules_next_24h=8,
        total_runs_next_hour=12,
        total_runs_next_24h=48,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return ScheduleHealthMetrics(**defaults)


# ---------------------------------------------------------------------------
# GET endpoints: execution list variants
# ---------------------------------------------------------------------------


def test_list_running_executions(mocker: pytest_mock.MockFixture):
    mock_execs = [
        _make_mock_execution("exec-run-1"),
        _make_mock_execution("exec-run-2"),
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_running_executions_details",
        return_value=mock_execs,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=_make_mock_diagnostics(),
    )

    response = client.get("/admin/diagnostics/executions/running?limit=50&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 15  # running_count(10) + queued_db_count(5)
    assert len(data["executions"]) == 2
    assert data["executions"][0]["execution_id"] == "exec-run-1"


def test_list_orphaned_executions(mocker: pytest_mock.MockFixture):
    mock_execs = [_make_mock_execution("exec-orphan-1", status="RUNNING")]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_orphaned_executions_details",
        return_value=mock_execs,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=_make_mock_diagnostics(),
    )

    response = client.get("/admin/diagnostics/executions/orphaned?limit=50&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3  # orphaned_running(2) + orphaned_queued(1)
    assert len(data["executions"]) == 1


def test_list_failed_executions(mocker: pytest_mock.MockFixture):
    mock_execs = [_make_mock_failed_execution("exec-fail-1")]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_failed_executions_details",
        return_value=mock_execs,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_failed_executions_count",
        return_value=42,
    )

    response = client.get(
        "/admin/diagnostics/executions/failed?limit=50&offset=0&hours=24"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 42
    assert len(data["executions"]) == 1
    assert data["executions"][0]["error_message"] == "Something went wrong"


def test_list_long_running_executions(mocker: pytest_mock.MockFixture):
    mock_execs = [_make_mock_execution("exec-long-1")]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_long_running_executions_details",
        return_value=mock_execs,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=_make_mock_diagnostics(),
    )

    response = client.get(
        "/admin/diagnostics/executions/long-running?limit=50&offset=0"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3  # stuck_running_24h
    assert len(data["executions"]) == 1


def test_list_stuck_queued_executions(mocker: pytest_mock.MockFixture):
    mock_execs = [
        _make_mock_execution("exec-stuck-1", status="QUEUED", started_at=None)
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_stuck_queued_executions_details",
        return_value=mock_execs,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_execution_diagnostics",
        return_value=_make_mock_diagnostics(),
    )

    response = client.get(
        "/admin/diagnostics/executions/stuck-queued?limit=50&offset=0"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2  # stuck_queued_1h
    assert len(data["executions"]) == 1


# ---------------------------------------------------------------------------
# GET endpoints: agent + schedule diagnostics
# ---------------------------------------------------------------------------


def test_get_agent_diagnostics(mocker: pytest_mock.MockFixture):
    mock_diag = AgentDiagnosticsSummary(
        agents_with_active_executions=7,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_agent_diagnostics",
        return_value=mock_diag,
    )

    response = client.get("/admin/diagnostics/agents")

    assert response.status_code == 200
    data = response.json()
    assert data["agents_with_active_executions"] == 7


def test_get_schedule_diagnostics(mocker: pytest_mock.MockFixture):
    mock_metrics = _make_mock_schedule_health()
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_schedule_health_metrics",
        return_value=mock_metrics,
    )

    response = client.get("/admin/diagnostics/schedules")

    assert response.status_code == 200
    data = response.json()
    assert data["user_schedules"] == 10
    assert data["total_orphaned"] == 3
    assert data["total_runs_next_hour"] == 12


def test_list_all_schedules(mocker: pytest_mock.MockFixture):
    mock_schedules = [
        ScheduleDetail(
            schedule_id="sched-1",
            schedule_name="Daily Run",
            graph_id="graph-1",
            graph_name="My Agent",
            graph_version=1,
            user_id="user-1",
            user_email="alice@example.com",
            cron="0 9 * * *",
            timezone="UTC",
            next_run_time=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_all_schedules_details",
        return_value=mock_schedules,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_schedule_health_metrics",
        return_value=_make_mock_schedule_health(),
    )

    response = client.get("/admin/diagnostics/schedules/all?limit=50&offset=0")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 10
    assert len(data["schedules"]) == 1
    assert data["schedules"][0]["schedule_name"] == "Daily Run"


def test_list_orphaned_schedules(mocker: pytest_mock.MockFixture):
    mock_orphans = [
        OrphanedScheduleDetail(
            schedule_id="sched-orphan-1",
            schedule_name="Ghost Schedule",
            graph_id="graph-deleted",
            graph_version=1,
            user_id="user-1",
            orphan_reason="deleted_graph",
            error_detail=None,
            next_run_time=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_orphaned_schedules_details",
        return_value=mock_orphans,
    )

    response = client.get("/admin/diagnostics/schedules/orphaned")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["schedules"][0]["orphan_reason"] == "deleted_graph"


# ---------------------------------------------------------------------------
# POST endpoints: bulk stop, cleanup, requeue
# ---------------------------------------------------------------------------


def test_stop_multiple_executions(mocker: pytest_mock.MockFixture):
    mock_exec_metas = [
        GraphExecutionMeta(
            id=f"exec-{i}",
            user_id=f"user-{i}",
            graph_id="graph-123",
            graph_version=1,
            inputs=None,
            credential_inputs=None,
            nodes_input_masks=None,
            preset_id=None,
            status=AgentExecutionStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            ended_at=None,
            stats=None,
        )
        for i in range(2)
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=mock_exec_metas,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.stop_graph_execution",
        return_value=AsyncMock(),
    )

    response = client.post(
        "/admin/diagnostics/executions/stop-bulk",
        json={"execution_ids": ["exec-0", "exec-1"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 2


def test_stop_multiple_executions_none_found(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[],
    )

    response = client.post(
        "/admin/diagnostics/executions/stop-bulk",
        json={"execution_ids": ["nonexistent"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["stopped_count"] == 0


def test_cleanup_orphaned_executions(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.cleanup_orphaned_executions_bulk",
        return_value=3,
    )

    response = client.post(
        "/admin/diagnostics/executions/cleanup-orphaned",
        json={"execution_ids": ["exec-1", "exec-2", "exec-3"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 3


def test_cleanup_orphaned_schedules(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.cleanup_orphaned_schedules_bulk",
        return_value=2,
    )

    response = client.post(
        "/admin/diagnostics/schedules/cleanup-orphaned",
        json={"schedule_ids": ["sched-1", "sched-2"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted_count"] == 2


def test_stop_all_long_running_executions(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.stop_all_long_running_executions",
        return_value=5,
    )

    response = client.post("/admin/diagnostics/executions/stop-all-long-running")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 5


def test_cleanup_all_orphaned_executions(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_all_orphaned_execution_ids",
        return_value=["exec-1", "exec-2"],
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.cleanup_orphaned_executions_bulk",
        return_value=2,
    )

    response = client.post("/admin/diagnostics/executions/cleanup-all-orphaned")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 2


def test_cleanup_all_orphaned_executions_none(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_all_orphaned_execution_ids",
        return_value=[],
    )

    response = client.post("/admin/diagnostics/executions/cleanup-all-orphaned")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 0
    assert "No orphaned" in data["message"]


def test_cleanup_all_stuck_queued_executions(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.cleanup_all_stuck_queued_executions",
        return_value=4,
    )

    response = client.post("/admin/diagnostics/executions/cleanup-all-stuck-queued")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["stopped_count"] == 4


def test_requeue_all_stuck_executions(mocker: pytest_mock.MockFixture):
    mock_exec_metas = [
        GraphExecutionMeta(
            id=f"exec-stuck-{i}",
            user_id=f"user-{i}",
            graph_id="graph-123",
            graph_version=1,
            inputs=None,
            credential_inputs=None,
            nodes_input_masks=None,
            preset_id=None,
            status=AgentExecutionStatus.QUEUED,
            started_at=None,
            ended_at=None,
            stats=None,
        )
        for i in range(3)
    ]
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_all_stuck_queued_execution_ids",
        return_value=["exec-stuck-0", "exec-stuck-1", "exec-stuck-2"],
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=mock_exec_metas,
    )
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.add_graph_execution",
        return_value=AsyncMock(),
    )

    response = client.post("/admin/diagnostics/executions/requeue-all-stuck")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["requeued_count"] == 3


def test_requeue_all_stuck_executions_none(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_all_stuck_queued_execution_ids",
        return_value=[],
    )

    response = client.post("/admin/diagnostics/executions/requeue-all-stuck")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["requeued_count"] == 0
    assert "No stuck" in data["message"]


def test_requeue_bulk_none_found(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[],
    )

    response = client.post(
        "/admin/diagnostics/executions/requeue-bulk",
        json={"execution_ids": ["nonexistent"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["requeued_count"] == 0


def test_stop_single_execution_not_found(mocker: pytest_mock.MockFixture):
    mocker.patch(
        "backend.api.features.admin.diagnostics_admin_routes.get_graph_executions",
        return_value=[],
    )

    response = client.post(
        "/admin/diagnostics/executions/stop",
        json={"execution_id": "nonexistent"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
