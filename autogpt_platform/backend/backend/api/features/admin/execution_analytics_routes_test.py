from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.admin.execution_analytics_routes import (
    ExecutionAnalyticsRequest,
    ExecutionAnalyticsResult,
    _process_batch,
    generate_execution_analytics,
    router,
)
from backend.data.analytics import AccuracyLatestData, AccuracyTrendsResponse
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.executor.activity_status_generator import INSUFFICIENT_BALANCE_SUMMARY
from backend.util.exceptions import ExecutionFailureReason


@pytest.mark.asyncio
async def test_process_batch_allows_credit_failure_without_openai_key():
    execution = SimpleNamespace(
        id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        user_id="user-1",
        status=ExecutionStatus.FAILED,
        stats=GraphExecutionMeta.Stats(
            error="Organization has 0 credits but needs 25",
            failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
        ),
        started_at=None,
        ended_at=None,
    )
    request = ExecutionAnalyticsRequest.model_validate({"graph_id": "graph-1"})
    with (
        patch(
            "backend.api.features.admin.execution_analytics_routes.update_graph_execution_stats",
            new=AsyncMock(),
        ) as mock_update,
        patch(
            "backend.executor.activity_status_generator.get_openai_client"
        ) as mock_get_openai_client,
    ):
        results = await _process_batch([execution], request, AsyncMock())

    assert results[0].status == "success"
    assert results[0].score == 0.0
    mock_update.assert_awaited_once()
    assert mock_update.await_args is not None
    update_call = mock_update.await_args.kwargs
    assert update_call["graph_exec_id"] == "exec-1"
    updated_stats = update_call["stats"]
    assert updated_stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE
    assert updated_stats.correctness_score == 0.0
    assert updated_stats.activity_status == INSUFFICIENT_BALANCE_SUMMARY
    mock_get_openai_client.assert_not_called()


@pytest.mark.asyncio
async def test_process_batch_skips_non_credit_failure_without_llm_client():
    execution = SimpleNamespace(
        id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        user_id="user-1",
        status=ExecutionStatus.FAILED,
        stats=GraphExecutionMeta.Stats(error="Connection timeout"),
        started_at=None,
        ended_at=None,
    )
    db_client = AsyncMock()
    request = ExecutionAnalyticsRequest.model_validate({"graph_id": "graph-1"})

    with (
        patch(
            "backend.api.features.admin.execution_analytics_routes.update_graph_execution_stats",
            new=AsyncMock(),
        ) as mock_update,
        patch(
            "backend.executor.activity_status_generator.get_openai_client",
            return_value=None,
        ) as mock_get_openai_client,
    ):
        results = await _process_batch([execution], request, db_client)

    assert len(results) == 1
    assert results[0].status == "skipped"
    assert results[0].summary_text is None
    assert results[0].score is None
    assert results[0].error_message == "Activity generation returned None"
    mock_update.assert_not_awaited()
    mock_get_openai_client.assert_called_once_with(prefer_openrouter=True)
    db_client.get_node_executions.assert_not_awaited()
    db_client.get_graph_metadata.assert_not_awaited()
    db_client.get_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_analytics_only_skips_complete_existing_analysis():
    complete_execution = SimpleNamespace(
        id="exec-complete",
        graph_id="graph-1",
        graph_version=1,
        user_id="user-1",
        status=ExecutionStatus.FAILED,
        stats=GraphExecutionMeta.Stats(
            activity_status="Existing deterministic summary",
            correctness_score=0.0,
        ),
        started_at=None,
        ended_at=None,
    )
    legacy_execution = SimpleNamespace(
        id="exec-legacy",
        graph_id="graph-1",
        graph_version=1,
        user_id="user-1",
        status=ExecutionStatus.COMPLETED,
        stats=GraphExecutionMeta.Stats(
            activity_status="Legacy summary without a score",
            correctness_score=None,
        ),
        started_at=None,
        ended_at=None,
    )
    request = ExecutionAnalyticsRequest.model_validate({"graph_id": "graph-1"})
    processed_result = ExecutionAnalyticsResult(
        agent_id="graph-1",
        version_id=1,
        user_id="user-1",
        exec_id="exec-legacy",
        summary_text="Regenerated summary",
        score=0.8,
        status="success",
    )
    db_client = AsyncMock()

    with (
        patch(
            "backend.api.features.admin.execution_analytics_routes.get_db_async_client",
            return_value=db_client,
        ),
        patch(
            "backend.api.features.admin.execution_analytics_routes.get_graph_executions",
            new=AsyncMock(return_value=[complete_execution, legacy_execution]),
        ),
        patch(
            "backend.api.features.admin.execution_analytics_routes._process_batch",
            new=AsyncMock(return_value=[processed_result]),
        ) as mock_process_batch,
    ):
        response = await generate_execution_analytics(
            request,
            admin_user_id="admin-1",
        )

    assert response.processed_executions == 1
    assert response.skipped_executions == 1
    assert {result.exec_id: result.status for result in response.results} == {
        "exec-complete": "skipped",
        "exec-legacy": "success",
    }
    mock_process_batch.assert_awaited_once_with(
        [legacy_execution],
        request,
        db_client,
    )


@pytest.mark.asyncio
async def test_generate_analytics_counts_processed_rows_skipped():
    execution = SimpleNamespace(
        id="exec-no-client",
        graph_id="graph-1",
        graph_version=1,
        user_id="user-1",
        status=ExecutionStatus.COMPLETED,
        stats=GraphExecutionMeta.Stats(),
        started_at=None,
        ended_at=None,
    )
    skipped_result = ExecutionAnalyticsResult(
        agent_id="graph-1",
        version_id=1,
        user_id="user-1",
        exec_id="exec-no-client",
        summary_text=None,
        score=None,
        status="skipped",
        error_message="Activity generation returned None",
    )

    with (
        patch(
            "backend.api.features.admin.execution_analytics_routes.get_db_async_client",
            return_value=AsyncMock(),
        ),
        patch(
            "backend.api.features.admin.execution_analytics_routes.get_graph_executions",
            new=AsyncMock(return_value=[execution]),
        ),
        patch(
            "backend.api.features.admin.execution_analytics_routes._process_batch",
            new=AsyncMock(return_value=[skipped_result]),
        ),
    ):
        response = await generate_execution_analytics(
            ExecutionAnalyticsRequest.model_validate({"graph_id": "graph-1"}),
            admin_user_id="admin-1",
        )

    assert response.processed_executions == 1
    assert response.successful_analytics == 0
    assert response.failed_analytics == 0
    assert response.skipped_executions == 1
    assert response.results == [skipped_result]


# --- GET /admin/execution_accuracy_trends: days_back bounds -----------------


@pytest.fixture
def accuracy_client(mock_jwt_admin):
    app = fastapi.FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield fastapi.testclient.TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("days_back", [-1, 0, 6, 91, 100000000])
def test_accuracy_trends_rejects_out_of_range_days_back(accuracy_client, days_back):
    with patch(
        "backend.api.features.admin.execution_analytics_routes.get_accuracy_trends_and_alerts",
        new=AsyncMock(),
    ) as mock_trends:
        response = accuracy_client.get(
            "/admin/execution_accuracy_trends",
            params={"graph_id": "graph-1", "days_back": days_back},
        )

    assert response.status_code == 422
    mock_trends.assert_not_awaited()


@pytest.mark.parametrize("days_back", [None, 7, 90])
def test_accuracy_trends_accepts_in_range_days_back(accuracy_client, days_back):
    params: dict[str, object] = {"graph_id": "graph-1"}
    if days_back is not None:
        params["days_back"] = days_back
    empty = AccuracyTrendsResponse(
        latest_data=AccuracyLatestData(
            date=datetime(2026, 10, 17, tzinfo=timezone.utc),
            daily_score=None,
            three_day_avg=None,
            seven_day_avg=None,
            fourteen_day_avg=None,
        ),
        alert=None,
    )
    with patch(
        "backend.api.features.admin.execution_analytics_routes.get_accuracy_trends_and_alerts",
        new=AsyncMock(return_value=empty),
    ) as mock_trends:
        response = accuracy_client.get(
            "/admin/execution_accuracy_trends", params=params
        )

    assert response.status_code == 200
    assert mock_trends.await_args is not None
    assert mock_trends.await_args.kwargs["days_back"] == (days_back or 30)
