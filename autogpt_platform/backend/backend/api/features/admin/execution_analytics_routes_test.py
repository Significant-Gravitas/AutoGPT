from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.admin.execution_analytics_routes import (
    ExecutionAnalyticsRequest,
    ExecutionAnalyticsResult,
    _process_batch,
    generate_execution_analytics,
)
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
    request = ExecutionAnalyticsRequest(graph_id="graph-1")
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
    request = ExecutionAnalyticsRequest(graph_id="graph-1")

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
    request = ExecutionAnalyticsRequest(graph_id="graph-1")
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
            ExecutionAnalyticsRequest(graph_id="graph-1"),
            admin_user_id="admin-1",
        )

    assert response.processed_executions == 1
    assert response.successful_analytics == 0
    assert response.failed_analytics == 0
    assert response.skipped_executions == 1
    assert response.results == [skipped_result]
