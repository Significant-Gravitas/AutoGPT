import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    GraphExecutionMeta,
)
from backend.data.model import GraphExecutionStats
from backend.executor.manager import (
    ExecutionProcessor,
    _get_execution_credit_balance,
    _propagate_node_failure,
    _record_execution_failure,
)
from backend.util.decorator import TimingInfo
from backend.util.exceptions import ExecutionFailureReason, InsufficientBalanceError


def test_continued_execution_clears_prior_terminal_analysis():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
    )
    execution_meta = SimpleNamespace(
        status=ExecutionStatus.RUNNING,
        stats=GraphExecutionMeta.Stats(
            error="You have no credits left to run an agent.",
            failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
            activity_status="The prior attempt ran out of credits.",
            correctness_score=0.0,
            cost=27,
            node_exec_count=3,
        ),
    )
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = execution_meta
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock(
        return_value=(
            TimingInfo(cpu_time=0.0, wall_time=0.0),
            ExecutionStatus.TERMINATED,
        )
    )

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.send_execution_update"),
        patch("backend.executor.manager.update_graph_execution_state"),
        patch("backend.executor.manager.billing.handle_agent_run_notif"),
        patch("backend.executor.manager.expert_posts.handle_expert_run_post"),
    ):
        processor.on_graph_execution(
            execution,
            threading.Event(),
            MagicMock(),
        )

    resumed_stats = processor._on_graph_execution.call_args.kwargs["execution_stats"]
    assert resumed_stats.error is None
    assert resumed_stats.failure_reason is None
    assert resumed_stats.activity_status is None
    assert resumed_stats.correctness_score is None
    assert resumed_stats.cost == 27
    assert resumed_stats.node_count == 3


def test_failed_execution_is_not_continued():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
    )
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = SimpleNamespace(
        status=ExecutionStatus.FAILED,
        stats=GraphExecutionMeta.Stats(
            error="You have no credits left to run an agent.",
            failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
            activity_status="The prior attempt ran out of credits.",
            correctness_score=0.0,
        ),
    )
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock()

    with patch("backend.executor.manager.get_db_client", return_value=db_client):
        processor.on_graph_execution(
            execution,
            threading.Event(),
            MagicMock(),
        )

    processor._on_graph_execution.assert_not_called()
    db_client.update_graph_execution_start_time.assert_not_called()


def test_execution_credit_balance_uses_organization_aware_lookup():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(organization_id="org-1"),
    )
    db_client = MagicMock()
    db_client.get_org_credits.return_value = 500
    db_client.get_credits.return_value = 0

    assert _get_execution_credit_balance(db_client, execution) == 500
    db_client.get_org_credits.assert_called_once_with(org_id="org-1")
    db_client.get_credits.assert_not_called()


def test_execution_credit_balance_uses_user_wallet_without_organization():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
    )
    db_client = MagicMock()
    db_client.get_credits.return_value = 250

    assert _get_execution_credit_balance(db_client, execution) == 250
    db_client.get_credits.assert_called_once_with("user-1")
    db_client.get_org_credits.assert_not_called()


def test_nested_credit_failure_is_propagated_to_graph_stats():
    graph_stats = GraphExecutionStats()
    error = InsufficientBalanceError(
        message="Organization has 0 credits but needs 25",
        user_id="user-1",
        balance=0,
        amount=25,
    )

    _propagate_node_failure(graph_stats, error)

    assert graph_stats.error == str(error)
    assert graph_stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE


def test_graph_error_does_not_erase_promoted_credit_failure_reason():
    graph_stats = GraphExecutionStats(
        error="Organization has 0 credits but needs 25",
        failure_reason=ExecutionFailureReason.INSUFFICIENT_BALANCE,
    )

    _record_execution_failure(graph_stats, RuntimeError("Output moderation failed"))

    assert graph_stats.error == "Output moderation failed"
    assert graph_stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE


def test_credit_node_failure_replaces_prior_graph_error():
    graph_stats = GraphExecutionStats(error="Earlier non-credit node error")
    error = InsufficientBalanceError(
        message="Organization has 0 credits but needs 25",
        user_id="user-1",
        balance=0,
        amount=25,
    )

    _propagate_node_failure(graph_stats, error)

    assert graph_stats.error == str(error)
    assert graph_stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE


def test_untyped_node_failure_is_not_promoted_to_graph_failure():
    graph_stats = GraphExecutionStats()

    _propagate_node_failure(
        graph_stats,
        RuntimeError("Third-party API reported insufficient balance"),
    )

    assert graph_stats.error is None
    assert graph_stats.failure_reason is None


def test_graph_start_credit_failure_records_structured_reason():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(organization_id="org-1"),
    )
    db_client = MagicMock()
    db_client.get_org_credits.return_value = 0
    processor = ExecutionProcessor()
    processor._cleanup_graph_execution = MagicMock()
    stats = GraphExecutionStats()

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.settings.config.enable_credit", True),
    ):
        _, status = processor._on_graph_execution(
            graph_exec=execution,
            cancel=threading.Event(),
            log_metadata=MagicMock(),
            execution_stats=stats,
            cluster_lock=MagicMock(),
        )

    assert status == ExecutionStatus.FAILED
    assert stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE
    assert stats.error == "The billed account has 0 credits but needs 1"


def test_graph_start_skips_balance_check_when_credits_are_disabled():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(organization_id="org-1"),
    )
    db_client = MagicMock()
    db_client.get_node_executions.return_value = []
    db_client.has_pending_reviews_for_graph_exec.return_value = False
    processor = ExecutionProcessor()
    processor._cleanup_graph_execution = MagicMock()
    processor.node_evaluation_loop = MagicMock()
    stats = GraphExecutionStats()
    completed_future = MagicMock()
    completed_future.result.return_value = None

    def complete_coroutine(coroutine, _loop):
        coroutine.close()
        return completed_future

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.settings.config.enable_credit", False),
        patch(
            "backend.executor.manager.asyncio.run_coroutine_threadsafe",
            side_effect=complete_coroutine,
        ),
    ):
        _, status = processor._on_graph_execution(
            graph_exec=execution,
            cancel=threading.Event(),
            log_metadata=MagicMock(),
            execution_stats=stats,
            cluster_lock=MagicMock(),
        )

    assert status == ExecutionStatus.COMPLETED
    db_client.get_org_credits.assert_not_called()
    db_client.get_credits.assert_not_called()


def test_nested_credit_failure_marker_makes_graph_fail_without_raised_graph_error():
    execution = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
    )
    db_client = MagicMock()
    db_client.get_credits.return_value = 100
    db_client.get_node_executions.return_value = []
    db_client.has_pending_reviews_for_graph_exec.return_value = False
    processor = ExecutionProcessor()
    processor._cleanup_graph_execution = MagicMock()
    processor.node_evaluation_loop = MagicMock()
    stats = GraphExecutionStats()
    _propagate_node_failure(
        stats,
        InsufficientBalanceError(
            message="Organization has 0 credits but needs 25",
            user_id="user-1",
            balance=0,
            amount=25,
        ),
    )
    completed_future = MagicMock()
    completed_future.result.return_value = None

    def complete_coroutine(coroutine, _loop):
        coroutine.close()
        return completed_future

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch(
            "backend.executor.manager.asyncio.run_coroutine_threadsafe",
            side_effect=complete_coroutine,
        ),
    ):
        _, status = processor._on_graph_execution(
            graph_exec=execution,
            cancel=threading.Event(),
            log_metadata=MagicMock(),
            execution_stats=stats,
            cluster_lock=MagicMock(),
        )

    assert status == ExecutionStatus.FAILED
    assert stats.failure_reason == ExecutionFailureReason.INSUFFICIENT_BALANCE
    assert stats.error == "Organization has 0 credits but needs 25"
