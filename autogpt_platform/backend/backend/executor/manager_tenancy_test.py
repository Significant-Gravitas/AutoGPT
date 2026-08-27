import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.db_accessors import LiveResourceAccessRevoked
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.model import GraphExecutionStats
from backend.executor.manager import (
    ExecutionProcessor,
    _generate_execution_activity,
    _handle_completion_communications,
    _has_live_execution_access,
    _moderate_execution_inputs,
    _moderate_execution_outputs,
)


def _graph_execution():
    return GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(
            organization_id="org-1",
            team_id="team-1",
        ),
    )


def _execution_meta(**overrides):
    values = {
        "id": "exec-1",
        "user_id": "user-1",
        "graph_id": "graph-1",
        "graph_version": 1,
        "organization_id": "org-1",
        "team_id": "team-1",
        "status": ExecutionStatus.QUEUED,
        "stats": None,
    }
    values.update(overrides)
    return MagicMock(**values)


def test_live_execution_access_uses_persisted_scope():
    db_client = MagicMock()
    db_client.has_live_resource_access.return_value = False
    graph_exec = _graph_execution()

    assert _has_live_execution_access(db_client, graph_exec) is False
    db_client.has_live_resource_access.assert_called_once_with(
        "user-1", "org-1", "team-1", "execute"
    )


def test_revoked_execution_is_terminated_before_worker_start():
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = _execution_meta()
    db_client.has_live_resource_access.return_value = False
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock()

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.update_graph_execution_state") as update,
    ):
        processor.on_graph_execution(_graph_execution(), threading.Event(), MagicMock())

    processor._on_graph_execution.assert_not_called()
    update.assert_called_once_with(
        db_client=db_client,
        graph_exec_id="exec-1",
        status=ExecutionStatus.TERMINATED,
    )


def test_persisted_scope_replaces_stale_queue_scope_before_authorization():
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = _execution_meta()
    db_client.has_live_resource_access.return_value = False
    execution = _graph_execution()
    execution.execution_context.organization_id = None
    execution.execution_context.team_id = "wrong-team"
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock()

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.update_graph_execution_state"),
    ):
        processor.on_graph_execution(execution, threading.Event(), MagicMock())

    db_client.has_live_resource_access.assert_called_once_with(
        "user-1", "org-1", "team-1", "execute"
    )
    processor._on_graph_execution.assert_not_called()


def test_mismatched_queue_identity_is_rejected_before_authorization():
    db_client = MagicMock()
    db_client.get_graph_execution_meta.return_value = _execution_meta(
        graph_id="persisted-graph"
    )
    processor = ExecutionProcessor()
    processor._on_graph_execution = MagicMock()

    with patch("backend.executor.manager.get_db_client", return_value=db_client):
        processor.on_graph_execution(_graph_execution(), threading.Event(), MagicMock())

    db_client.has_live_resource_access.assert_not_called()
    db_client.update_graph_execution_start_time.assert_not_called()
    processor._on_graph_execution.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [_moderate_execution_inputs, _moderate_execution_outputs],
)
async def test_denied_moderation_lease_never_reaches_automod(operation):
    db_client = MagicMock()
    db_client.acquire_live_resource_lease = AsyncMock(return_value=None)

    with (
        patch("backend.executor.manager.get_db_async_client", return_value=db_client),
        patch(
            "backend.executor.manager.automod_manager.moderate_graph_execution_inputs",
            new_callable=AsyncMock,
        ) as moderate_inputs,
        patch(
            "backend.executor.manager.automod_manager.moderate_graph_execution_outputs",
            new_callable=AsyncMock,
        ) as moderate_outputs,
    ):
        with pytest.raises(LiveResourceAccessRevoked):
            await operation(_graph_execution())

    moderate_inputs.assert_not_called()
    moderate_outputs.assert_not_called()


@pytest.mark.asyncio
async def test_denied_activity_lease_never_fetches_or_sends_execution_content():
    db_client = MagicMock()
    db_client.acquire_live_resource_lease = AsyncMock(return_value=None)

    with (
        patch("backend.executor.manager.get_db_async_client", return_value=db_client),
        patch(
            "backend.executor.manager.generate_activity_status_for_execution",
            new_callable=AsyncMock,
        ) as generate,
    ):
        with pytest.raises(LiveResourceAccessRevoked):
            await _generate_execution_activity(
                _graph_execution(),
                GraphExecutionStats(),
                ExecutionStatus.COMPLETED,
            )

    generate.assert_not_called()


def test_denied_completion_lease_never_reads_outputs_or_queues_notification():
    db_client = MagicMock()
    db_client.acquire_live_resource_lease.return_value = None

    with patch("backend.executor.manager.expert_posts.handle_expert_run_post") as post:
        result = _handle_completion_communications(
            db_client,
            _graph_execution(),
            ExecutionStatus.COMPLETED,
            GraphExecutionStats(),
        )

    assert result is False
    post.assert_not_called()


def _queued_node_execution() -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(
            organization_id="org-1",
            team_id="team-1",
        ),
    )


def _run_dispatch(processor, db_client, graph_exec):
    completed = MagicMock()
    completed.result.return_value = None
    processor._cleanup_graph_execution = MagicMock()

    def complete_coroutine(coroutine, _loop):
        coroutine.close()
        return completed

    with (
        patch("backend.executor.manager.get_db_client", return_value=db_client),
        patch("backend.executor.manager.settings.config.enable_credit", False),
        patch("backend.executor.manager.increment_execution_count", return_value=1),
        patch("backend.executor.manager.billing.handle_low_balance"),
        patch(
            "backend.executor.manager.asyncio.run_coroutine_threadsafe",
            side_effect=complete_coroutine,
        ),
        patch("backend.executor.manager.update_node_execution_status"),
        patch("backend.executor.manager.clean_exec_files"),
    ):
        return processor._on_graph_execution(
            graph_exec=graph_exec,
            cancel=threading.Event(),
            log_metadata=MagicMock(),
            execution_stats=GraphExecutionStats(),
            cluster_lock=MagicMock(),
        )[1]


def test_denied_node_dispatch_lease_never_debits_org():
    db_client = MagicMock()
    queued = MagicMock()
    queued.to_node_execution_entry.return_value = _queued_node_execution()
    db_client.get_node_executions.return_value = [queued]
    db_client.acquire_live_resource_lease.return_value = None
    processor = ExecutionProcessor()
    processor.node_evaluation_loop = MagicMock()
    processor.node_execution_loop = MagicMock()

    with patch("backend.executor.manager.billing.charge_usage") as charge:
        status = _run_dispatch(processor, db_client, _graph_execution())

    assert status == ExecutionStatus.TERMINATED
    charge.assert_not_called()
    db_client.acquire_live_resource_lease.assert_called_once_with(
        "user-1", "org-1", "team-1", "execute"
    )
    db_client.release_live_resource_lease.assert_not_called()


def test_preflight_billing_lease_is_transferred_to_node_execution():
    db_client = MagicMock()
    queued = MagicMock()
    queued.to_node_execution_entry.return_value = _queued_node_execution()
    db_client.get_node_executions.return_value = [queued]
    db_client.acquire_live_resource_lease.return_value = "lease-1"
    db_client.has_pending_reviews_for_graph_exec.return_value = False
    processor = ExecutionProcessor()
    processor.node_evaluation_loop = MagicMock()
    processor.node_execution_loop = MagicMock()
    processor.on_node_execution = MagicMock()

    async def node_execution(**_kwargs):
        return None

    processor.on_node_execution.side_effect = node_execution

    with patch(
        "backend.executor.manager.billing.charge_usage",
        return_value=(5, 95, 5),
    ) as charge:
        status = _run_dispatch(processor, db_client, _graph_execution())

    assert status == ExecutionStatus.COMPLETED
    charge.assert_called_once()
    assert processor.on_node_execution.call_args.kwargs["live_lease_id"] == "lease-1"
    db_client.release_live_resource_lease.assert_not_called()
