from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from backend.data.execution import ExecutionStatus
from backend.data.model import GraphExecutionStats
from backend.executor import manager
from backend.executor.manager import (
    _expert_run_completed_event,
    _observe_funnel_emission,
    _persist_graph_completion_and_emit_funnel,
)


def _graph_exec(expert_id="e-1", dry_run=False, parent=None, user_id="u-1"):
    return SimpleNamespace(
        user_id=user_id,
        graph_exec_id="run-1",
        execution_context=SimpleNamespace(
            expert_id=expert_id,
            dry_run=dry_run,
            parent_execution_id=parent,
        ),
    )


def test_emits_for_terminal_expert_run():
    event = _expert_run_completed_event(_graph_exec(), ExecutionStatus.COMPLETED)
    assert event == {
        "expert_id": "e-1",
        "status": ExecutionStatus.COMPLETED.value,
        "graph_exec_id": "run-1",
    }


def test_emits_for_failed_expert_run():
    event = _expert_run_completed_event(_graph_exec(), ExecutionStatus.FAILED)
    assert event == {
        "expert_id": "e-1",
        "status": ExecutionStatus.FAILED.value,
        "graph_exec_id": "run-1",
    }


def test_skips_non_expert_run():
    assert (
        _expert_run_completed_event(
            _graph_exec(expert_id=None), ExecutionStatus.COMPLETED
        )
        is None
    )


def test_skips_dry_run():
    assert (
        _expert_run_completed_event(
            _graph_exec(dry_run=True), ExecutionStatus.COMPLETED
        )
        is None
    )


def test_skips_subgraph_run():
    assert (
        _expert_run_completed_event(
            _graph_exec(parent="parent-1"), ExecutionStatus.COMPLETED
        )
        is None
    )


def test_skips_non_terminal_status():
    assert _expert_run_completed_event(_graph_exec(), ExecutionStatus.RUNNING) is None


def test_persists_before_scheduling_funnel_emission():
    calls: list[str] = []
    rpc_client = MagicMock(emit_funnel_event=AsyncMock())
    submitted: Future = Future()
    submitted.set_result(None)

    def submit(coroutine, event_loop):
        calls.append("emit")
        coroutine.close()
        assert event_loop is loop
        return submitted

    loop = MagicMock()
    graph_exec = _graph_exec()
    stats = GraphExecutionStats()
    with (
        patch.object(
            manager,
            "update_graph_execution_state",
            side_effect=lambda **kwargs: calls.append("persist"),
        ) as persist,
        patch.object(manager, "get_db_async_client", return_value=rpc_client),
        patch.object(manager.asyncio, "run_coroutine_threadsafe", side_effect=submit),
    ):
        _persist_graph_completion_and_emit_funnel(
            MagicMock(), graph_exec, ExecutionStatus.COMPLETED, stats, loop
        )

    assert calls == ["persist", "emit"]
    persist.assert_called_once()
    rpc_client.emit_funnel_event.assert_called_once_with(
        "u-1",
        "expert_run_completed",
        {
            "expert_id": "e-1",
            "status": ExecutionStatus.COMPLETED.value,
            "graph_exec_id": "run-1",
        },
        "expert_run_completed:run-1",
    )


def test_observe_funnel_emission_logs_background_failure():
    future: Future = Future()
    future.set_exception(RuntimeError("rpc unavailable"))

    with patch.object(manager.logger, "exception") as log_exception:
        _observe_funnel_emission(future)

    log_exception.assert_called_once_with(
        "Expert run funnel emission failed after submission"
    )
