from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.data.execution import ExecutionStatus
from backend.executor import manager
from backend.executor.manager import (
    _emit_expert_run_completed,
    _expert_run_completed_event,
    _observe_funnel_emission,
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


def test_emission_submits_to_the_given_loop_with_a_dedup_key():
    rpc_client = MagicMock(emit_funnel_event=MagicMock(return_value="coro"))
    submitted = MagicMock()
    loop = MagicMock()
    graph_exec = _graph_exec()
    run_event = _expert_run_completed_event(graph_exec, ExecutionStatus.COMPLETED)
    assert run_event is not None

    with (
        patch.object(manager, "get_db_async_client", return_value=rpc_client),
        patch.object(
            manager.asyncio, "run_coroutine_threadsafe", return_value=submitted
        ) as submit,
    ):
        _emit_expert_run_completed(graph_exec, run_event, loop)

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
    submit.assert_called_once_with("coro", loop)
    submitted.add_done_callback.assert_called_once_with(
        manager._observe_funnel_emission
    )


def test_emission_swallows_a_failed_submission():
    """A dead loop must not sink the run whose completion was just persisted."""
    with (
        patch.object(manager, "get_db_async_client", return_value=MagicMock()),
        patch.object(
            manager.asyncio,
            "run_coroutine_threadsafe",
            side_effect=RuntimeError("loop is closed"),
        ),
        patch.object(manager.logger, "exception") as log_exception,
    ):
        _emit_expert_run_completed(_graph_exec(), {"expert_id": "e-1"}, MagicMock())

    log_exception.assert_called_once()


def test_observe_funnel_emission_logs_background_failure():
    future: Future = Future()
    future.set_exception(RuntimeError("rpc unavailable"))

    with patch.object(manager.logger, "exception") as log_exception:
        _observe_funnel_emission(future)

    log_exception.assert_called_once_with(
        "Expert run funnel emission failed after submission"
    )
