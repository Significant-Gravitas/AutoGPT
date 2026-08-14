from types import SimpleNamespace

from backend.data.execution import ExecutionStatus
from backend.executor.manager import _expert_run_completed_event


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
        "status": str(ExecutionStatus.COMPLETED),
        "graph_exec_id": "run-1",
    }


def test_emits_for_failed_expert_run():
    event = _expert_run_completed_event(_graph_exec(), ExecutionStatus.FAILED)
    assert event == {
        "expert_id": "e-1",
        "status": str(ExecutionStatus.FAILED),
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
