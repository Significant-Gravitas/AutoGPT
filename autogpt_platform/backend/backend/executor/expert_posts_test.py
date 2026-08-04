"""Unit tests for expert run-result thread posts — no DB required."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
)
from backend.data.model import GraphExecutionStats
from backend.executor.expert_posts import (
    build_expert_run_message,
    handle_expert_run_post,
)

_MODULE = "backend.executor.expert_posts"


def _entry(expert_id: str | None = None, dry_run: bool = False) -> GraphExecutionEntry:
    return GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id="exec-1",
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(expert_id=expert_id, dry_run=dry_run),
    )


def _redis_allowing_posts() -> MagicMock:
    redis = MagicMock()
    redis.incr.return_value = 1
    return redis


def test_build_message_success_prefers_activity_summary():
    message = build_expert_run_message(
        agent_name="Morning Brief",
        succeeded=True,
        summary="Sent today's brief covering 3 meetings.",
        library_agent_id="lib-1",
    )
    assert "Morning Brief" in message
    assert "Sent today's brief covering 3 meetings." in message
    assert "/library/agents/lib-1" in message


def test_build_message_failure_states_reason():
    message = build_expert_run_message(
        agent_name="Morning Brief",
        succeeded=False,
        error="missing Gmail credentials",
    )
    assert "didn't finish" in message
    assert "missing Gmail credentials" in message


def test_post_skips_runs_without_expert():
    db_client = MagicMock()
    handle_expert_run_post(
        db_client, _entry(), ExecutionStatus.COMPLETED, GraphExecutionStats()
    )
    db_client.append_expert_run_message.assert_not_called()


def test_post_skips_dry_runs_and_non_terminal_statuses():
    db_client = MagicMock()
    handle_expert_run_post(
        db_client,
        _entry(expert_id="expert-1", dry_run=True),
        ExecutionStatus.COMPLETED,
        GraphExecutionStats(),
    )
    handle_expert_run_post(
        db_client,
        _entry(expert_id="expert-1"),
        ExecutionStatus.RUNNING,
        GraphExecutionStats(),
    )
    db_client.append_expert_run_message.assert_not_called()


def test_post_uses_deterministic_message_id_for_retries():
    db_client = MagicMock()
    db_client.get_graph_metadata.return_value = SimpleNamespace(name="Morning Brief")
    db_client.get_library_agent_id_by_graph_id.return_value = "lib-1"
    with patch(f"{_MODULE}.get_redis", return_value=_redis_allowing_posts()):
        handle_expert_run_post(
            db_client,
            _entry(expert_id="expert-1"),
            ExecutionStatus.COMPLETED,
            GraphExecutionStats(),
        )
        first_call = db_client.append_expert_run_message.call_args.kwargs
        assert "Morning Brief" in first_call["content"]
        first_id = first_call["message_id"]
        handle_expert_run_post(
            db_client,
            _entry(expert_id="expert-1"),
            ExecutionStatus.COMPLETED,
            GraphExecutionStats(),
        )
        second_id = db_client.append_expert_run_message.call_args.kwargs["message_id"]
    assert first_id == second_id


def test_post_respects_daily_cap():
    db_client = MagicMock()
    capped_redis = MagicMock()
    capped_redis.incr.return_value = 11
    with patch(f"{_MODULE}.get_redis", return_value=capped_redis):
        handle_expert_run_post(
            db_client,
            _entry(expert_id="expert-1"),
            ExecutionStatus.COMPLETED,
            GraphExecutionStats(),
        )
    db_client.append_expert_run_message.assert_not_called()


def test_post_never_raises_on_client_failure():
    db_client = MagicMock()
    db_client.get_graph_metadata.side_effect = RuntimeError("rpc down")
    with patch(f"{_MODULE}.get_redis", return_value=_redis_allowing_posts()):
        handle_expert_run_post(
            db_client,
            _entry(expert_id="expert-1"),
            ExecutionStatus.FAILED,
            GraphExecutionStats(),
        )
