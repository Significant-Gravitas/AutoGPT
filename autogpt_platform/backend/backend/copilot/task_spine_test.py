"""Unit tests for the chat-side task-receipt opener — no DB required."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession, ChatSessionMetadata
from backend.copilot.task_spine import (
    _describe_run,
    fail_task,
    mark_task_working,
    open_task_for_run,
)

_MODULE = "backend.copilot.task_spine"


def _session(expert_id: str | None = None) -> ChatSession:
    return ChatSession(
        session_id=str(uuid.uuid4()),
        user_id="user-1",
        messages=[],
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        successful_agent_runs={},
        successful_agent_schedules={},
        expert_id=expert_id,
        metadata=ChatSessionMetadata(origin="interactive"),
    )


def _client(create_result: object = None) -> MagicMock:
    client = MagicMock()
    client.create_delegated_task = AsyncMock(return_value=create_result)
    client.mark_delegated_task_working = AsyncMock()
    client.close_delegated_task = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_open_task_stamps_expert_and_origin_session():
    task = MagicMock(id="task-1")
    client = _client(task)
    session = _session(expert_id="expert-1")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        task_id = await open_task_for_run(
            "user-1", session, agent_name="Morning Brief", inputs={"topic": "news"}
        )

    assert task_id == "task-1"
    kwargs = client.create_delegated_task.call_args.kwargs
    assert kwargs["owner_id"] == "expert-1"
    assert kwargs["origin_session_id"] == session.session_id
    assert kwargs["created_by_type"] == "USER"
    assert kwargs["title"] == "Morning Brief"
    assert "topic" in kwargs["spec"]


@pytest.mark.asyncio
async def test_open_task_for_autopilot_session_has_no_owner():
    client = _client(MagicMock(id="task-1"))

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await open_task_for_run(
            "user-1", _session(), agent_name="Morning Brief", inputs={}
        )

    assert client.create_delegated_task.call_args.kwargs["owner_id"] is None


@pytest.mark.asyncio
async def test_open_task_failure_returns_none_instead_of_raising():
    client = _client()
    client.create_delegated_task.side_effect = RuntimeError("rpc down")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        task_id = await open_task_for_run(
            "user-1", _session(), agent_name="Morning Brief", inputs={}
        )

    assert task_id is None


@pytest.mark.asyncio
async def test_mark_working_swallows_rpc_failure():
    client = _client()
    client.mark_delegated_task_working.side_effect = RuntimeError("rpc down")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await mark_task_working("user-1", "task-1")


@pytest.mark.asyncio
async def test_fail_task_closes_receipt_as_failed():
    client = _client()

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await fail_task("user-1", "task-1", "The run could not be started.")

    kwargs = client.close_delegated_task.call_args.kwargs
    assert kwargs["succeeded"] is False
    assert kwargs["outcome_summary"] == "The run could not be started."


@pytest.mark.asyncio
async def test_fail_task_swallows_rpc_failure():
    client = _client()
    client.close_delegated_task.side_effect = RuntimeError("rpc down")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await fail_task("user-1", "task-1", "reason")


def test_describe_run_lists_inputs_and_truncates():
    assert _describe_run("Brief", {}) == "Run Brief."
    described = _describe_run("Brief", {"topic": "news"})
    assert "Run Brief with:" in described
    assert "- topic: 'news'" in described
    assert len(_describe_run("Brief", {"blob": "x" * 10_000})) <= 4_000
