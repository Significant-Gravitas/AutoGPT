"""Unit tests for the chat-side task-receipt opener — no DB required."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.task_spine import (
    _describe_run,
    fail_task,
    mark_task_working,
    open_task_for_run,
    settle_task_for_turn,
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


def _worker_session(
    task_id: str = "task-1", expert_id: str | None = "expert-1"
) -> ChatSession:
    session = _session(expert_id=expert_id)
    session.metadata.delegated_task_id = task_id
    return session


def _settle_client(status: str = "WORKING", owner_id: str | None = "expert-1"):
    client = _client()
    task = MagicMock(status=status)
    task.owner = MagicMock(id=owner_id) if owner_id else None
    client.get_delegated_task = AsyncMock(return_value=MagicMock(task=task))
    client.report_delegated_task = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_settle_reports_done_with_final_assistant_answer():
    client = _settle_client()
    session = _worker_session()
    fresh = _session(expert_id="expert-1")
    fresh.messages = [
        ChatMessage(role="user", content="build it"),
        ChatMessage(role="assistant", content="  Landing page\nshipped.  "),
    ]

    with (
        patch(f"{_MODULE}.get_database_manager_async_client", return_value=client),
        patch(f"{_MODULE}.get_chat_session", AsyncMock(return_value=fresh)),
    ):
        await settle_task_for_turn("user-1", session, error_message=None)

    args = client.report_delegated_task.call_args
    assert args.args == ("user-1", "task-1")
    assert args.kwargs["outcome_summary"] == "Landing page shipped."


@pytest.mark.asyncio
async def test_settle_failed_turn_closes_receipt_as_failed():
    client = _settle_client()

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn(
            "user-1", _worker_session(), error_message="LLM exploded"
        )

    kwargs = client.close_delegated_task.call_args.kwargs
    assert kwargs["succeeded"] is False
    assert "LLM exploded" in kwargs["outcome_summary"]
    client.report_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["WAITING_USER", "DONE", "FAILED", "CANCELLED"])
async def test_settle_leaves_non_working_receipts_alone(status: str):
    client = _settle_client(status=status)

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn("user-1", _worker_session(), error_message=None)

    client.report_delegated_task.assert_not_awaited()
    client.close_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_skips_receipt_handed_off_to_another_owner():
    client = _settle_client(owner_id="expert-2")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn("user-1", _worker_session(), error_message=None)

    client.report_delegated_task.assert_not_awaited()
    client.close_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_leaves_cancelled_turns_to_the_cancel_route():
    client = _settle_client()

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn(
            "user-1", _worker_session(), error_message="Operation cancelled"
        )

    client.get_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_without_receipt_makes_no_rpc_calls():
    client = _settle_client()
    session = _session(expert_id="expert-1")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn("user-1", session, error_message=None)

    client.get_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_settle_swallows_rpc_failure():
    client = _settle_client()
    client.get_delegated_task.side_effect = RuntimeError("rpc down")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await settle_task_for_turn("user-1", _worker_session(), error_message=None)


def test_describe_run_lists_inputs_and_truncates():
    assert _describe_run("Brief", {}) == "Run Brief."
    described = _describe_run("Brief", {"topic": "news"})
    assert "Run Brief with:" in described
    assert "- topic: 'news'" in described
    assert len(_describe_run("Brief", {"blob": "x" * 10_000})) <= 4_000
