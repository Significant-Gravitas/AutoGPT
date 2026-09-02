"""Unit tests for the chat-side task-receipt opener — no DB required."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.task_spine import (
    _describe_run,
    build_task_context,
    fail_task,
    mark_task_working,
    open_task_for_run,
    record_mid_task_instruction,
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


# ─── phase 3: mid-task instructions + task context ─────────────────────


def _amendment_client(task: object) -> MagicMock:
    client = MagicMock()
    client.append_task_amendment = AsyncMock()
    client.get_delegated_task = AsyncMock(
        return_value=MagicMock(task=task) if task is not None else None
    )
    return client


def _context_task(
    *,
    status: str = "WORKING",
    amendments: list | None = None,
):
    from backend.api.features.tasks.models import TaskAmendment

    task = MagicMock(status=status)
    task.id = "task-1"
    task.title = "Draft the weekly report"
    task.spec = "Cover revenue and churn."
    task.amendments = [
        TaskAmendment.model_validate(a) if isinstance(a, dict) else a
        for a in (amendments or [])
    ]
    return task


@pytest.mark.asyncio
async def test_mid_task_instruction_is_appended_as_a_user_note():
    client = _amendment_client(None)

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await record_mid_task_instruction(
            "user-1", _worker_session(), "Also include churn numbers.\n"
        )

    kwargs = client.append_task_amendment.call_args.kwargs
    args = client.append_task_amendment.call_args.args
    assert args == ("user-1", "task-1")
    assert kwargs["note"] == "Also include churn numbers."
    assert kwargs["by"] == "user"
    assert kwargs["kind"] == "note"


@pytest.mark.asyncio
async def test_mid_task_instruction_skips_sessions_without_a_task():
    client = _amendment_client(None)

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await record_mid_task_instruction("user-1", _session(), "hello")

    client.append_task_amendment.assert_not_awaited()


@pytest.mark.asyncio
async def test_mid_task_instruction_swallows_rpc_failure():
    client = _amendment_client(None)
    client.append_task_amendment.side_effect = RuntimeError("rpc down")

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        await record_mid_task_instruction("user-1", _worker_session(), "note")


@pytest.mark.asyncio
async def test_task_context_carries_spec_and_mid_task_instructions():
    from datetime import UTC as _UTC
    from datetime import datetime as _dt

    task = _context_task(
        amendments=[
            {
                "at": _dt(2026, 8, 30, 9, 30, tzinfo=_UTC),
                "by": "user",
                "note": "Also include churn numbers.",
                "kind": "note",
            },
            {
                "at": _dt(2026, 8, 30, 9, 0, tzinfo=_UTC),
                "by": "overseer",
                "note": "retried",
                "kind": "retry",
            },
        ]
    )
    client = _amendment_client(task)

    with patch(f"{_MODULE}.get_database_manager_async_client", return_value=client):
        context = await build_task_context("user-1", _worker_session())

    assert "task_id: task-1" in context
    assert "Cover revenue and churn." in context
    assert "added instructions mid-task" in context
    assert "Also include churn numbers." in context
    assert "retried" not in context


@pytest.mark.asyncio
async def test_task_context_is_empty_for_closed_or_missing_tasks():
    with patch(
        f"{_MODULE}.get_database_manager_async_client",
        return_value=_amendment_client(_context_task(status="DONE")),
    ):
        assert await build_task_context("user-1", _worker_session()) == ""

    with patch(
        f"{_MODULE}.get_database_manager_async_client",
        return_value=_amendment_client(None),
    ):
        assert await build_task_context("user-1", _worker_session()) == ""

    assert await build_task_context("user-1", _session()) == ""
