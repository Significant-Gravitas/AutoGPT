"""Tests for escalate_task's two routes up: user and manager.

The user route is exercised end-to-end in tasks_db_test; what matters here
is the tool-side contract — the target flows through to the RPC, a manager
escalation delivers the question into the delegator's origin session, and a
root task cannot pretend it has a manager.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.tasks.models import TaskEscalationTarget
from backend.util.exceptions import TaskDelegationRefusedError

from .escalate_task import EscalateTaskTool
from .models import ErrorResponse, TaskUpdateResponse


def _session(session_id: str = "worker-sess") -> MagicMock:
    sess = MagicMock()
    sess.session_id = session_id
    sess.user_id = "alice"
    return sess


def _task(
    task_id: str = "t-child",
    *,
    status: str = "WORKING",
    parent_task_id: str | None = "t-root",
    owner_name: str | None = "Bea",
) -> MagicMock:
    task = MagicMock()
    task.id = task_id
    task.title = "Child task"
    task.status = status
    task.parent_task_id = parent_task_id
    task.owner = MagicMock() if owner_name else None
    if task.owner:
        task.owner.name = owner_name
        task.owner.id = "expert-b"
        task.owner.avatar_url = None
        task.owner.role = "Analyst"
    return task


def _rpc(
    task: MagicMock,
    parent_origin_session: str | None = "manager-sess",
    *,
    parent_handoffs: int = 0,
):
    client = MagicMock()
    client.escalate_delegated_task = AsyncMock(return_value=task)
    detail = MagicMock()
    detail.task.origin_session_id = parent_origin_session
    detail.task.handoff_count = parent_handoffs
    client.get_delegated_task = AsyncMock(return_value=detail)
    return client


@pytest.mark.asyncio
async def test_user_escalation_passes_target_through():
    task = _task(status="WAITING_USER")
    client = _rpc(task)
    with patch(
        "backend.copilot.tools.escalate_task.get_database_manager_async_client",
        return_value=client,
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session(),
            task_id="t-child",
            question="Staging or prod?",
            options=["Staging", "Prod"],
        )

    assert isinstance(result, TaskUpdateResponse)
    assert result.action == "escalation"
    assert "waiting on the user" in result.message
    kwargs = client.escalate_delegated_task.await_args.kwargs
    assert kwargs["target"] == "user"
    client.get_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_escalation_notifies_the_delegator_session():
    task = _task()
    client = _rpc(task, parent_origin_session="manager-sess")
    queued = AsyncMock()
    with (
        patch(
            "backend.copilot.tools.escalate_task.get_database_manager_async_client",
            return_value=client,
        ),
        patch("backend.copilot.tools.escalate_task.queue_user_message", queued),
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session("worker-sess"),
            task_id="t-child",
            question="Which repo does this belong in?",
            target="manager",
        )

    assert isinstance(result, TaskUpdateResponse)
    assert "delegated it" in result.message
    assert client.escalate_delegated_task.await_args.kwargs["target"] == "manager"
    queued.assert_awaited_once()
    delivery = queued.await_args.kwargs
    assert delivery["session_id"] == "manager-sess"
    assert "Which repo does this belong in?" in delivery["message"]
    assert "not the user speaking" in delivery["message"]


@pytest.mark.asyncio
async def test_manager_escalation_skips_a_handed_off_parent_session():
    """After the parent changed hands its origin session belongs to the
    previous owner; the question stays on the timeline instead."""
    task = _task()
    client = _rpc(task, parent_origin_session="old-owner-sess", parent_handoffs=1)
    queued = AsyncMock()
    with (
        patch(
            "backend.copilot.tools.escalate_task.get_database_manager_async_client",
            return_value=client,
        ),
        patch("backend.copilot.tools.escalate_task.queue_user_message", queued),
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session("worker-sess"),
            task_id="t-child",
            question="Which repo does this belong in?",
            target="manager",
        )

    assert isinstance(result, TaskUpdateResponse)
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_manager_escalation_skips_notify_into_own_session():
    task = _task()
    client = _rpc(task, parent_origin_session="worker-sess")
    queued = AsyncMock()
    with (
        patch(
            "backend.copilot.tools.escalate_task.get_database_manager_async_client",
            return_value=client,
        ),
        patch("backend.copilot.tools.escalate_task.queue_user_message", queued),
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session("worker-sess"),
            task_id="t-child",
            question="Which repo?",
            target="manager",
        )

    assert isinstance(result, TaskUpdateResponse)
    queued.assert_not_awaited()


@pytest.mark.asyncio
async def test_root_task_refusal_reaches_the_model():
    client = MagicMock()
    client.escalate_delegated_task = AsyncMock(
        side_effect=TaskDelegationRefusedError(
            "This task has no delegator above it — nobody is managing it "
            'but the user. Escalate with target="user" instead.'
        )
    )
    with patch(
        "backend.copilot.tools.escalate_task.get_database_manager_async_client",
        return_value=client,
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session(),
            task_id="t-root",
            question="Who decides?",
            target="manager",
        )

    assert isinstance(result, ErrorResponse)
    assert "no delegator" in result.message


@pytest.mark.asyncio
async def test_invalid_target_is_rejected_before_any_write():
    client = MagicMock()
    client.escalate_delegated_task = AsyncMock()
    with patch(
        "backend.copilot.tools.escalate_task.get_database_manager_async_client",
        return_value=client,
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session(),
            task_id="t-child",
            question="Hm?",
            target=cast(TaskEscalationTarget, "ceo"),
        )

    assert isinstance(result, ErrorResponse)
    client.escalate_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_manager_notification_still_records_the_escalation():
    task = _task()
    client = _rpc(task)
    client.get_delegated_task = AsyncMock(side_effect=RuntimeError("rpc down"))
    with patch(
        "backend.copilot.tools.escalate_task.get_database_manager_async_client",
        return_value=client,
    ):
        result = await EscalateTaskTool()._execute(
            user_id="alice",
            session=_session(),
            task_id="t-child",
            question="Which repo?",
            target="manager",
        )

    assert isinstance(result, TaskUpdateResponse)
