"""Tests for ListSchedulesTool and DeleteScheduleTool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.manage_schedules import (
    DeleteScheduleTool,
    ListSchedulesTool,
    ScheduleDeletedResponse,
    ScheduleListResponse,
)
from backend.copilot.tools.models import ErrorResponse
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from ._test_data import make_session

_USER = "test-user-schedules"
_SCHEDULES_PATH = "backend.copilot.tools.manage_schedules"


def _make_graph_info() -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        schedule_id="sched-1",
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        agent_name="My Schedule",
        cron="*/5 * * * *",
        input_data={"input": "data"},
        input_credentials={},
        id="sched-1",
        name="My Schedule",
        next_run_time="2026-04-13T12:00:00",
        timezone="UTC",
    )


def _make_copilot_info() -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        schedule_id="cop-1",
        user_id=_USER,
        session_id="session-xyz",
        message="check CI on PR #999",
        run_at=datetime(2026, 5, 22, 19, 0, tzinfo=timezone.utc),
        id="cop-1",
        name="copilot turn (session session-)",
        next_run_time="2026-05-22T19:00:00+00:00",
        timezone="UTC",
    )


@pytest.fixture
def list_tool():
    return ListSchedulesTool()


@pytest.fixture
def delete_tool():
    return DeleteScheduleTool()


@pytest.fixture
def session():
    return make_session(_USER)


# ── ListSchedulesTool ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_schedules_no_auth(list_tool, session):
    result = await list_tool._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_list_schedules_returns_graph_kind(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_graph_info()])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    summary = result.schedules[0]
    assert summary.schedule_id == "sched-1"
    assert summary.kind == "graph"
    assert summary.cron == "*/5 * * * *"
    assert summary.graph_id == "graph-1"
    assert summary.graph_version == 1
    assert summary.session_id is None
    assert summary.message is None


@pytest.mark.asyncio
async def test_list_schedules_returns_copilot_turn_kind(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[_make_copilot_info()])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    summary = result.schedules[0]
    assert summary.schedule_id == "cop-1"
    assert summary.kind == "copilot_turn"
    assert summary.session_id == "session-xyz"
    assert summary.message == "check CI on PR #999"
    assert summary.run_at == "2026-05-22T19:00:00+00:00"
    assert summary.graph_id is None
    assert summary.graph_version is None


@pytest.mark.asyncio
async def test_list_schedules_returns_mixed_kinds(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(
        return_value=[_make_graph_info(), _make_copilot_info()]
    )

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 2
    assert {s.kind for s in result.schedules} == {"graph", "copilot_turn"}


@pytest.mark.asyncio
async def test_list_schedules_empty(list_tool, session):
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 0
    assert "No schedules" in result.message


@pytest.mark.asyncio
async def test_list_schedules_by_library_agent(list_tool, session):
    mock_agent = MagicMock(graph_id="graph-42")
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[])

    with (
        patch(
            f"{_SCHEDULES_PATH}.get_library_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ),
        patch(
            f"{_SCHEDULES_PATH}.get_scheduler_client",
            return_value=mock_client,
        ),
    ):
        result = await list_tool._execute(
            user_id=_USER,
            session=session,
            library_agent_id="lib-agent-1",
        )

    assert isinstance(result, ScheduleListResponse)
    mock_client.get_execution_schedules.assert_called_once_with(
        graph_id="graph-42", user_id=_USER
    )


@pytest.mark.asyncio
async def test_list_schedules_library_agent_not_found(list_tool, session):
    from backend.util.exceptions import NotFoundError

    with patch(
        f"{_SCHEDULES_PATH}.get_library_agent",
        new_callable=AsyncMock,
        side_effect=NotFoundError("not found"),
    ):
        result = await list_tool._execute(
            user_id=_USER,
            session=session,
            library_agent_id="missing",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "library_agent_not_found"


# ── DeleteScheduleTool ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_schedule_no_auth(delete_tool, session):
    result = await delete_tool._execute(user_id=None, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_delete_schedule_missing_id(delete_tool, session):
    result = await delete_tool._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_schedule_id"


@pytest.mark.asyncio
async def test_delete_schedule_success(delete_tool, session):
    mock_client = AsyncMock()
    mock_client.delete_schedule = AsyncMock()

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="sched-1",
        )

    assert isinstance(result, ScheduleDeletedResponse)
    assert result.schedule_id == "sched-1"
    mock_client.delete_schedule.assert_called_once_with(
        schedule_id="sched-1", user_id=_USER
    )


@pytest.mark.asyncio
async def test_delete_schedule_not_found(delete_tool, session):
    from backend.util.exceptions import NotFoundError

    mock_client = AsyncMock()
    mock_client.delete_schedule = AsyncMock(side_effect=NotFoundError("Job not found"))

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="missing",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "schedule_not_found"


@pytest.mark.asyncio
async def test_delete_schedule_not_authorized(delete_tool, session):
    from backend.util.exceptions import NotAuthorizedError

    mock_client = AsyncMock()
    mock_client.delete_schedule = AsyncMock(
        side_effect=NotAuthorizedError("wrong user")
    )

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await delete_tool._execute(
            user_id=_USER,
            session=session,
            schedule_id="sched-1",
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "not_authorized"
