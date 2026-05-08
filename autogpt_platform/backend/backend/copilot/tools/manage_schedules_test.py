"""Tests for ListSchedulesTool and DeleteScheduleTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.manage_schedules import (
    DeleteScheduleTool,
    ListSchedulesTool,
    ScheduleDeletedResponse,
    ScheduleListResponse,
)
from backend.copilot.tools.models import ErrorResponse

from ._test_data import make_session

_USER = "test-user-schedules"
_SCHEDULES_PATH = "backend.copilot.tools.manage_schedules"


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
async def test_list_schedules_returns_results(list_tool, session):
    mock_job = MagicMock(spec=[])
    mock_job.id = "sched-1"
    mock_job.name = "My Schedule"
    mock_job.cron = "*/5 * * * *"
    mock_job.timezone = "UTC"
    mock_job.next_run_time = "2026-04-13T12:00:00"
    mock_job.graph_id = "graph-1"
    mock_job.graph_version = 1
    mock_client = AsyncMock()
    mock_client.get_execution_schedules = AsyncMock(return_value=[mock_job])

    with patch(
        f"{_SCHEDULES_PATH}.get_scheduler_client",
        return_value=mock_client,
    ):
        result = await list_tool._execute(user_id=_USER, session=session)

    assert isinstance(result, ScheduleListResponse)
    assert len(result.schedules) == 1
    assert result.schedules[0].schedule_id == "sched-1"
    assert result.schedules[0].cron == "*/5 * * * *"


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
