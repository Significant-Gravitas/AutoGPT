"""Tests for ScheduleFollowupTool."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.schedule_followup import (
    ScheduleCreatedResponse,
    ScheduleFollowupTool,
)
from backend.executor.scheduler import CopilotTurnJobInfo

from ._test_data import make_session

_USER = "test-user-followup"
_TOOL_PATH = "backend.copilot.tools.schedule_followup"


def _info(*, cron=None, run_at=None) -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo(
        schedule_id="cop-1",
        user_id=_USER,
        session_id="placeholder",
        message="check CI",
        cron=cron,
        run_at=run_at,
        id="cop-1",
        name="copilot turn",
        next_run_time="2026-05-22T19:00:00+00:00",
        timezone="UTC",
    )


@pytest.fixture
def tool():
    return ScheduleFollowupTool()


@pytest.fixture
def session():
    return make_session(_USER, guide_read=True)


@pytest.mark.asyncio
async def test_no_auth(tool, session):
    result = await tool._execute(user_id=None, session=session, message="x")
    assert isinstance(result, ErrorResponse)
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_missing_message(tool, session):
    result = await tool._execute(user_id=_USER, session=session)
    assert isinstance(result, ErrorResponse)
    assert result.error == "missing_message"


@pytest.mark.asyncio
async def test_neither_cron_nor_delay(tool, session):
    result = await tool._execute(user_id=_USER, session=session, message="check CI")
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_trigger"


@pytest.mark.asyncio
async def test_both_cron_and_delay(tool, session):
    result = await tool._execute(
        user_id=_USER,
        session=session,
        message="check CI",
        delay_seconds=60,
        cron="* * * * *",
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "invalid_trigger"


@pytest.mark.asyncio
async def test_delay_too_short(tool, session):
    result = await tool._execute(
        user_id=_USER,
        session=session,
        message="check CI",
        delay_seconds=30,
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "delay_too_short"


@pytest.mark.asyncio
async def test_one_shot_delay_schedules(tool, session):
    mock_user_db = MagicMock()
    mock_user = MagicMock(timezone="America/New_York")
    mock_user_db().get_user_by_id = AsyncMock(return_value=mock_user)

    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule = AsyncMock(
        return_value=_info(run_at=datetime(2026, 5, 22, 19, 0, tzinfo=timezone.utc))
    )

    with (
        patch(f"{_TOOL_PATH}.user_db", return_value=mock_user_db()),
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="check CI",
            delay_seconds=1200,
        )

    assert isinstance(result, ScheduleCreatedResponse)
    assert result.schedule_id == "cop-1"
    assert result.is_recurring is False
    call_kwargs = mock_client.add_copilot_turn_schedule.call_args.kwargs
    assert call_kwargs["session_id"] == session.session_id
    assert call_kwargs["message"] == "check CI"
    assert call_kwargs["cron"] is None
    assert call_kwargs["run_at"] is not None
    assert call_kwargs["user_timezone"] == "America/New_York"


@pytest.mark.asyncio
async def test_cron_schedules_recurring(tool, session):
    mock_user_db = MagicMock()
    mock_user = MagicMock(timezone=None)
    mock_user_db().get_user_by_id = AsyncMock(return_value=mock_user)

    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule = AsyncMock(
        return_value=_info(cron="0 9 * * 1")
    )

    with (
        patch(f"{_TOOL_PATH}.user_db", return_value=mock_user_db()),
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="weekly check",
            cron="0 9 * * 1",
        )

    assert isinstance(result, ScheduleCreatedResponse)
    assert result.is_recurring is True
    call_kwargs = mock_client.add_copilot_turn_schedule.call_args.kwargs
    assert call_kwargs["cron"] == "0 9 * * 1"
    assert call_kwargs["run_at"] is None
    assert call_kwargs["user_timezone"] == "UTC"
