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
async def test_one_shot_delay_omitted_session_id_passes_new_chat_sentinel(
    tool, session
):
    """Omitting session_id is the sentinel for "create a fresh chat at fire
    time" — the scheduler must receive ``session_id=None`` (not the current
    session's id) so the executor's create-session branch fires.
    """
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
    assert call_kwargs["session_id"] is None  # sentinel for "create fresh chat"
    assert call_kwargs["message"] == "check CI"
    assert call_kwargs["cron"] is None
    assert call_kwargs["run_at"] is not None
    assert call_kwargs["user_timezone"] == "America/New_York"
    assert "fresh chat" in result.message  # user-facing copy reflects sentinel


@pytest.mark.asyncio
async def test_explicit_null_session_id_passes_new_chat_sentinel(tool, session):
    """Explicit ``session_id=None`` is equivalent to omitting it — both are
    the sentinel for "create a fresh chat at fire time"."""
    mock_user_db = MagicMock()
    mock_user_db().get_user_by_id = AsyncMock(return_value=MagicMock(timezone="UTC"))
    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule = AsyncMock(return_value=_info())

    with (
        patch(f"{_TOOL_PATH}.user_db", return_value=mock_user_db()),
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="daily brief",
            delay_seconds=600,
            session_id=None,
        )

    assert isinstance(result, ScheduleCreatedResponse)
    assert mock_client.add_copilot_turn_schedule.call_args.kwargs["session_id"] is None


@pytest.mark.asyncio
async def test_session_id_override_targets_a_different_owned_session(tool, session):
    """Passing session_id targets that session instead of the current one."""
    other_session_id = "session-other-owned-by-same-user"
    mock_user_db = MagicMock()
    mock_user = MagicMock(timezone="UTC")
    mock_user_db().get_user_by_id = AsyncMock(return_value=mock_user)

    mock_get_session = AsyncMock(return_value=MagicMock())  # session exists + owned
    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule = AsyncMock(return_value=_info())

    with (
        patch(f"{_TOOL_PATH}.user_db", return_value=mock_user_db()),
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
        patch(f"{_TOOL_PATH}.get_chat_session", new=mock_get_session),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="check on the long job",
            delay_seconds=600,
            session_id=other_session_id,
        )

    assert isinstance(result, ScheduleCreatedResponse)
    mock_get_session.assert_awaited_once_with(other_session_id, _USER)
    call_kwargs = mock_client.add_copilot_turn_schedule.call_args.kwargs
    assert call_kwargs["session_id"] == other_session_id


@pytest.mark.asyncio
async def test_session_id_override_rejects_session_not_owned(tool, session):
    """get_chat_session returns None when not found OR not owned — both 'session_not_found'."""
    other_session_id = "session-someone-elses"
    mock_get_session = AsyncMock(return_value=None)
    mock_client = AsyncMock()

    with (
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
        patch(f"{_TOOL_PATH}.get_chat_session", new=mock_get_session),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="x",
            delay_seconds=600,
            session_id=other_session_id,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "session_not_found"
    mock_client.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_id_override_to_current_session_skips_lookup(tool, session):
    """Passing session_id equal to the current session avoids the get_chat_session round-trip."""
    mock_user_db = MagicMock()
    mock_user_db().get_user_by_id = AsyncMock(return_value=MagicMock(timezone=None))
    mock_get_session = AsyncMock()  # should NOT be called
    mock_client = AsyncMock()
    mock_client.add_copilot_turn_schedule = AsyncMock(return_value=_info())

    with (
        patch(f"{_TOOL_PATH}.user_db", return_value=mock_user_db()),
        patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=mock_client),
        patch(f"{_TOOL_PATH}.get_chat_session", new=mock_get_session),
    ):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="x",
            delay_seconds=600,
            session_id=session.session_id,
        )

    assert isinstance(result, ScheduleCreatedResponse)
    mock_get_session.assert_not_awaited()


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


# ---------------------------------------------------------------------------
# COPILOT_SCHEDULED_FOLLOWUPS LaunchDarkly kill-switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_feature_disabled_when_flag_off(tool):
    """When the per-user LD flag is off, ``_execute`` short-circuits to a
    structured ``feature_disabled`` error WITHOUT touching the scheduler
    client.  This is the rollback story — flipping the flag has to make
    new schedules impossible without a redeploy."""
    session = make_session(_USER)

    scheduler_client = MagicMock()
    scheduler_client.add_copilot_turn_schedule = AsyncMock(
        side_effect=AssertionError("scheduler must not be touched when flag off")
    )

    with patch(
        f"{_TOOL_PATH}.is_followups_feature_enabled",
        new=AsyncMock(return_value=False),
    ), patch(f"{_TOOL_PATH}.get_scheduler_client", return_value=scheduler_client):
        result = await tool._execute(
            user_id=_USER,
            session=session,
            message="ping me later",
            delay_seconds=120,
        )

    assert isinstance(result, ErrorResponse)
    assert result.error == "feature_disabled"
    scheduler_client.add_copilot_turn_schedule.assert_not_awaited()
