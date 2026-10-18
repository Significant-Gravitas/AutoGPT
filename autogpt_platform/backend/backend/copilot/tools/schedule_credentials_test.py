"""Choosing, when a schedule is made, the accounts its turns run on (SECRT-2804)."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.api.features.experts.models import ExpertRoutine
from backend.copilot.credential_selection import CredentialPin
from backend.copilot.model import ChatSession
from backend.copilot.tools.models import SetupRequirementsResponse
from backend.copilot.tools.routines import RoutineResponse, ScheduleRoutineTool
from backend.copilot.tools.schedule_followup import (
    ScheduleCreatedResponse,
    ScheduleFollowupTool,
)
from backend.data.model import APIKeyCredentials
from backend.executor.scheduler import CopilotTurnJobInfo
from backend.integrations.credentials_store import exa_credentials

from ._test_data import make_session

_USER = "user-1"
_PINS = "backend.copilot.tools.schedule_credentials"
_FOLLOWUP = "backend.copilot.tools.schedule_followup"
_ROUTINES = "backend.copilot.tools.routines"


def _exa_key(cred_id: str, title: str) -> APIKeyCredentials:
    return APIKeyCredentials(
        id=cred_id, provider="exa", title=title, api_key=SecretStr("k")
    )


_PERSONAL = _exa_key("exa-old", "Personal")
_WORK = _exa_key("exa-new", "Work")


@contextmanager
def _account(saved: list, picks: dict[str, str] | None = None):
    """The user's saved credentials (oldest first, then the platform's) and
    what they picked in this chat."""
    with (
        patch(
            f"{_PINS}.get_user_credentials",
            AsyncMock(return_value=[*saved, exa_credentials]),
        ),
        patch(f"{_PINS}.selected_credentials", AsyncMock(return_value=picks or {})),
    ):
        yield


@contextmanager
def _scheduler():
    client = AsyncMock()
    client.add_copilot_turn_schedule = AsyncMock(
        return_value=CopilotTurnJobInfo(
            schedule_id="cop-1",
            user_id=_USER,
            session_id=None,
            message="Daily briefing",
            run_at=None,
            cron="0 9 * * *",
            id="cop-1",
            name="briefing",
            next_run_time="2026-10-03T09:00:00+00:00",
            timezone="UTC",
        )
    )
    user_db = MagicMock()
    user_db.get_user_by_id = AsyncMock(return_value=MagicMock(timezone="UTC"))
    with (
        patch(f"{_FOLLOWUP}.get_scheduler_client", return_value=client),
        patch(f"{_FOLLOWUP}.user_db", return_value=user_db),
        patch(
            f"{_FOLLOWUP}.is_followups_feature_enabled",
            AsyncMock(return_value=True),
        ),
    ):
        yield client


async def _schedule_briefing(session: ChatSession, **kwargs):
    return await ScheduleFollowupTool()._execute(
        user_id=_USER,
        session=session,
        message="Daily briefing: search Exa for AI news",
        cron="0 9 * * *",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# schedule_followup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_briefing_using_one_of_two_exa_keys_asks_which_first():
    with _account([_PERSONAL, _WORK]), _scheduler() as scheduler:
        result = await _schedule_briefing(make_session(_USER), integrations=["exa"])

    assert isinstance(result, SetupRequirementsResponse)
    missing = result.setup_info.user_readiness.missing_credentials
    assert [entry["provider"] for entry in missing.values()] == ["exa"]
    assert "Personal" in result.message and "Work" in result.message
    scheduler.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_once_they_pick_the_schedule_stores_that_key():
    with (
        _account([_PERSONAL, _WORK], picks={"exa": "exa-new"}),
        _scheduler() as scheduler,
    ):
        result = await _schedule_briefing(make_session(_USER), integrations=["exa"])

    assert isinstance(result, ScheduleCreatedResponse), result
    kwargs = scheduler.add_copilot_turn_schedule.call_args.kwargs
    assert kwargs["credential_pins"] == {
        "exa": CredentialPin(id="exa-new", title="Work")
    }
    assert "Work" in result.message


@pytest.mark.asyncio
async def test_a_pick_already_made_in_the_chat_is_kept_without_naming_it():
    with (
        _account([_PERSONAL, _WORK], picks={"exa": "exa-new"}),
        _scheduler() as scheduler,
    ):
        await _schedule_briefing(make_session(_USER))

    kwargs = scheduler.add_copilot_turn_schedule.call_args.kwargs
    assert kwargs["credential_pins"]["exa"].id == "exa-new"


@pytest.mark.asyncio
async def test_a_single_exa_key_needs_no_choice_and_no_pin():
    with _account([_PERSONAL]), _scheduler() as scheduler:
        result = await _schedule_briefing(make_session(_USER), integrations=["exa"])

    assert isinstance(result, ScheduleCreatedResponse)
    assert scheduler.add_copilot_turn_schedule.call_args.kwargs["credential_pins"] == {}


@pytest.mark.asyncio
async def test_a_pick_of_someone_elses_credential_is_not_pinned():
    with (
        _account([_PERSONAL, _WORK], picks={"exa": "not-mine"}),
        _scheduler() as scheduler,
    ):
        result = await _schedule_briefing(make_session(_USER))

    assert isinstance(result, ScheduleCreatedResponse)
    assert scheduler.add_copilot_turn_schedule.call_args.kwargs["credential_pins"] == {}


@pytest.mark.asyncio
async def test_nothing_named_and_nothing_picked_never_reads_credentials():
    lookup = AsyncMock()
    with (
        patch(f"{_PINS}.get_user_credentials", lookup),
        patch(f"{_PINS}.selected_credentials", AsyncMock(return_value={})),
        _scheduler() as scheduler,
    ):
        result = await _schedule_briefing(make_session(_USER))

    assert isinstance(result, ScheduleCreatedResponse)
    lookup.assert_not_awaited()
    assert scheduler.add_copilot_turn_schedule.call_args.kwargs["credential_pins"] == {}


# ---------------------------------------------------------------------------
# schedule_routine
# ---------------------------------------------------------------------------


def _routine(**overrides) -> ExpertRoutine:
    return ExpertRoutine(
        **{
            "id": "routine-1",
            "title": "Morning briefing",
            "prompt": "Search Exa for AI news",
            "crons": ["0 9 * * *"],
            "session_mode": "THREAD",
            "source": "OWNER",
            "enabled": True,
            "grants_credentials": True,
            **overrides,
        }
    )


@pytest.fixture
def experts():
    db = MagicMock()
    db.get_routine = AsyncMock(return_value=None)
    db.create_routine = AsyncMock(return_value=_routine(enabled=False))
    db.enable_routine = AsyncMock(return_value=_routine())
    db.disable_routine = AsyncMock(return_value=_routine(enabled=False))
    with patch(f"{_ROUTINES}.experts_db", return_value=db):
        yield db


async def _routine_call(**kwargs):
    return await ScheduleRoutineTool()._execute(
        _USER, ChatSession.new(_USER, dry_run=False), **kwargs
    )


@pytest.mark.asyncio
async def test_a_new_routine_using_one_of_two_exa_keys_asks_which_first(experts):
    with _account([_PERSONAL, _WORK]):
        result = await _routine_call(
            title="Morning briefing",
            prompt="Search Exa for AI news",
            crons=["0 9 * * *"],
            enabled=True,
            integrations=["exa"],
        )

    assert isinstance(result, SetupRequirementsResponse)
    experts.create_routine.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_new_routine_stores_the_picked_key_on_its_row(experts):
    with _account([_PERSONAL, _WORK], picks={"exa": "exa-new"}):
        result = await _routine_call(
            title="Morning briefing",
            prompt="Search Exa for AI news",
            crons=["0 9 * * *"],
            enabled=True,
            integrations=["exa"],
        )

    assert isinstance(result, RoutineResponse), result
    pins = experts.create_routine.await_args.kwargs["credential_pins"]
    assert pins == {"exa": CredentialPin(id="exa-new", title="Work")}


@pytest.mark.asyncio
async def test_changing_a_routine_keeps_its_key_unless_another_is_picked(experts):
    experts.get_routine = AsyncMock(
        return_value=_routine(
            credential_pins={"exa": CredentialPin(id="exa-new", title="Work")}
        )
    )
    with _account([_PERSONAL, _WORK]):
        result = await _routine_call(
            routine_id="routine-1",
            enabled=True,
            prompt="Search Exa for robotics news",
            integrations=["exa"],
        )

    assert isinstance(result, RoutineResponse), result
    pins = experts.enable_routine.await_args.kwargs["credential_pins"]
    assert pins["exa"].id == "exa-new"


@pytest.mark.asyncio
async def test_a_pick_in_the_chat_moves_a_routine_to_that_key(experts):
    experts.get_routine = AsyncMock(
        return_value=_routine(
            credential_pins={"exa": CredentialPin(id="exa-new", title="Work")}
        )
    )
    with _account([_PERSONAL, _WORK], picks={"exa": "exa-old"}):
        await _routine_call(routine_id="routine-1", enabled=True)

    pins = experts.enable_routine.await_args.kwargs["credential_pins"]
    assert pins["exa"].id == "exa-old"


@pytest.mark.asyncio
async def test_a_routine_whose_key_was_deleted_asks_again(experts):
    experts.get_routine = AsyncMock(
        return_value=_routine(
            credential_pins={"exa": CredentialPin(id="exa-gone", title="Old")}
        )
    )
    with _account([_PERSONAL, _WORK]):
        result = await _routine_call(
            routine_id="routine-1", enabled=True, integrations=["exa"]
        )

    assert isinstance(result, SetupRequirementsResponse)
    experts.enable_routine.assert_not_awaited()


@pytest.mark.asyncio
async def test_switching_a_routine_off_leaves_its_keys_alone(experts):
    with _account([_PERSONAL, _WORK], picks={"exa": "exa-old"}):
        await _routine_call(routine_id="routine-1", enabled=False)

    experts.get_routine.assert_not_awaited()
    experts.disable_routine.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_routine_that_reaches_no_service_pins_nothing(experts):
    with _account([_PERSONAL, _WORK]):
        result = await _routine_call(
            title="Draft only",
            prompt="Draft a note",
            crons=["0 9 * * *"],
            enabled=True,
            integrations=["exa"],
            grants_credentials=False,
        )

    assert isinstance(result, RoutineResponse), result
    assert experts.create_routine.await_args.kwargs["credential_pins"] is None
