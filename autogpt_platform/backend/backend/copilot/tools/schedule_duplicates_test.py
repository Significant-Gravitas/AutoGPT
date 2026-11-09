from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.credential_selection import CredentialPin
from backend.data.model import CredentialsMetaInput
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from ._test_data import make_session
from .schedule_duplicates import find_duplicate_followup, find_duplicate_graph_schedule

_USER = "test-user-duplicates"
_NOON = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
_MESSAGE = "Check whether the repo is trending and post if it is."


def _followup(**overrides) -> CopilotTurnJobInfo:
    fields = {
        "schedule_id": "existing",
        "id": "existing",
        "user_id": _USER,
        "session_id": None,
        "message": _MESSAGE,
        "cron": "0 12 * * *",
        "run_at": None,
        "name": "copilot turn (new chat)",
        "credential_pins": {"github": CredentialPin(id="account-1", title="Work")},
        "next_run_time": _NOON.isoformat(),
        "timezone": "America/Chicago",
    }
    fields.update(overrides)
    return CopilotTurnJobInfo(**fields)


def _account(account_id: str) -> CredentialsMetaInput:
    return CredentialsMetaInput(id=account_id, provider="github", type="api_key")


def _graph_schedule(**overrides) -> GraphExecutionJobInfo:
    fields = {
        "schedule_id": "existing",
        "id": "existing",
        "user_id": _USER,
        "graph_id": "graph",
        "graph_version": 2,
        "cron": "0 12 * * *",
        "input_data": {"repo": "Significant-Gravitas/AutoGPT"},
        "input_credentials": {"github": _account("account-1")},
        "name": "Trending check",
        "organization_id": "org-a",
        "team_id": "team-a",
        "next_run_time": _NOON.isoformat(),
        "timezone": "America/Chicago",
    }
    fields.update(overrides)
    return GraphExecutionJobInfo(**fields)


def _scheduler(*jobs) -> MagicMock:
    return MagicMock(get_execution_schedules=AsyncMock(return_value=list(jobs)))


def _session(*, organization_id=None, team_id=None, expert_id=None):
    session = make_session(_USER, expert_id=expert_id)
    session.organization_id = organization_id
    session.team_id = team_id
    return session


async def _find_followup(job, session=None, **request):
    wanted = {
        "message": _MESSAGE,
        "target_session_id": None,
        "cron": "0 12 * * *",
        "run_at": None,
        "timezone": "America/Chicago",
        "credential_pins": {"github": CredentialPin(id="account-1")},
    }
    wanted.update(request)
    return await find_duplicate_followup(
        _scheduler(job), _USER, session or _session(), **wanted
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_overrides",
    [
        pytest.param({}, id="same cron and message"),
        pytest.param(
            {"message": f"  {_MESSAGE.replace(' ', '  ')}\n", "cron": "0  12 * * *"},
            id="only the spacing differs",
        ),
    ],
)
async def test_the_same_recurring_followup_is_a_duplicate(request_overrides):
    found = await _find_followup(_followup(), **request_overrides)
    assert found is not None and found.id == "existing"


@pytest.mark.asyncio
async def test_a_one_shot_asked_for_again_minutes_later_is_a_duplicate():
    """The ticket's 12:00 and 12:04: the same "at noon" asked for twice, with
    the delay counted from two moments a few minutes apart."""
    job = _followup(cron=None, run_at=_NOON)
    found = await _find_followup(job, cron=None, run_at=_NOON + timedelta(minutes=4))
    assert found is not None and found.id == "existing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "job_overrides, request_overrides",
    [
        pytest.param({}, {"cron": "0 17 * * *"}, id="another time"),
        pytest.param({}, {"timezone": "Europe/London"}, id="another timezone"),
        pytest.param({}, {"message": "Check the CI."}, id="another message"),
        pytest.param({}, {"target_session_id": "chat-2"}, id="another chat"),
        pytest.param({"expert_id": "expert-1"}, {}, id="an expert's schedule"),
        pytest.param({"routine_id": "routine-1"}, {}, id="a routine's job"),
        pytest.param(
            {},
            {"credential_pins": {"github": CredentialPin(id="account-2")}},
            id="another account",
        ),
        pytest.param({}, {"credential_pins": {}}, id="no account pinned"),
        pytest.param(
            {"organization_id": "org-a", "team_id": "team-a"},
            {},
            id="another organization",
        ),
        pytest.param(
            {"cron": None, "run_at": _NOON},
            {"cron": None, "run_at": _NOON + timedelta(minutes=30)},
            id="one-shots half an hour apart",
        ),
        pytest.param(
            {"cron": None, "run_at": _NOON},
            {},
            id="a cadence where a one-shot is set",
        ),
    ],
)
async def test_a_different_followup_is_not_a_duplicate(
    job_overrides, request_overrides
):
    assert await _find_followup(_followup(**job_overrides), **request_overrides) is None


async def _find_graph_schedule(job, session=None, **request):
    wanted = {
        "graph_id": "graph",
        "graph_version": 2,
        "cron": "0 12 * * *",
        "timezone": "America/Chicago",
        "inputs": {"repo": "Significant-Gravitas/AutoGPT"},
        "input_defaults": {},
        "credentials": {"github": _account("account-1")},
        "organization_id": "org-a",
        "team_id": "team-a",
    }
    wanted.update(request)
    return await find_duplicate_graph_schedule(
        _scheduler(job), _USER, session or _session(), **wanted
    )


@pytest.mark.asyncio
async def test_the_same_workflow_schedule_under_another_name_is_a_duplicate():
    found = await _find_graph_schedule(_graph_schedule(name="Daily trending"))
    assert found is not None and found.id == "existing"


@pytest.mark.asyncio
async def test_an_input_left_to_its_default_matches_the_default_given():
    """use_defaults stores no value; the run fills the same default in."""
    defaults = {"repo": "Significant-Gravitas/AutoGPT"}
    found = await _find_graph_schedule(
        _graph_schedule(input_data={}), inputs=defaults, input_defaults=defaults
    )
    assert found is not None and found.id == "existing"


@pytest.mark.asyncio
async def test_an_experts_schedule_matches_whatever_tenancy_it_was_moved_to():
    """An expert's jobs are re-homed to her private tenancy when made, so the
    chat's own organization is not what they carry."""
    found = await _find_graph_schedule(
        _graph_schedule(expert_id="expert-1", organization_id="private-org"),
        session=_session(expert_id="expert-1"),
    )
    assert found is not None and found.id == "existing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_overrides",
    [
        pytest.param({"cron": "0 17 * * *"}, id="another time"),
        pytest.param({"timezone": "UTC"}, id="another timezone"),
        pytest.param({"graph_version": 3}, id="another version"),
        pytest.param({"inputs": {"repo": "other/repo"}}, id="other inputs"),
        pytest.param(
            {"credentials": {"github": _account("account-2")}},
            id="another account",
        ),
        pytest.param({"team_id": "team-b"}, id="another team"),
        pytest.param({"inputs": {}}, id="a missing input with no default"),
    ],
)
async def test_a_different_workflow_schedule_is_not_a_duplicate(request_overrides):
    assert await _find_graph_schedule(_graph_schedule(), **request_overrides) is None
