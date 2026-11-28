"""What the owner of a schedule whose runs are skipped reads (#17245).

Each problem names its own fix. A removed key isn't an expired sign-in, so
nothing here says to reconnect: a schedule's key can't be changed in the
Library yet, so the fix is a new schedule.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import AlertCause, AlertConditionStatus

from backend.data.alerts import AlertConditionDTO
from backend.notifications.alert_causes import (
    _BY_CAUSE,
    SEVERITY,
    ScheduleSkippedCause,
    parse_cause,
)
from backend.notifications.alerts import ALERT_DEBOUNCE, build_alert_email
from backend.notifications.conftest import make_db_client

CTA = "/library/agents/lib-1?activeTab=scheduled&activeItem=sched-1"


def _cause(problem: str, **slots) -> ScheduleSkippedCause:
    return ScheduleSkippedCause.model_validate(
        {
            "cta_path": CTA,
            "agent": "Daily digest",
            "problem": problem,
            "schedule": "Morning run",
            **slots,
        }
    )


@pytest.mark.parametrize(
    "problem, slots, subject, preheader, cta",
    [
        (
            "key_removed",
            {"provider": "Unreal Speech", "key_title": "Work"},
            "Daily digest is skipping its scheduled runs: the key it used was "
            "removed",
            "The Unreal Speech key “Work” its schedule “Morning run” ran on was "
            "removed.",
            "See the schedule",
        ),
        (
            "key_unusable",
            {},
            "Daily digest is skipping its scheduled runs: it can’t use one of its "
            "keys",
            "One of the keys its schedule “Morning run” runs on can’t be used.",
            "See the schedule",
        ),
        (
            "inputs_invalid",
            {},
            "Daily digest is skipping its scheduled runs: its inputs no longer "
            "pass the workflow’s checks",
            "The inputs its schedule “Morning run” runs with no longer pass the "
            "workflow’s checks.",
            "See the schedule",
        ),
        (
            "step_disabled",
            {"step": "Text to Speech"},
            "Daily digest is skipping its scheduled runs: one of its steps is "
            "switched off",
            "Its step “Text to Speech” is switched off, so its schedule “Morning "
            "run” can’t start a run.",
            "Replace the step",
        ),
    ],
)
def test_each_problem_names_its_own_fix(problem, slots, subject, preheader, cta):
    cause = _cause(problem, **slots)

    assert cause.subject_line == subject
    assert cause.preheader == preheader
    assert cause.body.startswith(f"{preheader} The schedule is kept")
    assert cause.cta_label == cta
    assert "reconnect" not in cause.body.lower()


def test_a_removed_key_reads_well_without_its_name_or_its_schedules():
    cause = _cause("key_removed", schedule=None)

    assert cause.preheader == "A key its schedule ran on was removed."
    assert cause.body.endswith(
        "skips each run until it has another key: schedule the workflow again "
        "with one, then delete this schedule."
    )


@pytest.mark.asyncio
async def test_the_alert_engine_sends_it_like_any_other_cause():
    cause = _cause("key_removed", provider="Unreal Speech")
    row = AlertConditionDTO(
        id="c-1",
        user_id="user-1",
        cause=AlertCause.SCHEDULE_SKIPPED,
        cause_key="schedule_skipped:sched-1",
        data=cause.model_dump(mode="json"),
        status=AlertConditionStatus.PENDING,
        created_at=datetime.now(tz=timezone.utc) - ALERT_DEBOUNCE - timedelta(1),
        sent_at=None,
        briefed_at=None,
    )
    client = make_db_client(get_pending_alert_conditions=AsyncMock(return_value=[row]))

    with patch("backend.notifications.alerts._db", return_value=client):
        built = await build_alert_email("user-1", alerts_enabled=True)

    assert built is not None
    assert built.data.primary.subject == cause.subject_line
    assert built.data.primary.cta_url.endswith(CTA)
    assert parse_cause(row.cause, row.data) == cause


def test_every_cause_has_a_model_and_a_severity():
    """The engine sorts by severity and rebuilds rows by cause: a cause
    missing from either fails the send it is in."""
    assert set(_BY_CAUSE) == set(SEVERITY) == set(AlertCause)
