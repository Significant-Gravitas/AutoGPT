"""The ending experiment's arms: where each is offered, and which wait for a
confirm."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from prisma.errors import PrismaError
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import interview_endings
from backend.api.features.onboarding_dump.interview_endings import (
    CONFIRM_ENDINGS,
    ENDING_EXPERIMENTS,
    MODE_ENDINGS,
    InterviewEnding,
    assigned_ending,
    hands_off_on_close,
)
from backend.data.experiments import Experiment, ExperimentAssignment


def test_every_ending_is_an_arm_of_a_mode() -> None:
    offered = MODE_ENDINGS["text"] | MODE_ENDINGS["voice"]
    assert offered == set(InterviewEnding)


def test_each_mode_offers_exactly_the_arms_that_fit_it() -> None:
    """The variant keys each mode's flag carries."""
    offered = {
        mode: {ending.value for ending in endings}
        for mode, endings in MODE_ENDINGS.items()
    }
    assert offered == {
        "text": {
            "02-profile-card",
            "03-summary-chips",
            "04-team-reveal",
            "06-coverage-ring",
            "08-transcript-highlights",
            "09-first-task",
        },
        "voice": {
            "01-spoken-yes",
            "02-profile-card",
            "04-team-reveal",
            "06-coverage-ring",
            "07-hang-up",
            "09-first-task",
        },
    }


def test_each_mode_has_its_own_experiment() -> None:
    assert ENDING_EXPERIMENTS == {
        "text": Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT,
    }
    assert Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT.value == (
        "onboarding-interview-ending-text"
    )


def test_the_card_chips_highlights_and_first_task_end_on_a_confirm() -> None:
    assert CONFIRM_ENDINGS == {
        "02-profile-card",
        "03-summary-chips",
        "08-transcript-highlights",
        "09-first-task",
    }
    for ending in InterviewEnding:
        assert hands_off_on_close(ending.value) is (ending.value not in CONFIRM_ENDINGS)


def test_anyone_not_enrolled_is_handed_off_as_the_interview_closes() -> None:
    assert hands_off_on_close(None)
    assert hands_off_on_close("an-arm-this-code-does-not-know")


def _assignment(variant: str) -> ExperimentAssignment:
    return ExperimentAssignment(
        experiment_key=Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT.value,
        variant=variant,
        source="backend",
        assigned_at=datetime(2026, 11, 3, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_the_arm_is_read_for_the_mode_in_use_only(mocker: MockerFixture) -> None:
    assign = mocker.patch.object(
        interview_endings.experiments,
        "assign",
        new=AsyncMock(return_value=_assignment("03-summary-chips")),
    )

    assert await assigned_ending("user-1", "text") == "03-summary-chips"

    assign.assert_awaited_once_with(
        "user-1", Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT
    )


@pytest.mark.asyncio
async def test_an_unenrolled_person_has_no_arm(mocker: MockerFixture) -> None:
    mocker.patch.object(
        interview_endings.experiments, "assign", new=AsyncMock(return_value=None)
    )

    assert await assigned_ending("user-1", "text") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [PrismaError("down"), RuntimeError("posthog down")])
async def test_an_arm_that_cannot_be_read_never_holds_up_the_interview(
    mocker: MockerFixture, error: Exception
) -> None:
    mocker.patch.object(
        interview_endings.experiments, "assign", new=AsyncMock(side_effect=error)
    )

    assert await assigned_ending("user-1", "text") is None
