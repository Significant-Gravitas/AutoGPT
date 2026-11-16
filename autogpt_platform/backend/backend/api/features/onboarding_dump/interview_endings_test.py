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
    TextPick,
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
            "02-profile-card",
            "03-summary-chips",
            "04-team-reveal",
            "08-transcript-highlights",
            "09-first-task",
        },
    }


def test_each_mode_has_its_own_experiment() -> None:
    assert ENDING_EXPERIMENTS == {
        "text": Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT,
        "voice": Experiment.ONBOARDING_INTERVIEW_ENDING_VOICE,
    }
    assert Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT.value == (
        "onboarding-interview-ending-text"
    )
    assert Experiment.ONBOARDING_INTERVIEW_ENDING_VOICE.value == (
        "onboarding-interview-ending-voice"
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
@pytest.mark.parametrize(
    ("mode", "experiment", "arm"),
    [
        ("text", Experiment.ONBOARDING_INTERVIEW_ENDING_TEXT, "03-summary-chips"),
        ("voice", Experiment.ONBOARDING_INTERVIEW_ENDING_VOICE, "04-team-reveal"),
    ],
)
async def test_the_arm_is_read_for_the_mode_in_use_only(
    mocker: MockerFixture, mode, experiment: Experiment, arm: str
) -> None:
    assign = mocker.patch.object(
        interview_endings.experiments,
        "assign",
        new=AsyncMock(return_value=_assignment(arm)),
    )

    assert await assigned_ending("user-1", mode) == arm

    assign.assert_awaited_once_with("user-1", experiment)


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


# ---------- a tester's pick ----------


@pytest.fixture
def picker(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        interview_endings,
        "evaluate_feature_flag",
        new=AsyncMock(return_value=(True, True)),
    )


@pytest.fixture
def pick_store(mocker: MockerFixture):
    from fakeredis.aioredis import FakeRedis

    redis = FakeRedis(decode_responses=True)
    mocker.patch.object(
        interview_endings, "get_redis_async", new=AsyncMock(return_value=redis)
    )
    return redis


@pytest.mark.asyncio
async def test_a_testers_typed_pick_is_kept_until_the_close(picker, pick_store):
    await interview_endings.keep_text_pick("user-1", "06-coverage-ring")

    assert await interview_endings.text_pick("user-1") == TextPick(
        arm="06-coverage-ring", governs=True
    )


@pytest.mark.asyncio
async def test_default_in_the_picker_forgets_the_pick(picker, pick_store):
    await interview_endings.keep_text_pick("user-1", "06-coverage-ring")
    await interview_endings.keep_text_pick("user-1", None)

    # No pick governs: the experiment decides.
    assert await interview_endings.text_pick("user-1") == TextPick()


@pytest.mark.asyncio
async def test_a_pick_is_ignored_while_the_picker_is_off(picker, pick_store):
    picker.return_value = (False, True)

    await interview_endings.keep_text_pick("user-1", "06-coverage-ring")

    assert await pick_store.keys("*") == []
    assert await interview_endings.text_pick("user-1") == TextPick()


@pytest.mark.asyncio
async def test_a_pick_kept_while_the_picker_was_on_lapses_once_it_is_off(
    picker, pick_store
):
    await interview_endings.keep_text_pick("user-1", "03-summary-chips")
    picker.return_value = (False, True)

    # The kept pick still governs, so the close stamps no arm and enrols
    # nobody.
    assert await interview_endings.text_pick("user-1") == TextPick(governs=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("picker_on", [True, False])
async def test_a_pick_that_cannot_be_read_governs_only_for_a_tester(
    picker, mocker: MockerFixture, picker_on: bool
):
    picker.return_value = (picker_on, True)
    mocker.patch.object(
        interview_endings,
        "get_redis_async",
        new=AsyncMock(side_effect=ConnectionError("redis down")),
    )

    assert await interview_endings.text_pick("user-1") == TextPick(governs=picker_on)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "pick", "honoured"),
    [
        ("voice", "09-first-task", True),
        ("voice", "06-coverage-ring", False),
        ("text", "06-coverage-ring", True),
        ("text", "99-an-arm-from-a-later-release", False),
        ("voice", None, False),
    ],
)
async def test_only_an_arm_that_fits_the_mode_is_honoured(picker, mode, pick, honoured):
    expected = pick if honoured else None
    assert await interview_endings.honoured_pick("user-1", mode, pick) == expected


@pytest.mark.asyncio
async def test_an_unreadable_picker_flag_honours_no_pick(picker):
    picker.side_effect = RuntimeError("flag vendor down")

    assert (
        await interview_endings.honoured_pick("user-1", "voice", "09-first-task")
        is None
    )
