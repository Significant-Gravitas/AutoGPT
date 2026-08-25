from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock

from backend.data.onboarding import (
    OnboardingStep,
    _reward_user,
    ensure_user_onboarding,
    format_onboarding_for_extraction,
    get_user_onboarding,
)


def test_format_onboarding_for_extraction_basic():
    result = format_onboarding_for_extraction(
        user_name="John",
        user_role="Founder/CEO",
        pain_points=["Finding leads", "Email & outreach"],
    )
    assert "Q: What is your name?" in result
    assert "A: John" in result
    assert "Q: What best describes your role?" in result
    assert "A: Founder/CEO" in result
    assert "Q: What tasks are eating your time?" in result
    assert "Finding leads" in result
    assert "Email & outreach" in result


def test_format_onboarding_for_extraction_with_other():
    result = format_onboarding_for_extraction(
        user_name="Jane",
        user_role="Data Scientist",
        pain_points=["Research", "Building dashboards"],
    )
    assert "A: Jane" in result
    assert "A: Data Scientist" in result
    assert "Research, Building dashboards" in result


@pytest.mark.asyncio(loop_scope="function")
@pytest.mark.parametrize(
    "step,expected_reward",
    [
        (OnboardingStep.ONBOARDING_COMPLETE, 300),
        (OnboardingStep.AGENT_NEW_RUN, 300),
        (OnboardingStep.MARKETPLACE_ADD_AGENT, 100),
        (OnboardingStep.LIBRARY_RUN_AGENT, 100),
        (OnboardingStep.SCHEDULE_AGENT, 100),
        (OnboardingStep.RUN_3_DAYS, 100),
        (OnboardingStep.TRIGGER_WEBHOOK, 100),
        (OnboardingStep.RUN_14_DAYS, 100),
        (OnboardingStep.RUN_AGENTS_100, 100),
    ],
)
async def test_reward_user_grants_expected_amount(
    mocker: pytest_mock.MockFixture,
    step: OnboardingStep,
    expected_reward: int,
):
    onboarding = Mock()
    onboarding.rewardedFor = []

    credit_model = Mock()
    credit_model.onboarding_reward = AsyncMock()
    mocker.patch(
        "backend.data.onboarding.get_user_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mock_prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    mock_prisma.return_value.update = AsyncMock()

    await _reward_user("user-1", onboarding, step)

    credit_model.onboarding_reward.assert_called_once_with(
        "user-1", expected_reward, step
    )


@pytest.mark.asyncio(loop_scope="function")
async def test_reward_user_skips_if_already_rewarded(
    mocker: pytest_mock.MockFixture,
):
    onboarding = Mock()
    onboarding.rewardedFor = [OnboardingStep.RUN_14_DAYS]

    credit_model = Mock()
    credit_model.onboarding_reward = AsyncMock()
    mocker.patch(
        "backend.data.onboarding.get_user_credit_model",
        AsyncMock(return_value=credit_model),
    )

    await _reward_user("user-1", onboarding, OnboardingStep.RUN_14_DAYS)

    credit_model.onboarding_reward.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_reward_user_noop_for_zero_reward_step(
    mocker: pytest_mock.MockFixture,
):
    # Steps without a configured reward (e.g. WELCOME) must never touch the
    # credit model — they are pure progress markers.
    onboarding = Mock()
    onboarding.rewardedFor = []

    credit_model = Mock()
    credit_model.onboarding_reward = AsyncMock()
    mocker.patch(
        "backend.data.onboarding.get_user_credit_model",
        AsyncMock(return_value=credit_model),
    )

    await _reward_user("user-1", onboarding, OnboardingStep.WELCOME)

    credit_model.onboarding_reward.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_reward_user_accepts_plain_string_step(
    mocker: pytest_mock.MockFixture,
):
    # The completion endpoint validates against ``FrontendOnboardingStep``
    # (a ``Literal[OnboardingStep, ...]``), which can hand the data layer a
    # plain ``str`` rather than an enum instance. The persist path must coerce
    # with ``str()``, not ``.value`` (which would AttributeError on a str).
    onboarding = Mock()
    onboarding.rewardedFor = []

    credit_model = Mock()
    credit_model.onboarding_reward = AsyncMock()
    mocker.patch(
        "backend.data.onboarding.get_user_credit_model",
        AsyncMock(return_value=credit_model),
    )
    mock_prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    mock_prisma.return_value.update = AsyncMock()

    # Pass a raw string typed as OnboardingStep to mimic the endpoint boundary.
    await _reward_user(
        "user-1", onboarding, cast(OnboardingStep, "ONBOARDING_COMPLETE")
    )

    credit_model.onboarding_reward.assert_called_once()
    persisted = mock_prisma.return_value.update.call_args.kwargs["data"]["rewardedFor"]
    assert persisted == ["ONBOARDING_COMPLETE"]


def test_onboarding_step_values_are_plain_strings():
    # StrEnum members must round-trip through ``str`` unchanged so they can be
    # written directly into the ``String[]`` columns on UserOnboarding.
    assert str(OnboardingStep.ONBOARDING_COMPLETE) == "ONBOARDING_COMPLETE"
    assert OnboardingStep.ONBOARDING_COMPLETE == "ONBOARDING_COMPLETE"
    # Legacy values that have been retired (e.g. VISIT_COPILOT) must not appear
    # in the active set — existing rows containing them remain inert strings.
    assert "VISIT_COPILOT" not in {step.value for step in OnboardingStep}


@pytest.mark.asyncio(loop_scope="function")
async def test_get_user_onboarding_does_not_write_for_unprovisioned_user(
    mocker: pytest_mock.MockerFixture,
):
    # A valid session can outrun the platform User row it hangs off. This read
    # runs on every page load, so it has to answer from what is already there:
    # creating a row for a user that does not exist yet trips the FK to User
    # and 500s the request instead.
    mock_prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    mock_prisma.return_value.find_unique = AsyncMock(return_value=None)
    mock_prisma.return_value.upsert = AsyncMock()

    onboarding = await get_user_onboarding("user-1")

    mock_prisma.return_value.upsert.assert_not_called()
    assert onboarding.userId == "user-1"
    assert onboarding.completedSteps == []
    assert onboarding.agentRuns == 0


@pytest.mark.asyncio(loop_scope="function")
async def test_get_user_onboarding_returns_the_stored_row(
    mocker: pytest_mock.MockerFixture,
):
    stored = Mock(userId="user-1", completedSteps=["ONBOARDING_COMPLETE"])
    mock_prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    mock_prisma.return_value.find_unique = AsyncMock(return_value=stored)
    mock_prisma.return_value.upsert = AsyncMock()

    assert await get_user_onboarding("user-1") is stored
    mock_prisma.return_value.upsert.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_ensure_user_onboarding_creates_the_row(
    mocker: pytest_mock.MockerFixture,
):
    # Write paths still need the row to exist: each one follows up with an
    # `update`, which has nothing to target if the row was never created.
    mock_prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    mock_prisma.return_value.upsert = AsyncMock(return_value=Mock())

    await ensure_user_onboarding("user-1")

    assert mock_prisma.return_value.upsert.call_args.kwargs["where"] == {
        "userId": "user-1"
    }
