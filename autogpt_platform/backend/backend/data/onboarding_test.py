from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import OnboardingStep

from backend.data.onboarding import _reward_user, format_onboarding_for_extraction


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
        (OnboardingStep.VISIT_COPILOT, 300),
        (OnboardingStep.AGENT_NEW_RUN, 300),
        (OnboardingStep.MARKETPLACE_ADD_AGENT, 100),
        (OnboardingStep.MARKETPLACE_RUN_AGENT, 100),
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
