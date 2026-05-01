from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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


@pytest.mark.asyncio
async def test_visit_copilot_grants_no_reward():
    # The $5 signup bonus tied to VISIT_COPILOT (PR #11862) has been retired.
    # The step still completes for redirect/analytics purposes, but must not
    # mint credits.
    onboarding = MagicMock(rewardedFor=[])

    with patch(
        "backend.data.onboarding.get_user_credit_model", new=AsyncMock()
    ) as get_credit_model:
        await _reward_user(
            user_id="test-user-id",
            onboarding=onboarding,
            step=OnboardingStep.VISIT_COPILOT,
        )

    get_credit_model.assert_not_called()
    assert onboarding.rewardedFor == []
