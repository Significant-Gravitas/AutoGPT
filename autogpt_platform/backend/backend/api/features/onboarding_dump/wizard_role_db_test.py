"""The hire step's team for a user still in the wizard, against Postgres.

The wizard keeps the user's answers in its saved progress until its last
step, which comes after the hire step, so only the real save shows what the
team can be built from by then.
"""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import BackgroundTasks
from prisma.models import User
from pytest_mock import MockerFixture

from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import recommend_experts, service
from backend.api.features.onboarding_dump.models import ExpertRecommendations
from backend.data.onboarding import UserOnboardingUpdate, update_user_onboarding
from backend.data.onboarding_wizard import OnboardingWizardProgress

ROSTER = [
    Expert.model_construct(
        id=f"tpl-{name.lower()}",
        name=name,
        role=role,
        avatar_url=None,
        tagline=f"{name} handles {role.lower()}.",
        workflows=[],
    )
    for name, role in [("Max", "Sales"), ("Maya", "Marketing")]
]


@pytest_asyncio.fixture(loop_scope="session")
async def user_id(server) -> AsyncGenerator[str, None]:
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    yield user_id
    await User.prisma().delete(where={"id": user_id})


@pytest.fixture(autouse=True)
def roster(mocker: MockerFixture) -> None:
    mocker.patch.object(service, "list_templates", new=AsyncMock(return_value=ROSTER))
    mocker.patch.object(service, "enforce_personalization_budget", new=AsyncMock())
    mocker.patch(
        "backend.api.features.onboarding_dump.quality.check_transcript_quality",
        new=AsyncMock(return_value=None),
    )


async def test_a_skip_after_picking_a_role_is_offered_that_roles_team(user_id: str):
    await _pick_role(user_id, "Marketing")
    await service.finalize_skipped_dump(user_id, "rec-skip")

    response = await service.get_recommended_experts(user_id)

    assert response.ready
    assert response.team is not None
    assert [expert.name for expert in response.team.experts] == ["Maya"]


async def test_the_team_job_gets_the_role_picked_in_the_wizard(
    user_id: str, mocker: MockerFixture
):
    await _pick_role(user_id, "Sales/BD")
    mocker.patch.object(service, "_run_completion", new=AsyncMock(return_value=True))
    mocker.patch.object(service, "_run_provider_recommendations", new=AsyncMock())
    pick_team = mocker.patch.object(
        recommend_experts,
        "generate_expert_recommendations",
        new=AsyncMock(return_value=ExpertRecommendations(source="llm")),
    )
    tasks = BackgroundTasks()
    await service.finalize_typed_dump(
        user_id, "rec-typed", "I run a two-person agency and lose leads.", tasks
    )

    await tasks()

    assert pick_team.await_args is not None
    assert pick_team.await_args.kwargs["user_role"] == "Sales/BD"


async def _pick_role(user_id: str, role: str) -> None:
    await update_user_onboarding(
        user_id,
        UserOnboardingUpdate(
            wizardProgress=OnboardingWizardProgress(
                version=1,
                currentStep="painPoints",
                completedSteps=["team", "autopilot", "role"],
                role=role,
                otherRole="",
                painPoints=[],
                otherPainPoint="",
                selectedBilling="monthly",
                hasUserSelectedBilling=False,
                selectedCountryCode="US",
                hiredTemplateIds=[],
            ),
            wizardRevision=0,
            wizardUserId=user_id,
        ),
    )
