from typing import Annotated, get_args

import pydantic
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Security

from backend.api.features.onboarding_dump.rate_limit import (
    enforce_personalization_budget,
)
from backend.api.features.store.model import StoreAgentDetails
from backend.data.model import UserOnboarding
from backend.data.onboarding import (
    FrontendOnboardingStep,
    OnboardingStep,
    UserOnboardingUpdate,
    complete_onboarding_step,
    format_onboarding_for_extraction,
    get_recommended_agents,
    get_user_onboarding,
    reset_user_onboarding,
    update_user_onboarding,
)
from backend.data.onboarding_audience import queue_onboarding_role
from backend.data.onboarding_role import OnboardingRole, save_onboarding_role
from backend.data.onboarding_wizard import OnboardingWizardConflict
from backend.data.tally import extract_business_understanding
from backend.data.understanding import (
    BusinessUnderstandingInput,
    get_business_understanding,
    upsert_business_understanding,
)
from backend.util.product_analytics import set_onboarding_role

# Tags stay per-route: /onboarding/completed publishes ["onboarding", "public"]
# while the other six publish ["onboarding"].
router = APIRouter(dependencies=[Security(requires_user)])


@router.get(
    "/onboarding",
    summary="Onboarding state",
    tags=["onboarding"],
    response_model=UserOnboarding,
)
async def get_onboarding(user_id: Annotated[str, Security(get_user_id)]):
    return await get_user_onboarding(user_id)


@router.patch(
    "/onboarding",
    summary="Update onboarding state",
    tags=["onboarding"],
    response_model=UserOnboarding,
    responses={409: {"description": "The wizard draft changed in another session"}},
)
async def update_onboarding(
    user_id: Annotated[str, Security(get_user_id)], data: UserOnboardingUpdate
):
    try:
        return await update_user_onboarding(user_id, data)
    except OnboardingWizardConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/onboarding/step",
    summary="Complete onboarding step",
    tags=["onboarding"],
)
async def onboarding_complete_step(
    user_id: Annotated[str, Security(get_user_id)], step: FrontendOnboardingStep
):
    if step not in get_args(FrontendOnboardingStep):
        raise HTTPException(status_code=400, detail="Invalid onboarding step")
    return await complete_onboarding_step(user_id, step)


@router.get(
    "/onboarding/agents",
    summary="Recommended onboarding agents",
    tags=["onboarding"],
)
async def get_onboarding_agents(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[StoreAgentDetails]:
    return await get_recommended_agents(user_id)


class OnboardingProfileRequest(pydantic.BaseModel):
    """Request body for onboarding profile submission."""

    user_name: str = pydantic.Field(min_length=1, max_length=100)
    user_role: str = pydantic.Field(min_length=1, max_length=100)
    pain_points: list[Annotated[str, pydantic.StringConstraints(max_length=2000)]] = (
        pydantic.Field(default_factory=list, max_length=20)
    )


class OnboardingStatusResponse(pydantic.BaseModel):
    """Response for onboarding completion check."""

    is_completed: bool


@router.get(
    "/onboarding/completed",
    summary="Check if onboarding is completed",
    tags=["onboarding", "public"],
    response_model=OnboardingStatusResponse,
)
async def is_onboarding_completed(
    user_id: Annotated[str, Security(get_user_id)],
) -> OnboardingStatusResponse:
    user_onboarding = await get_user_onboarding(user_id)
    return OnboardingStatusResponse(
        is_completed=OnboardingStep.ONBOARDING_COMPLETE
        in user_onboarding.completedSteps,
    )


@router.post(
    "/onboarding/reset",
    summary="Reset onboarding progress",
    tags=["onboarding"],
    response_model=UserOnboarding,
)
async def reset_onboarding(user_id: Annotated[str, Security(get_user_id)]):
    return await reset_user_onboarding(user_id)


@router.post(
    "/onboarding/profile",
    summary="Submit onboarding profile",
    tags=["onboarding"],
)
async def submit_onboarding_profile(
    data: OnboardingProfileRequest,
    user_id: Annotated[str, Security(get_user_id)],
):
    existing = await get_business_understanding(user_id)
    if (
        existing
        and existing.user_name == data.user_name
        and existing.user_role == data.user_role
        and existing.pain_points == data.pain_points
    ):
        await _keep_role(user_id, data.user_role)
        return {"status": "ok"}
    await enforce_personalization_budget(user_id)
    formatted = format_onboarding_for_extraction(
        user_name=data.user_name,
        user_role=data.user_role,
        pain_points=data.pain_points,
    )

    try:
        understanding_input = await extract_business_understanding(formatted)
    except Exception:
        understanding_input = BusinessUnderstandingInput.model_construct()

    # Ensure the direct fields are set even if LLM missed them
    understanding_input.user_name = data.user_name
    understanding_input.user_role = data.user_role
    if not understanding_input.pain_points:
        understanding_input.pain_points = data.pain_points

    await upsert_business_understanding(user_id, understanding_input)
    await _keep_role(user_id, data.user_role)

    return {"status": "ok"}


async def _keep_role(user_id: str, answer: str) -> None:
    """Keep the pick apart from the understanding, whose copy AutoPilot
    rewrites, and send it to MailerLite and PostHog. An unchanged profile
    keeps it too, so a retry repairs a write an earlier attempt lost."""
    role = OnboardingRole.from_answer(answer)
    await save_onboarding_role(user_id, role)
    await queue_onboarding_role(user_id, role)
    set_onboarding_role(user_id=user_id, role=role)
