import logging
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
from backend.data.onboarding_role import (
    OnboardingRole,
    get_onboarding_role,
    kept_role,
    save_onboarding_role,
)
from backend.data.onboarding_wizard import OnboardingWizardConflict
from backend.data.tally import extract_business_understanding
from backend.data.understanding import (
    BusinessUnderstandingInput,
    get_business_understanding,
    upsert_business_understanding,
)
from backend.util.product_analytics import set_onboarding_role

logger = logging.getLogger(__name__)

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
    answered = (
        data.wizardProgress is not None
        and OnboardingRole.from_wizard(data.wizardProgress) is not None
    )
    sent = await _sent_role(user_id) if answered else None
    try:
        onboarding = await update_user_onboarding(user_id, data)
    except OnboardingWizardConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if answered and kept_role(onboarding) != sent:
        await _send_answered_role(user_id, sent)
    return onboarding


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

    # Empty for an account that gave no name at sign-up.
    user_name: Annotated[
        str, pydantic.StringConstraints(strip_whitespace=True, max_length=100)
    ]
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
        and (existing.user_name or "") == data.user_name
        and existing.user_role == data.user_role
        and existing.pain_points == data.pain_points
    ):
        await _keep_role(user_id, OnboardingRole.from_answer(data.user_role))
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
    understanding_input.user_name = data.user_name or None
    understanding_input.user_role = data.user_role
    if not understanding_input.pain_points:
        understanding_input.pain_points = data.pain_points

    await upsert_business_understanding(user_id, understanding_input)
    await _keep_role(user_id, OnboardingRole.from_answer(data.user_role))

    return {"status": "ok"}


async def _sent_role(user_id: str) -> OnboardingRole | None:
    """The pick kept before this save, to tell whether the save changes it.
    A failed read never costs the draft its save: the pick is then sent as if
    new, a harmless repeat."""
    try:
        return await get_onboarding_role(user_id)
    except Exception:
        logger.warning(f"Could not read the onboarding role of {user_id}")
        return None


async def _send_answered_role(user_id: str, sent: OnboardingRole | None) -> None:
    """Send the role once the person moves on from the role step, not only at
    the end of the wizard, which most people who stop at the paywall never
    reach. The draft save keeps it on the account (`update_user_onboarding`),
    so the checkout event that follows carries it to MailerLite
    (`checkout_audience`); PostHog, and a MailerLite subscriber who already
    opened checkout, get it now.

    What is kept is read again just before sending, so a request that paused
    after its save sends the newer pick another tab has saved since, not its
    own. Only a pick that differs from the one kept before the save is sent,
    so the draft saves of every later step send nothing. The draft is saved
    whatever happens here: the profile step at the end sends the role again,
    and the role backfill fills in what a failed send missed."""
    try:
        role = await get_onboarding_role(user_id)
        if role is None or role == sent:
            return
        set_onboarding_role(user_id=user_id, role=role)
        await queue_onboarding_role(user_id, role)
    except Exception:
        logger.exception(f"Could not send the onboarding role of {user_id}")


async def _keep_role(user_id: str, role: OnboardingRole) -> None:
    """Keep the pick apart from the understanding, whose copy AutoPilot
    rewrites, and send it to MailerLite and PostHog. An unchanged profile
    keeps it too, so a retry repairs a write an earlier attempt lost."""
    await save_onboarding_role(user_id, role)
    await queue_onboarding_role(user_id, role)
    set_onboarding_role(user_id=user_id, role=role)
