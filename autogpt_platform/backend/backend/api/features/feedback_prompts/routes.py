"""In-app feedback prompts: claim a showing, then save each tap as it happens.

An admin acting as a user creates no rows and sends no events.
"""

from typing import Annotated, cast

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, HTTPException, Request, Security
from prisma.models import FeedbackPrompt
from starlette.status import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY

from backend.api.features.feedback_prompts.analytics import (
    capture_prompt_answered,
    capture_prompt_shown,
)
from backend.api.features.feedback_prompts.model import (
    ClaimFeedbackPromptRequest,
    ClaimFeedbackPromptResponse,
    FeedbackPromptState,
    UpdateFeedbackPromptRequest,
)
from backend.copilot.llm_attribution import is_impersonated_request
from backend.data.feedback_prompts import (
    REASONS_BY_KIND,
    SCORES_BY_KIND,
    FeedbackPromptKind,
    claim_feedback_prompt,
    get_feedback_prompt,
    update_feedback_prompt,
)

router = APIRouter()

# These answers reach PostHog as ``$ai_metric`` on the reply's trace, through
# the chat feedback route. Their row only holds the card's state.
_ANSWERED_THROUGH_CHAT_FEEDBACK = frozenset({"work_check", "otto_mistake"})

_NOT_FOUND_RESPONSES: dict[int | str, dict[str, str]] = {
    HTTP_404_NOT_FOUND: {"description": "No such prompt, or not the caller's"},
}


@router.post(
    "/claim",
    summary="Claim feedback prompt",
    dependencies=[Security(requires_user)],
)
async def claim_prompt(
    body: ClaimFeedbackPromptRequest,
    user_id: Annotated[str, Security(get_user_id)],
    request: Request,
) -> ClaimFeedbackPromptResponse:
    """Record a showing if the prompt's limit allows one."""
    if is_impersonated_request(request):
        return ClaimFeedbackPromptResponse(id=None)
    prompt_id = await claim_feedback_prompt(user_id=user_id, kind=body.kind)
    if prompt_id is not None:
        capture_prompt_shown(user_id, prompt_id, body.kind)
    return ClaimFeedbackPromptResponse(id=prompt_id)


@router.patch(
    "/{id}",
    summary="Update feedback prompt",
    dependencies=[Security(requires_user)],
    responses=_NOT_FOUND_RESPONSES,
)
async def update_prompt(
    id: str,
    body: UpdateFeedbackPromptRequest,
    user_id: Annotated[str, Security(get_user_id)],
    request: Request,
) -> FeedbackPromptState:
    """Save the fields given and keep the rest."""
    prompt = await _owned_prompt(user_id, id)
    _check_reason(prompt, body.reason)
    if body.score is not None and body.score not in SCORES_BY_KIND.get(
        cast(FeedbackPromptKind, prompt.kind), frozenset()
    ):
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This prompt does not offer that score",
        )
    if is_impersonated_request(request):
        return FeedbackPromptState.from_row(prompt)
    updated = await update_feedback_prompt(
        user_id=user_id,
        prompt_id=id,
        reason=body.reason,
        note=body.note,
        score=body.score,
        dismissed=body.dismissed,
    )
    if updated is None:
        raise _not_found(id)
    if body.has_answer and updated.kind not in _ANSWERED_THROUGH_CHAT_FEEDBACK:
        capture_prompt_answered(user_id, updated)
    return FeedbackPromptState.from_row(updated)


@router.get(
    "/{id}",
    summary="Get feedback prompt",
    dependencies=[Security(requires_user)],
    responses=_NOT_FOUND_RESPONSES,
)
async def get_prompt(
    id: str,
    user_id: Annotated[str, Security(get_user_id)],
) -> FeedbackPromptState:
    """What a chat card needs to tell asked, answered and closed apart."""
    return FeedbackPromptState.from_row(await _owned_prompt(user_id, id))


async def _owned_prompt(user_id: str, prompt_id: str) -> FeedbackPrompt:
    prompt = await get_feedback_prompt(user_id=user_id, prompt_id=prompt_id)
    if prompt is None:
        raise _not_found(prompt_id)
    return prompt


def _check_reason(prompt: FeedbackPrompt, reason: str | None) -> None:
    if reason is None:
        return
    if reason not in REASONS_BY_KIND[cast(FeedbackPromptKind, prompt.kind)]:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{reason!r} is not a reason this prompt offers",
        )


def _not_found(prompt_id: str) -> HTTPException:
    return HTTPException(
        status_code=HTTP_404_NOT_FOUND,
        detail=f"Feedback prompt {prompt_id} not found or access denied",
    )
