"""The raise dialog's fast path.

The dialog asks for a name and a role and opens the new expert's page at
once. These routes back what happens there: the expert's opening line and
questions, fetched while the user is still typing, and the soul it writes
from the answers in the background.
"""

from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Security

from backend.api.features.experts import experts_db, raise_intro, raise_soul
from backend.api.features.experts.raise_intro import RaiseIntro, RaiseIntroRequest
from backend.api.features.experts.raise_soul import SoulDraftQueued, SoulDraftRequest
from backend.api.utils.rate_limit import RateLimiter
from backend.copilot.rate_limit import enforce_payment_paywall

router = APIRouter()

# Each call can reach a model. The dialog asks for an opening a few times per
# expert while the user types, and a soul is drafted once per expert.
_intro_limiter = RateLimiter(
    "experts:raise-intro", max_requests=60, window_seconds=3600
)
_soul_limiter = RateLimiter("experts:soul-draft", max_requests=20, window_seconds=3600)


@router.get(
    "/raise/intro",
    operation_id="get_raise_intro",
    dependencies=[Depends(enforce_payment_paywall)],
    responses={429: {"description": "Too many openings requested"}},
)
async def get_raise_intro(
    request: Annotated[RaiseIntroRequest, Query()],
    user_id: str = Security(auth.get_user_id),
) -> RaiseIntro:
    """What a new expert says first, inferred from its name and role.

    Stateless and never fails: a slow or missing model answers with a
    template so the expert page always has an opening.
    """
    await _intro_limiter.check(user_id)
    return await raise_intro.generate_intro(request)


@router.post(
    "/{expert_id}/soul/draft",
    operation_id="draft_expert_soul",
    status_code=202,
    dependencies=[Depends(enforce_payment_paywall)],
    responses={
        404: {"description": "Expert not found"},
        429: {"description": "Too many drafts requested"},
    },
)
async def draft_expert_soul(
    expert_id: str,
    request: SoulDraftRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Security(auth.get_user_id),
) -> SoulDraftQueued:
    """Write the expert's soul from its opening answers, in the background.

    The write only lands if the soul is unchanged by then, so an edit the
    user makes meanwhile always wins.
    """
    await _soul_limiter.check(user_id)
    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    if expert is None:
        raise HTTPException(status_code=404, detail="Expert not found")
    background_tasks.add_task(raise_soul.write_soul, user_id, expert, request)
    return SoulDraftQueued(expected_identity=expert.identity)
