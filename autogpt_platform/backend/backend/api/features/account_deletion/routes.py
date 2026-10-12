"""Self-serve account deletion (SECRT-2877).

Both routes 404 while the self-serve-account-deletion flag is off for the
caller. The deletion route authenticates with ``get_user_id_allowing_closed``
so a repeat call from an account it already closed still answers 200; every
other route rejects a closed account. Neither route sits behind the payment
paywall: an account without a plan must be able to delete itself.
"""

from typing import Annotated

from autogpt_libs.auth import get_user_id, get_user_id_allowing_closed
from fastapi import APIRouter, Depends, HTTPException, Security, status
from prisma.enums import AccountDeletionSource

from backend.api.features.account_deletion.feedback import record_feedback
from backend.api.features.account_deletion.models import (
    AccountDeletionRequestBody,
    AccountDeletionResponse,
    DeletionFeedbackRequest,
)
from backend.api.features.account_deletion.service import (
    EmailMismatchError,
    PlanCancellationError,
    delete_account,
)
from backend.util.feature_flag import Flag, is_feature_enabled

EMAIL_MISMATCH_DETAIL = "That email doesn't match your account."
PLAN_CANCELLATION_DETAIL = (
    "We couldn't cancel your plan. Nothing was deleted. Try again."
)


async def require_account_deletion_flag(
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> None:
    """Fail-closed gate, read through ``is_feature_enabled`` so the
    ``FORCE_FLAG_*`` override works locally (see onboarding_dump's gate)."""
    if not await is_feature_enabled(Flag.SELF_SERVE_ACCOUNT_DELETION, user_id):
        raise HTTPException(status_code=404, detail="Feature not available")


router = APIRouter(
    prefix="/account/deletion",
    tags=["account"],
    dependencies=[Depends(require_account_deletion_flag)],
)


@router.post(
    "",
    summary="Delete account",
    responses={
        400: {"description": EMAIL_MISMATCH_DETAIL},
        404: {"description": "The self-serve-account-deletion flag is off"},
        502: {"description": PLAN_CANCELLATION_DETAIL},
    },
)
async def request_account_deletion(
    body: AccountDeletionRequestBody,
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> AccountDeletionResponse:
    try:
        await delete_account(
            user_id=user_id,
            confirm_email=body.confirm_email,
            source=AccountDeletionSource(body.source.upper()),
        )
    except EmailMismatchError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, EMAIL_MISMATCH_DETAIL)
    except PlanCancellationError:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, PLAN_CANCELLATION_DETAIL)
    return AccountDeletionResponse(status="scheduled")


@router.post(
    "/feedback",
    summary="Send exit feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"description": "The self-serve-account-deletion flag is off"},
        429: {"description": "Too many answers from this account"},
    },
)
async def send_account_deletion_feedback(
    body: DeletionFeedbackRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    await record_feedback(user_id=user_id, body=body)
