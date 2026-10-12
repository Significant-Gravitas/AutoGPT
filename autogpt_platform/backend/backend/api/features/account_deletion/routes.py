"""Self-serve account deletion (SECRT-2877).

Both routes 404 while the self-serve-account-deletion flag is off for the
caller, and answer 403 to an admin acting as another user: closing an account
is its owner's own act. The deletion route authenticates with
``get_user_id_allowing_closed`` so a repeat call from an account it already
closed still answers 200; every other route rejects a closed account. Neither
route sits behind the payment paywall: an account without a plan must be able
to delete itself.
"""

from typing import Annotated

from autogpt_libs.auth import get_user_id, get_user_id_allowing_closed
from autogpt_libs.auth.dependencies import IMPERSONATION_HEADER_NAME
from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
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
IMPERSONATION_DETAIL = "Account deletion isn't available while impersonating a user."


async def require_account_deletion_flag(
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> None:
    """Fail-closed gate, read through ``is_feature_enabled`` so the
    ``FORCE_FLAG_*`` override works locally (see onboarding_dump's gate)."""
    if not await is_feature_enabled(Flag.SELF_SERVE_ACCOUNT_DELETION, user_id):
        raise HTTPException(status_code=404, detail="Feature not available")


async def reject_impersonation(request: Request) -> None:
    """An admin acting as a user can neither close their account nor answer
    the exit question for them. After the flag gate, so a flag that is off
    still hides both routes from everyone."""
    if request.headers.get(IMPERSONATION_HEADER_NAME, "").strip():
        raise HTTPException(status.HTTP_403_FORBIDDEN, IMPERSONATION_DETAIL)


router = APIRouter(
    prefix="/account/deletion",
    tags=["account"],
    dependencies=[
        Depends(require_account_deletion_flag),
        Depends(reject_impersonation),
    ],
)


@router.post(
    "",
    summary="Delete account",
    responses={
        400: {"description": EMAIL_MISMATCH_DETAIL},
        403: {"description": IMPERSONATION_DETAIL},
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
        403: {"description": IMPERSONATION_DETAIL},
        404: {"description": "The self-serve-account-deletion flag is off"},
        429: {"description": "Too many answers from this account"},
    },
)
async def send_account_deletion_feedback(
    body: DeletionFeedbackRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    await record_feedback(user_id=user_id, body=body)
