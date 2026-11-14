"""Self-serve account deletion (SECRT-2877).

Every signed-in user can delete their account. Deleting, the feedback and
keeping the account answer 403 to an admin acting as another user: each is
its owner's own act. Reading the deletion status does not.

The deletion, status and keep routes authenticate with
``get_user_id_allowing_closed``: they answer a closed account (a repeat call,
the screen shown at sign-in during the 14 days) and a banned one (deleting
the account is its owner's right whatever the ban; keeping it is not, so the
keep refuses a banned account as every other route does). Every other route
rejects both. None of them sits behind the payment paywall: an account without a plan
must be able to delete itself.
"""

from typing import Annotated

from autogpt_libs.auth import get_user_id, get_user_id_allowing_closed
from autogpt_libs.auth.dependencies import (
    ACCOUNT_BANNED_DETAIL,
    IMPERSONATION_HEADER_NAME,
)
from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
from prisma.enums import AccountDeletionSource

from backend.api.features.account_deletion.feedback import record_feedback
from backend.api.features.account_deletion.models import (
    AccountDeletionRequestBody,
    AccountDeletionResponse,
    DeletionFeedbackRequest,
)
from backend.api.features.account_deletion.restore import (
    BannedAccountError,
    TooLateToRestoreError,
    get_deletion_status,
    restore_account,
)
from backend.api.features.account_deletion.service import (
    EmailMismatchError,
    PlanCancellationError,
    delete_account,
)

EMAIL_MISMATCH_DETAIL = "That email doesn't match your account."
PLAN_CANCELLATION_DETAIL = (
    "We couldn't cancel your plan. Nothing was deleted. Try again."
)
IMPERSONATION_DETAIL = "Account deletion isn't available while impersonating a user."
TOO_LATE_DETAIL = (
    "It's too late to keep this account: its data is being erased. Contact "
    "contact@agpt.co if you need help."
)


async def reject_impersonation(request: Request) -> None:
    """An admin acting as a user can neither close their account nor answer
    the exit question for them."""
    if request.headers.get(IMPERSONATION_HEADER_NAME, "").strip():
        raise HTTPException(status.HTTP_403_FORBIDDEN, IMPERSONATION_DETAIL)


router = APIRouter(
    prefix="/account/deletion",
    tags=["account"],
    dependencies=[Depends(reject_impersonation)],
)


@router.post(
    "",
    summary="Delete account",
    responses={
        400: {"description": EMAIL_MISMATCH_DETAIL},
        403: {"description": IMPERSONATION_DETAIL},
        502: {"description": PLAN_CANCELLATION_DETAIL},
    },
)
async def request_account_deletion(
    body: AccountDeletionRequestBody,
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> AccountDeletionResponse:
    try:
        erase_after = await delete_account(
            user_id=user_id,
            confirm_email=body.confirm_email,
            source=AccountDeletionSource(body.source.upper()),
        )
    except EmailMismatchError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, EMAIL_MISMATCH_DETAIL)
    except PlanCancellationError:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, PLAN_CANCELLATION_DETAIL)
    return AccountDeletionResponse(status="scheduled", erase_after=erase_after)


@router.post(
    "/feedback",
    summary="Send exit feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        403: {"description": IMPERSONATION_DETAIL},
        429: {"description": "Too many answers from this account"},
    },
)
async def send_account_deletion_feedback(
    body: DeletionFeedbackRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> None:
    await record_feedback(user_id=user_id, body=body)


# Reading the status is open to an admin acting as the user; keeping the
# account refuses them on its own route.
status_router = APIRouter(prefix="/account/deletion", tags=["account"])


@status_router.get("", summary="Get account deletion status")
async def get_account_deletion_status(
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> AccountDeletionResponse:
    return await get_deletion_status(user_id)


@status_router.post(
    "/restore",
    summary="Keep my account",
    dependencies=[Depends(reject_impersonation)],
    responses={
        403: {
            "description": f"{IMPERSONATION_DETAIL} Or the account is banned "
            "(`account_banned`)."
        },
        409: {"description": TOO_LATE_DETAIL},
    },
)
async def keep_account(
    user_id: Annotated[str, Security(get_user_id_allowing_closed)],
) -> AccountDeletionResponse:
    try:
        await restore_account(user_id)
    except BannedAccountError:
        raise HTTPException(status.HTTP_403_FORBIDDEN, ACCOUNT_BANNED_DETAIL)
    except TooLateToRestoreError:
        raise HTTPException(status.HTTP_409_CONFLICT, TOO_LATE_DETAIL)
    return AccountDeletionResponse(status="open")
