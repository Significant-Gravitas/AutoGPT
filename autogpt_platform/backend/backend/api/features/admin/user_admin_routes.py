import logging

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, File, Security, UploadFile

from backend.data.invited_user import (
    BulkInvitedUsersResult,
    InvitedUserRecord,
    bulk_create_invited_users_from_file,
    create_invited_user,
    list_invited_users,
    retry_invited_user_tally,
    revoke_invited_user,
)

from .model import (
    BulkInvitedUserRowResponse,
    BulkInvitedUsersResponse,
    CreateInvitedUserRequest,
    InvitedUserResponse,
    InvitedUsersResponse,
)

logger = logging.getLogger(__name__)


def _redact_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:2]}***@{domain}" if domain else f"{local[:2]}***"


router = APIRouter(
    prefix="/admin",
    tags=["users", "admin"],
    dependencies=[Security(requires_admin_user)],
)


def _to_response(invited_user: InvitedUserRecord) -> InvitedUserResponse:
    return InvitedUserResponse(**invited_user.model_dump())


def _to_bulk_response(result: BulkInvitedUsersResult) -> BulkInvitedUsersResponse:
    return BulkInvitedUsersResponse(
        created_count=result.created_count,
        skipped_count=result.skipped_count,
        error_count=result.error_count,
        results=[
            BulkInvitedUserRowResponse(
                row_number=row.row_number,
                email=row.email,
                name=row.name,
                status=row.status,
                message=row.message,
                invited_user=(
                    _to_response(row.invited_user)
                    if row.invited_user is not None
                    else None
                ),
            )
            for row in result.results
        ],
    )


@router.get(
    "/invited-users",
    response_model=InvitedUsersResponse,
    summary="List Invited Users",
)
async def get_invited_users(
    admin_user_id: str = Security(get_user_id),
) -> InvitedUsersResponse:
    logger.info("Admin user %s requested invited users", admin_user_id)
    invited_users = await list_invited_users()
    return InvitedUsersResponse(
        invited_users=[_to_response(invited_user) for invited_user in invited_users]
    )


@router.post(
    "/invited-users",
    response_model=InvitedUserResponse,
    summary="Create Invited User",
)
async def create_invited_user_route(
    request: CreateInvitedUserRequest,
    admin_user_id: str = Security(get_user_id),
) -> InvitedUserResponse:
    logger.info(
        "Admin user %s creating invited user for %s",
        admin_user_id,
        _redact_email(request.email),
    )
    invited_user = await create_invited_user(request.email, request.name)
    logger.info(
        "Admin user %s created invited user %s",
        admin_user_id,
        invited_user.id,
    )
    return _to_response(invited_user)


@router.post(
    "/invited-users/bulk",
    response_model=BulkInvitedUsersResponse,
    summary="Bulk Create Invited Users",
    operation_id="postV2BulkCreateInvitedUsers",
)
async def bulk_create_invited_users_route(
    file: UploadFile = File(...),
    admin_user_id: str = Security(get_user_id),
) -> BulkInvitedUsersResponse:
    logger.info(
        "Admin user %s bulk invited users from %s",
        admin_user_id,
        file.filename or "<unnamed>",
    )
    content = await file.read()
    result = await bulk_create_invited_users_from_file(file.filename, content)
    return _to_bulk_response(result)


@router.post(
    "/invited-users/{invited_user_id}/revoke",
    response_model=InvitedUserResponse,
    summary="Revoke Invited User",
)
async def revoke_invited_user_route(
    invited_user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> InvitedUserResponse:
    logger.info(
        "Admin user %s revoking invited user %s", admin_user_id, invited_user_id
    )
    invited_user = await revoke_invited_user(invited_user_id)
    logger.info("Admin user %s revoked invited user %s", admin_user_id, invited_user_id)
    return _to_response(invited_user)


@router.post(
    "/invited-users/{invited_user_id}/retry-tally",
    response_model=InvitedUserResponse,
    summary="Retry Invited User Tally",
)
async def retry_invited_user_tally_route(
    invited_user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> InvitedUserResponse:
    logger.info(
        "Admin user %s retrying Tally seed for invited user %s",
        admin_user_id,
        invited_user_id,
    )
    invited_user = await retry_invited_user_tally(invited_user_id)
    logger.info(
        "Admin user %s retried Tally seed for invited user %s",
        admin_user_id,
        invited_user_id,
    )
    return _to_response(invited_user)
