import logging
import math

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, File, HTTPException, Query, Security, UploadFile

from backend.copilot.autopilot import trigger_autopilot_session_for_user
from backend.data.invited_user import (
    bulk_create_invited_users_from_file,
    create_invited_user,
    list_invited_users,
    retry_invited_user_tally,
    revoke_invited_user,
)
from backend.data.tally import mask_email
from backend.data.user import search_users
from backend.util.models import Pagination

from .model import (
    AdminCopilotUsersResponse,
    AdminCopilotUserSummary,
    BulkInvitedUsersResponse,
    CreateInvitedUserRequest,
    InvitedUserResponse,
    InvitedUsersResponse,
    TriggerCopilotSessionRequest,
    TriggerCopilotSessionResponse,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin",
    tags=["users", "admin"],
    dependencies=[Security(requires_admin_user)],
)


@router.get(
    "/invited-users",
    response_model=InvitedUsersResponse,
    summary="List Invited Users",
)
async def get_invited_users(
    admin_user_id: str = Security(get_user_id),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> InvitedUsersResponse:
    logger.info("Admin user %s requested invited users", admin_user_id)
    invited_users, total = await list_invited_users(page=page, page_size=page_size)
    return InvitedUsersResponse(
        invited_users=[InvitedUserResponse.from_record(iu) for iu in invited_users],
        pagination=Pagination(
            total_items=total,
            total_pages=max(1, math.ceil(total / page_size)),
            current_page=page,
            page_size=page_size,
        ),
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
        mask_email(request.email),
    )
    invited_user = await create_invited_user(request.email, request.name)
    logger.info(
        "Admin user %s created invited user %s",
        admin_user_id,
        invited_user.id,
    )
    return InvitedUserResponse.from_record(invited_user)


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
    return BulkInvitedUsersResponse.from_result(result)


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
    return InvitedUserResponse.from_record(invited_user)


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
    return InvitedUserResponse.from_record(invited_user)


@router.get(
    "/copilot/users",
    response_model=AdminCopilotUsersResponse,
    summary="Search Copilot Users",
    operation_id="getV2SearchCopilotUsers",
)
async def search_copilot_users_route(
    search: str = Query("", description="Search by email, name, or user ID"),
    limit: int = Query(20, ge=1, le=50),
    admin_user_id: str = Security(get_user_id),
) -> AdminCopilotUsersResponse:
    logger.info(
        "Admin user %s searched Copilot users (query_length=%s, limit=%s)",
        admin_user_id,
        len(search.strip()),
        limit,
    )
    users = await search_users(search, limit=limit)
    return AdminCopilotUsersResponse(
        users=[AdminCopilotUserSummary.from_user(user) for user in users]
    )


@router.post(
    "/copilot/trigger",
    response_model=TriggerCopilotSessionResponse,
    summary="Trigger Copilot Session",
    operation_id="postV2TriggerCopilotSession",
)
async def trigger_copilot_session_route(
    request: TriggerCopilotSessionRequest,
    admin_user_id: str = Security(get_user_id),
) -> TriggerCopilotSessionResponse:
    logger.info(
        "Admin user %s manually triggered %s for user %s",
        admin_user_id,
        request.start_type,
        request.user_id,
    )
    try:
        session = await trigger_autopilot_session_for_user(
            request.user_id,
            start_type=request.start_type,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "Admin user %s created manual Copilot session %s for user %s",
        admin_user_id,
        session.session_id,
        request.user_id,
    )
    return TriggerCopilotSessionResponse(
        session_id=session.session_id,
        start_type=request.start_type,
    )
