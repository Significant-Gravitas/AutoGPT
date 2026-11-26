"""The signed-in user's own OAuth grants: the apps they let into their account."""

import logging

from autogpt_libs.auth import get_user_id
from fastapi import APIRouter, HTTPException, Security, status

from backend.api.features.oauth import push_shim_revocation
from backend.data.auth.oauth import get_oauth_application_by_id
from backend.data.auth.oauth_grants import (
    OAuthGrantInfo,
    list_user_oauth_grants,
    revoke_user_oauth_grant,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/grants")
async def list_my_oauth_grants(
    user_id: str = Security(get_user_id),
) -> list[OAuthGrantInfo]:
    """List the apps that hold access to the current user's account."""
    return await list_user_oauth_grants(user_id)


@router.delete(
    "/grants/{app_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "The app holds no access to this account"}},
)
async def revoke_my_oauth_grant(
    app_id: str,
    user_id: str = Security(get_user_id),
) -> None:
    """End every access and refresh token one app holds for the current user."""
    if not await revoke_user_oauth_grant(user_id, app_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This app has no access to your account",
        )
    logger.info(f"User #{user_id} revoked the OAuth grant of app #{app_id}")
    if app := await get_oauth_application_by_id(app_id):
        await push_shim_revocation(user_id, app.client_id)
