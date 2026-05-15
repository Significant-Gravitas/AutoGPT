"""User-facing platform_linking REST routes (JWT auth)."""

import logging
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Path, Security

from backend.data.db_accessors import platform_linking_db
from backend.platform_linking.models import (
    ConfirmLinkResponse,
    ConfirmUserLinkResponse,
    DeleteLinkResponse,
    LinkTokenInfoResponse,
    PlatformLinkInfo,
    PlatformUserLinkInfo,
)
from backend.util.exceptions import (
    LinkAlreadyExistsError,
    LinkFlowMismatchError,
    LinkTokenExpiredError,
    NotAuthorizedError,
    NotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

TokenPath = Annotated[
    str,
    Path(max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
]


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, NotAuthorizedError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, LinkAlreadyExistsError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, LinkTokenExpiredError):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, LinkFlowMismatchError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Internal error.")


@router.get(
    "/tokens/{token}/info",
    response_model=LinkTokenInfoResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Get display info for a link token",
)
async def get_link_token_info_route(token: TokenPath) -> LinkTokenInfoResponse:
    try:
        return await platform_linking_db().get_link_token_info(token)
    except (NotFoundError, LinkTokenExpiredError) as exc:
        raise _translate(exc) from exc


@router.post(
    "/tokens/{token}/confirm",
    response_model=ConfirmLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a SERVER link token (user must be authenticated)",
)
async def confirm_link_token(
    token: TokenPath,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ConfirmLinkResponse:
    try:
        return await platform_linking_db().confirm_server_link(token, user_id)
    except (
        NotFoundError,
        LinkFlowMismatchError,
        LinkTokenExpiredError,
        LinkAlreadyExistsError,
    ) as exc:
        raise _translate(exc) from exc


@router.post(
    "/user-tokens/{token}/confirm",
    response_model=ConfirmUserLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Confirm a USER link token (user must be authenticated)",
)
async def confirm_user_link_token(
    token: TokenPath,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> ConfirmUserLinkResponse:
    try:
        return await platform_linking_db().confirm_user_link(token, user_id)
    except (
        NotFoundError,
        LinkFlowMismatchError,
        LinkTokenExpiredError,
        LinkAlreadyExistsError,
    ) as exc:
        raise _translate(exc) from exc


@router.get(
    "/links",
    response_model=list[PlatformLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all platform servers linked to the authenticated user",
)
async def list_my_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformLinkInfo]:
    return await platform_linking_db().list_server_links(user_id)


@router.get(
    "/user-links",
    response_model=list[PlatformUserLinkInfo],
    dependencies=[Security(auth.requires_user)],
    summary="List all DM links for the authenticated user",
)
async def list_my_user_links(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> list[PlatformUserLinkInfo]:
    return await platform_linking_db().list_user_links(user_id)


@router.delete(
    "/links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a platform server",
)
async def delete_link(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    try:
        return await platform_linking_db().delete_server_link(link_id, user_id)
    except (NotFoundError, NotAuthorizedError) as exc:
        raise _translate(exc) from exc


@router.delete(
    "/user-links/{link_id}",
    response_model=DeleteLinkResponse,
    dependencies=[Security(auth.requires_user)],
    summary="Unlink a DM / user link",
)
async def delete_user_link_route(
    link_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> DeleteLinkResponse:
    try:
        return await platform_linking_db().delete_user_link(link_id, user_id)
    except (NotFoundError, NotAuthorizedError) as exc:
        raise _translate(exc) from exc
