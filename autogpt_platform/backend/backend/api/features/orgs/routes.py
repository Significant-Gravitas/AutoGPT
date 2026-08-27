"""Organization management API routes."""

import mimetypes
import os
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from typing import IO, Annotated

from autogpt_libs.auth import get_user_id, requires_org_permission, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, Depends, HTTPException, Query, Security, UploadFile
from fastapi.responses import StreamingResponse

from backend.api.features.store import exceptions as store_exceptions
from backend.api.features.store import media as store_media
from backend.api.live_auth import requires_live_org_permission
from backend.data.org_credit import get_org_spend_by_team
from backend.data.tenancy import live_org_permission_barrier

from . import db as org_db
from .model import (
    AddMemberRequest,
    CreateAliasRequest,
    CreateOrgRequest,
    OrgAliasResponse,
    OrgMemberResponse,
    OrgResponse,
    OrgSpendResponse,
    TeamSpendBucket,
    TransferOwnershipRequest,
    UpdateMemberRequest,
    UpdateOrgData,
    UpdateOrgRequest,
)

router = APIRouter()


def _stream_open_file(file: IO[bytes]) -> Iterator[bytes]:
    try:
        while chunk := file.read(64 * 1024):
            yield chunk
    finally:
        file.close()


def _verify_org_path(ctx: RequestContext, org_id: str) -> None:
    """Ensure the authenticated user's active org matches the path parameter.

    Prevents authorization bypass where a user sends X-Org-Id for org A
    but targets org B in the URL path.
    """
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")


@router.post(
    "",
    summary="Create organization",
    tags=["orgs"],
    dependencies=[Security(requires_user)],
)
async def create_org(
    request: CreateOrgRequest,
    user_id: Annotated[str, Security(get_user_id)],
) -> OrgResponse:
    return await org_db.create_org(
        name=request.name,
        slug=request.slug,
        user_id=user_id,
        description=request.description,
    )


@router.get(
    "",
    summary="List user organizations",
    tags=["orgs"],
    dependencies=[Security(requires_user)],
)
async def list_orgs(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[OrgResponse]:
    return await org_db.list_user_orgs(user_id)


@router.get(
    "/{org_id}",
    summary="Get organization details",
    tags=["orgs"],
)
async def get_org(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        requires_live_org_permission(OrgAction.VIEW_ORG),
    ],
) -> OrgResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await org_db.get_org(org_id)


@router.patch(
    "/{org_id}",
    summary="Update organization",
    tags=["orgs"],
)
async def update_org(
    org_id: str,
    request: UpdateOrgRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.RENAME_ORG)),
    ],
) -> OrgResponse:
    _verify_org_path(ctx, org_id)
    return await org_db.update_org(
        org_id,
        UpdateOrgData(
            name=request.name,
            slug=request.slug,
            description=request.description,
            avatar_url=request.avatar_url,
            memory_hold_buffer=request.memory_hold_buffer,
        ),
        actor_user_id=ctx.user_id,
    )


_AVATAR_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_AVATAR_EXTENSIONS_BY_CONTENT_TYPE = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/gif": {".gif"},
    "image/webp": {".webp"},
}


async def _require_org_avatar_access(
    org_id: str,
    user_id: Annotated[str, Security(get_user_id)],
) -> AsyncIterator[None]:
    async with live_org_permission_barrier(
        user_id, org_id, OrgAction.VIEW_ORG
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Not a member of this organization")
        yield


@router.post(
    "/{org_id}/avatar",
    summary="Upload organization avatar",
    tags=["orgs"],
)
async def upload_org_avatar(
    org_id: str,
    file: UploadFile,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.RENAME_ORG)),
    ],
) -> OrgResponse:
    """Upload an avatar image for the organization and persist its URL.

    The storage path is derived server-side from the verified org id; the
    client-supplied filename is only used for extension validation.
    """
    _verify_org_path(ctx, org_id)
    async with live_org_permission_barrier(
        ctx.user_id, org_id, OrgAction.RENAME_ORG
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Organization access was revoked")
        if file.content_type not in store_media.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                400,
                detail=(
                    "Avatar must be an image; allowed content types: "
                    f"{', '.join(sorted(store_media.ALLOWED_IMAGE_TYPES))}"
                ),
            )
        extension = os.path.splitext(file.filename or "")[1].lower()
        if extension not in _AVATAR_ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                detail=(
                    "Avatar file extension must be one of: "
                    f"{', '.join(sorted(_AVATAR_ALLOWED_EXTENSIONS))}"
                ),
            )
        if extension not in _AVATAR_EXTENSIONS_BY_CONTENT_TYPE[file.content_type]:
            raise HTTPException(
                400,
                detail="Avatar file extension does not match its content type",
            )

        try:
            avatar_url = await store_media.upload_media(
                user_id=ctx.user_id, file=file, organization_id=org_id
            )
        except store_exceptions.MediaUploadError as e:
            raise HTTPException(400, detail=str(e)) from e

    return await org_db.update_org(
        org_id,
        UpdateOrgData(avatar_url=avatar_url),
        actor_user_id=ctx.user_id,
    )


@router.get(
    "/{org_id}/avatar/{filename}",
    summary="Get organization avatar",
    tags=["orgs"],
    response_class=StreamingResponse,
    dependencies=[Depends(_require_org_avatar_access)],
)
async def get_org_avatar(
    org_id: str,
    filename: str,
) -> StreamingResponse:
    normalized_org_id = org_id.replace("\\", "/")
    safe_org_id = os.path.basename(normalized_org_id)
    normalized_filename = filename.replace("\\", "/")
    safe_filename = os.path.basename(normalized_filename)
    if (
        not safe_org_id
        or safe_org_id != normalized_org_id
        or not safe_filename
        or safe_filename != normalized_filename
    ):
        raise HTTPException(404, detail="Avatar not found")
    extension = os.path.splitext(safe_filename)[1].lower()
    if extension not in _AVATAR_ALLOWED_EXTENSIONS:
        raise HTTPException(404, detail="Avatar not found")
    media_root = store_media.get_local_media_root()
    path = os.path.realpath(
        os.path.join(media_root, "orgs", safe_org_id, "images", safe_filename)
    )
    if not path.startswith(os.path.join(media_root, "")):
        raise HTTPException(404, detail="Avatar not found")
    if not os.path.isfile(path):
        raise HTTPException(404, detail="Avatar not found")
    try:
        file = open(path, "rb")
    except OSError as error:
        raise HTTPException(404, detail="Avatar not found") from error
    return StreamingResponse(
        _stream_open_file(file),
        media_type=mimetypes.guess_type(safe_filename)[0],
        headers={
            "Cache-Control": "private, no-store",
            "Content-Length": str(os.fstat(file.fileno()).st_size),
        },
    )


@router.delete(
    "/{org_id}",
    summary="Delete organization",
    tags=["orgs"],
    status_code=204,
)
async def delete_org(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.DELETE_ORG)),
    ],
) -> None:
    _verify_org_path(ctx, org_id)
    await org_db.delete_org(org_id, actor_user_id=ctx.user_id)


@router.post(
    "/{org_id}/convert",
    summary="Convert personal org to team org",
    tags=["orgs"],
)
async def convert_org(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.DELETE_ORG)),
    ],
) -> OrgResponse:
    _verify_org_path(ctx, org_id)
    return await org_db.convert_personal_org(org_id, ctx.user_id)


# --- Members ---


@router.get(
    "/{org_id}/members",
    summary="List organization members",
    tags=["orgs"],
)
async def list_members(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        requires_live_org_permission(OrgAction.VIEW_ORG),
    ],
) -> list[OrgMemberResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await org_db.list_org_members(org_id)


@router.post(
    "/{org_id}/members",
    summary="Add member to organization",
    tags=["orgs"],
)
async def add_member(
    org_id: str,
    request: AddMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> OrgMemberResponse:
    if request.user_id == ctx.user_id:
        raise HTTPException(409, detail="You are already a member")
    _verify_org_path(ctx, org_id)
    return await org_db.add_org_member(
        org_id=org_id,
        user_id=request.user_id,
        is_admin=request.is_admin,
        is_billing_manager=request.is_billing_manager,
        invited_by=ctx.user_id,
    )


@router.patch(
    "/{org_id}/members/{uid}",
    summary="Update member role",
    tags=["orgs"],
)
async def update_member(
    org_id: str,
    uid: str,
    request: UpdateMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> OrgMemberResponse:
    if uid == ctx.user_id:
        raise HTTPException(400, detail="You cannot change your own organization role")
    _verify_org_path(ctx, org_id)
    return await org_db.update_org_member(
        org_id=org_id,
        user_id=uid,
        is_admin=request.is_admin,
        is_billing_manager=request.is_billing_manager,
        requesting_user_id=ctx.user_id,
    )


@router.delete(
    "/{org_id}/members/{uid}",
    summary="Remove member from organization",
    tags=["orgs"],
    status_code=204,
)
async def remove_member(
    org_id: str,
    uid: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_MEMBERS)),
    ],
) -> None:
    if uid == ctx.user_id:
        raise HTTPException(
            400, detail="You cannot remove yourself from the organization"
        )
    _verify_org_path(ctx, org_id)
    await org_db.remove_org_member(org_id, uid, requesting_user_id=ctx.user_id)


@router.post(
    "/{org_id}/transfer-ownership",
    summary="Transfer organization ownership",
    tags=["orgs"],
)
async def transfer_ownership(
    org_id: str,
    request: TransferOwnershipRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.DELETE_ORG)),
    ],
) -> None:
    _verify_org_path(ctx, org_id)
    await org_db.transfer_ownership(org_id, ctx.user_id, request.new_owner_id)


# --- Spend ---


@router.get(
    "/{org_id}/spend",
    summary="Per-team spend breakdown",
    tags=["orgs"],
)
async def get_org_spend(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        requires_live_org_permission(OrgAction.MANAGE_BILLING),
    ],
    from_time: Annotated[datetime | None, Query(alias="from")] = None,
    to_time: Annotated[datetime | None, Query(alias="to")] = None,
) -> OrgSpendResponse:
    """Credits spent by the org, grouped by the team each debit was attributed to.

    Requires org-level MANAGE_BILLING (owner or billing_manager). Usage with no
    team attribution — org-home spend and legacy personal-org migrations — is
    reported in a single bucket with ``team_id = null``.
    """
    _verify_org_path(ctx, org_id)
    buckets = await get_org_spend_by_team(
        org_id, start_time=from_time, end_time=to_time
    )
    return OrgSpendResponse(teams=[TeamSpendBucket(**bucket) for bucket in buckets])


# --- Aliases ---


@router.get(
    "/{org_id}/aliases",
    summary="List organization aliases",
    tags=["orgs"],
)
async def list_aliases(
    org_id: str,
    ctx: Annotated[
        RequestContext,
        requires_live_org_permission(OrgAction.VIEW_ORG),
    ],
) -> list[OrgAliasResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await org_db.list_org_aliases(org_id)


@router.post(
    "/{org_id}/aliases",
    summary="Create organization alias",
    tags=["orgs"],
)
async def create_alias(
    org_id: str,
    request: CreateAliasRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.RENAME_ORG)),
    ],
) -> OrgAliasResponse:
    _verify_org_path(ctx, org_id)
    return await org_db.create_org_alias(org_id, request.alias_slug, ctx.user_id)
