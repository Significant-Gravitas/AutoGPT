"""Organization management API routes."""

from typing import Annotated

from autogpt_libs.auth import (
    get_request_context,
    get_user_id,
    requires_org_permission,
    requires_user,
)
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Security

from . import db as org_db
from .model import (
    AddMemberRequest,
    CreateAliasRequest,
    CreateOrgRequest,
    OrgAliasResponse,
    OrgMemberResponse,
    OrgResponse,
    TransferOwnershipRequest,
    UpdateMemberRequest,
    UpdateOrgRequest,
)

router = APIRouter()


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
    result = await org_db.create_org(
        name=request.name,
        slug=request.slug,
        user_id=user_id,
        description=request.description,
    )
    return OrgResponse(**result)


@router.get(
    "",
    summary="List user organizations",
    tags=["orgs"],
    dependencies=[Security(requires_user)],
)
async def list_orgs(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[OrgResponse]:
    results = await org_db.list_user_orgs(user_id)
    return [OrgResponse(**r) for r in results]


@router.get(
    "/{org_id}",
    summary="Get organization details",
    tags=["orgs"],
)
async def get_org(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> OrgResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    result = await org_db.get_org(org_id)
    return OrgResponse(**result)


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
    result = await org_db.update_org(
        org_id,
        {
            "name": request.name,
            "slug": request.slug,
            "description": request.description,
            "avatarUrl": request.avatarUrl,
        },
    )
    return OrgResponse(**result)


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
    await org_db.delete_org(org_id)


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
    result = await org_db.convert_personal_org(org_id)
    return OrgResponse(**result)


# --- Members ---


@router.get(
    "/{org_id}/members",
    summary="List organization members",
    tags=["orgs"],
)
async def list_members(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[OrgMemberResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    results = await org_db.list_org_members(org_id)
    return [OrgMemberResponse(**r) for r in results]


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
    result = await org_db.add_org_member(
        org_id=org_id,
        user_id=request.userId,
        is_admin=request.isAdmin,
        is_billing_manager=request.isBillingManager,
        invited_by=ctx.user_id,
    )
    return OrgMemberResponse(**result)


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
    result = await org_db.update_org_member(
        org_id=org_id,
        user_id=uid,
        is_admin=request.isAdmin,
        is_billing_manager=request.isBillingManager,
    )
    return OrgMemberResponse(**result)


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
    await org_db.remove_org_member(org_id, uid)


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
    await org_db.transfer_ownership(org_id, ctx.user_id, request.newOwnerId)


# --- Aliases ---


@router.get(
    "/{org_id}/aliases",
    summary="List organization aliases",
    tags=["orgs"],
)
async def list_aliases(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[OrgAliasResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    results = await org_db.list_org_aliases(org_id)
    return [OrgAliasResponse(**r) for r in results]


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
    result = await org_db.create_org_alias(org_id, request.aliasSlug, ctx.user_id)
    return OrgAliasResponse(**result)
