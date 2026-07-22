"""Agent-graph grant routes (share-with-team), nested under /api/orgs/{org_id}."""

from typing import Annotated

from autogpt_libs.auth import get_request_context, requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction
from fastapi import APIRouter, HTTPException, Security

from backend.util.exceptions import NotAuthorizedError, NotFoundError

from . import grant_db
from .grant_model import CreateGrantRequest, GrantResponse, ReceivedGrantResponse

router = APIRouter()


@router.post(
    "/graphs/{graph_id}/grants",
    summary="Share graph with a team",
    tags=["orgs", "grants"],
)
async def create_grant(
    org_id: str,
    graph_id: str,
    request: CreateGrantRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.SHARE_RESOURCES)),
    ],
) -> GrantResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    try:
        return await grant_db.upsert_grant(
            org_id=org_id,
            graph_id=graph_id,
            principal_type=request.principal_type,
            principal_id=request.principal_id,
            graph_version=request.graph_version,
            capability=request.capability,
            credential_mode=request.credential_mode,
            follow_latest=request.follow_latest,
            created_by_user_id=ctx.user_id,
            sharer_is_org_admin=ctx.is_org_admin or ctx.is_org_owner,
        )
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except NotAuthorizedError as e:
        raise HTTPException(403, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@router.get(
    "/graphs/{graph_id}/grants",
    summary="List grants on a graph",
    tags=["orgs", "grants"],
)
async def list_grants(
    org_id: str,
    graph_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.SHARE_RESOURCES)),
    ],
) -> list[GrantResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await grant_db.list_grants_for_graph(org_id, graph_id)


@router.delete(
    "/graphs/{graph_id}/grants/{grant_id}",
    summary="Revoke a grant",
    tags=["orgs", "grants"],
    status_code=204,
)
async def revoke_grant(
    org_id: str,
    graph_id: str,
    grant_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.SHARE_RESOURCES)),
    ],
) -> None:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    try:
        await grant_db.revoke_grant(
            org_id,
            graph_id,
            grant_id,
            revoked_by_user_id=ctx.user_id,
            revoker_is_org_admin=ctx.is_org_admin or ctx.is_org_owner,
        )
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except NotAuthorizedError as e:
        raise HTTPException(403, detail=str(e))
    except ValueError as e:
        raise HTTPException(400, detail=str(e))


@router.get(
    "/grants/received",
    summary="List grants shared with my teams",
    tags=["orgs", "grants"],
)
async def list_received_grants(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[ReceivedGrantResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await grant_db.list_received_grants(org_id, ctx.user_id)
