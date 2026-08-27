"""Agent-graph grant routes (share-with-team), nested under /api/orgs/{org_id}."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from autogpt_libs.auth import requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction, check_org_permission
from fastapi import APIRouter, HTTPException, Security

from backend.api.live_auth import requires_live_resource_permission
from backend.data.tenancy import live_org_context_barrier
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from . import grant_db
from .grant_model import CreateGrantRequest, GrantResponse, ReceivedGrantResponse

router = APIRouter()


@asynccontextmanager
async def _live_share_action(ctx: RequestContext, org_id: str) -> AsyncIterator[bool]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    async with live_org_context_barrier(ctx.user_id, org_id) as live_ctx:
        if live_ctx is None or not check_org_permission(
            live_ctx, OrgAction.SHARE_RESOURCES
        ):
            raise HTTPException(403, detail="Resource sharing access was revoked")
        yield live_ctx.is_org_admin or live_ctx.is_org_owner


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
    async with _live_share_action(ctx, org_id) as is_org_admin:
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
                sharer_is_org_admin=is_org_admin,
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
    async with _live_share_action(ctx, org_id) as is_org_admin:
        try:
            return await grant_db.list_grants_for_graph(
                org_id,
                graph_id,
                requested_by_user_id=ctx.user_id,
                requester_is_org_admin=is_org_admin,
            )
        except NotFoundError as error:
            raise HTTPException(404, detail=str(error))
        except NotAuthorizedError as error:
            raise HTTPException(403, detail=str(error))


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
    async with _live_share_action(ctx, org_id) as is_org_admin:
        try:
            await grant_db.revoke_grant(
                org_id,
                graph_id,
                grant_id,
                revoked_by_user_id=ctx.user_id,
                revoker_is_org_admin=is_org_admin,
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
    ctx: Annotated[
        RequestContext,
        requires_live_resource_permission(
            OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS
        ),
    ],
) -> list[ReceivedGrantResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await grant_db.list_received_grants(org_id, ctx.user_id)
