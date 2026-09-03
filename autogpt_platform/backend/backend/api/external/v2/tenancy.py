"""
V2 External API - Tenancy resolution

Every v2 request acts inside exactly one organization, and optionally inside one
team within it. The tenant is bound to the credential: an API key is minted in
an org (`organizationId`) and may be pinned to a team (`teamIdRestriction`),
which is the same pair v1 external already honours. A key with no org — every
key minted before orgs existed — resolves to the caller's personal org, so
personal keys keep behaving exactly as they did.

A key that is not pinned to a team may select one per request with `X-Team-Id`,
matching the header the internal API uses. Without it the request runs at
org-home, which is what the web app does when no team is selected.

Handlers depend on `require_permission`/`require_auth` from this module rather
than from `..middleware`, so no v2 endpoint can reach the database without a
membership-verified tenant.
"""

import logging
from typing import Literal, Optional

from fastapi import HTTPException, Request, Security
from prisma.enums import APIKeyPermission
from pydantic import BaseModel
from starlette import status

from backend.api.external import middleware
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.db import prisma
from backend.util.exceptions import NotAuthorizedError

logger = logging.getLogger(__name__)

TEAM_HEADER_NAME = "X-Team-Id"


class TenantContext(BaseModel):
    """An authenticated caller with its organization and team resolved.

    Deliberately not an `APIAuthorizationInfo` subclass: `organization_id` is
    optional there and always resolved here, and a handler has no business
    reading a credential's hash metadata.
    """

    user_id: str
    scopes: list[APIKeyPermission]
    type: Literal["oauth", "api_key"]
    organization_id: str
    team_id: Optional[str] = None


async def require_auth(
    request: Request,
    auth: APIAuthorizationInfo = Security(middleware.require_auth),
) -> TenantContext:
    """Require valid credentials, then resolve the tenant they act in.

    Every v2 route reaches this one dependency, so tests override it once and
    `tenancy_test` can assert no handler bypasses it.
    """
    return await resolve_tenant(request, auth)


def require_permission(*permissions: APIKeyPermission):
    """Require the given scopes, and resolve the credential's tenant."""

    async def check_permissions(
        tenant: TenantContext = Security(
            require_auth, scopes=[p.value for p in permissions]
        ),
    ) -> TenantContext:
        if missing := [p for p in permissions if p not in tenant.scopes]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Missing required permission(s): "
                f"{', '.join(p.value for p in missing)}",
            )
        return tenant

    return check_permissions


async def resolve_tenant(request: Request, auth: APIAuthorizationInfo) -> TenantContext:
    """Resolve the org and team this request acts in, verifying membership."""
    organization_id, team_id = await resolve_credential_tenancy(auth)

    if requested_team := request.headers.get(TEAM_HEADER_NAME, "").strip():
        if auth.team_id_restriction and requested_team != auth.team_id_restriction:
            raise NotAuthorizedError(
                f"These credentials are restricted to team {auth.team_id_restriction}"
            )
        await _assert_active_team_member(auth.user_id, organization_id, requested_team)
        team_id = requested_team

    return TenantContext(
        user_id=auth.user_id,
        scopes=auth.scopes,
        type=auth.type,
        organization_id=organization_id,
        team_id=team_id,
    )


async def resolve_credential_tenancy(
    auth: APIAuthorizationInfo,
) -> tuple[str, Optional[str]]:
    """The org and team a credential is bound to, independent of any request.

    Also used by the MCP server, which authenticates outside the dependency
    chain and so cannot go through `resolve_tenant`.
    """
    if auth.organization_id:
        await _assert_active_org_member(auth.user_id, auth.organization_id)
        return auth.organization_id, auth.team_id_restriction
    return await _personal_tenancy(auth.user_id)


async def _assert_active_org_member(user_id: str, organization_id: str) -> None:
    member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": organization_id, "userId": user_id}},
        include={"Org": True},
    )
    if member is None or member.status != "ACTIVE":
        raise NotAuthorizedError(
            "These credentials belong to an organization you are not an active "
            "member of"
        )
    if member.Org is not None and member.Org.deletedAt is not None:
        raise NotAuthorizedError("This organization has been deleted")


async def _assert_active_team_member(
    user_id: str, organization_id: str, team_id: str
) -> None:
    member = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": team_id, "userId": user_id}},
        include={"Team": True},
    )
    if member is None or member.status != "ACTIVE":
        raise NotAuthorizedError(f"You are not an active member of team {team_id}")
    if member.Team is None or member.Team.orgId != organization_id:
        raise NotAuthorizedError(
            f"Team {team_id} does not belong to the organization these credentials "
            "act in"
        )


async def _personal_tenancy(user_id: str) -> tuple[str, Optional[str]]:
    """Org and team for a credential minted before orgs, or outside one.

    Same pair v1 external falls back to, and it bootstraps a missing personal
    org rather than leaving the account permanently unable to call the API.
    """
    from backend.api.features.orgs.db import get_user_default_team

    organization_id, team_id = await get_user_default_team(user_id)
    if organization_id is None:
        logger.warning(
            f"User {user_id} has no personal org and bootstrap failed — "
            "account in inconsistent state"
        )
        raise NotAuthorizedError(
            "No organization context available. Your account may be in an "
            "inconsistent state — please contact support."
        )
    return organization_id, team_id
