"""V2 External API - Identity Endpoint"""

import logging

from fastapi import APIRouter, Security
from prisma.enums import APIKeyPermission

from backend.api.features.orgs.db import get_org
from backend.data.db import prisma
from backend.data.user import get_user_by_id

from .models import Identity, TenantOrganization, TenantTeam
from .tenancy import TenantContext, require_permission

logger = logging.getLogger(__name__)

identity_router = APIRouter(tags=["identity"])


@identity_router.get(
    path="/me",
    summary="Get the identity and tenancy of the current credentials",
    operation_id="getIdentity",
)
async def get_identity(
    auth: TenantContext = Security(require_permission(APIKeyPermission.IDENTITY)),
) -> Identity:
    """Get the user, organization and team these credentials act as.

    The organization is fixed when the credentials are created; `team` reflects
    the team they are pinned to, or the one selected with `X-Team-Id`.
    """
    user = await get_user_by_id(auth.user_id)
    org = await get_org(auth.organization_id)

    team = None
    if auth.team_id and (
        row := await prisma.team.find_unique(where={"id": auth.team_id})
    ):
        team = TenantTeam(id=row.id, name=row.name)

    return Identity(
        user_id=user.id,
        email=user.email,
        name=user.name,
        timezone=user.timezone,
        organization=TenantOrganization(
            id=org.id, name=org.name, is_personal=org.is_personal
        ),
        team=team,
        scopes=[scope.value for scope in auth.scopes],
        credential_type=auth.type,
    )
