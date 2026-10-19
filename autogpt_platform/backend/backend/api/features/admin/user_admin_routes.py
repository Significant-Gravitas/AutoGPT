"""Admin endpoints backing the user view on the admin users page."""

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, Security

from backend.api.features.orgs import db as org_db
from backend.api.features.orgs.model import OrgResponse

router = APIRouter(
    prefix="/admin/users",
    dependencies=[Security(requires_admin_user)],
)


@router.get(
    "/{user_id}/organizations",
    summary="Get user organizations",
)
async def get_user_organizations(user_id: str) -> list[OrgResponse]:
    """The organizations a user is an active member of. Admin-only."""
    return await org_db.list_user_orgs(user_id)
