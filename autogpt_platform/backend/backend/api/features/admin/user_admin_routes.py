"""Admin endpoints backing the user view on the admin users page."""

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, Security

from backend.api.features.orgs import db as org_db
from backend.api.features.orgs.model import OrgResponse
from backend.data.auth_ip_events import AccountIPReport, get_account_ip_report
from backend.util.settings import Settings

settings = Settings()

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


@router.get(
    "/{user_id}/ip-report",
    summary="Get user IP report",
)
async def get_user_ip_report(user_id: str) -> AccountIPReport:
    """Where an account signed up and signed in from, the other accounts on
    those IPs or its email domain, and the blocks covering them. Admin-only."""
    return await get_account_ip_report(
        user_id, settings.config.auth_ip_event_retention_days
    )
