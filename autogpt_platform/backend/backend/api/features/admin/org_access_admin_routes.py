"""Admin endpoints that turn organizations on or off for one account."""

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel, Field

from backend.data.org_access import (
    OrgAccessGrantPage,
    OrgAccessStatus,
    find_user_id_by_email,
    get_org_access_status,
    grant_org_access,
    list_org_access_grants,
    revoke_org_access,
)

router = APIRouter(
    prefix="/admin/org-access",
    dependencies=[Security(requires_admin_user)],
)


class EnableOrgAccessRequest(BaseModel):
    note: str | None = Field(
        default=None,
        max_length=500,
        description="Why the account gets organizations, e.g. the plan or trial.",
    )


class EnableOrgAccessByEmailRequest(EnableOrgAccessRequest):
    email: str = Field(min_length=3, max_length=320)


@router.get("", summary="List organization access")
async def list_org_access(
    search: str | None = Query(default=None, max_length=320),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> OrgAccessGrantPage:
    """Accounts an admin has given organizations, newest first. Admin-only."""
    return await list_org_access_grants(search, page, page_size)


@router.get("/users/{user_id}", summary="Get organization access")
async def get_org_access(user_id: str) -> OrgAccessStatus:
    """Whether an account has organizations, and through which grant. Admin-only."""
    return await get_org_access_status(user_id)


@router.put("/users/{user_id}", summary="Enable organization access")
async def enable_org_access(
    user_id: str,
    request: EnableOrgAccessRequest,
    admin_user_id: str = Security(get_user_id),
) -> OrgAccessStatus:
    """Give an account organizations; its team organizations pass them on to
    their members and invitees. Admin-only."""
    return await grant_org_access(user_id, admin_user_id, request.note)


@router.post("/by-email", summary="Enable organization access by email")
async def enable_org_access_by_email(
    request: EnableOrgAccessByEmailRequest,
    admin_user_id: str = Security(get_user_id),
) -> OrgAccessStatus:
    """Same as enabling by id, for an account found by its sign-in email."""
    user_id = await find_user_id_by_email(request.email)
    if user_id is None:
        raise HTTPException(status_code=404, detail="No account uses that email")
    return await grant_org_access(user_id, admin_user_id, request.note)


@router.delete("/users/{user_id}", summary="Disable organization access")
async def disable_org_access(
    user_id: str,
    admin_user_id: str = Security(get_user_id),
) -> OrgAccessStatus:
    """Take an account's organizations away. Memberships and data are kept,
    and access through another owner's team organization remains. Admin-only."""
    return await revoke_org_access(user_id, admin_user_id)
