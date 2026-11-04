"""Admin endpoints for a sales-led team plan.

Sales invoices the customer outside the product. An admin then finds the
customer's team organization, sets its plan (tier, seats, end date, status,
note) and adds the credits the invoice bought to its wallet, keyed by the
invoice so adding them twice adds them once.
"""

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel, Field

from backend.data.org_access import find_user_id_by_email
from backend.data.org_plan_admin import (
    OrgCreditGrantResult,
    OrgPlanInput,
    grant_org_credits,
    list_account_team_orgs,
    set_org_plan,
)
from backend.data.org_plan_summary import OrgPlanSummary, org_plan_summary

router = APIRouter(
    prefix="/admin/org-plans",
    tags=["admin"],
    dependencies=[Security(requires_admin_user)],
)


class AddOrgCreditsRequest(BaseModel):
    amount: int = Field(gt=0, le=100_000_000, description="Credits to add (100 = $1).")
    reference: str = Field(
        min_length=1,
        max_length=200,
        description="The invoice or other reference; one reference adds once.",
    )
    note: str | None = Field(default=None, max_length=500)


@router.get("/orgs", summary="List an account's team organizations")
async def list_team_orgs(
    user_id: str | None = Query(default=None),
    email: str | None = Query(default=None, max_length=320),
) -> list[OrgPlanSummary]:
    """The team organizations an account belongs to, by id or sign-in email,
    with each one's plan, seats in use and wallet balance. Admin-only."""
    if user_id is None and email:
        user_id = await find_user_id_by_email(email)
        if user_id is None:
            raise HTTPException(status_code=404, detail="No account uses that email")
    if user_id is None:
        raise HTTPException(status_code=422, detail="Give a user_id or an email")
    return await list_account_team_orgs(user_id)


@router.get("/orgs/{org_id}", summary="Get a team organization's plan")
async def get_team_org_plan(org_id: str) -> OrgPlanSummary:
    """A team organization's plan, seats in use and wallet balance. Admin-only."""
    return await org_plan_summary(org_id)


@router.put("/orgs/{org_id}", summary="Set a team organization's plan")
async def put_team_org_plan(
    org_id: str,
    request: OrgPlanInput,
    admin_user_id: str = Security(get_user_id),
) -> OrgPlanSummary:
    """Set the plan's tier, seats, end date, status and note. Its members use
    paid features under the tier while the plan is active and not past its end
    date. Admin-only."""
    return await set_org_plan(org_id, admin_user_id, request)


@router.post("/orgs/{org_id}/credits", summary="Add credits to a team organization")
async def add_team_org_credits(
    org_id: str,
    request: AddOrgCreditsRequest,
    admin_user_id: str = Security(get_user_id),
) -> OrgCreditGrantResult:
    """Add credits to the organization's wallet. Repeating a reference adds
    nothing and answers with ``granted`` false. Admin-only."""
    return await grant_org_credits(
        org_id, admin_user_id, request.amount, request.reference, request.note
    )
