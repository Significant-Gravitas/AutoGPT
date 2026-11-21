"""Shared org/team visibility predicate for tenancy-scoped reads.

One definition of "what can this user see in this org" so every list
and fetch surface applies identical semantics:

- org-home rows (``teamId`` NULL) are visible to every org member
- team rows are visible to members of that team
- a user's own rows are always visible to them within the org

A team membership whose only role is billing manager covers the team's
spend, not its workflows and runs (the role matrix in
``autogpt_libs.auth.permissions``), so it doesn't make the team's rows
visible, and neither does any team membership of an org billing manager. The
routes that list and read rows also check the caller's org role
(``requires_resource_permission``, and ``resource_role_refusal`` on the
External API), which keeps org billing managers from org-home rows there.

``organization_id`` must come from a membership-verified RequestContext
(or an equally trusted source such as ExecutionContext) — this module
does not re-verify org membership.
"""

import asyncio
import logging

from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import (
    OrgAction,
    TeamAction,
    check_org_permission,
    check_team_permission,
)
from prisma.models import OrgMember, TeamMember

from backend.data.db import prisma
from backend.data.org_migration import personal_org_tenancy
from backend.util.exceptions import NotAuthorizedError

logger = logging.getLogger(__name__)


async def tenancy_for_new_row(
    user_id: str, organization_id: str | None, team_id: str | None = None
) -> tuple[str, str | None]:
    """The org and team a new row of *user_id*'s is written with.

    The given ones, or without an org the user's personal org and its default
    team: the tenancy the org bootstrap gave every row written before tagging.
    """
    if organization_id:
        return organization_id, team_id
    return await personal_org_tenancy(user_id)


async def get_user_team_ids(user_id: str, organization_id: str) -> list[str]:
    """IDs of the teams in *organization_id* whose rows *user_id* may see.

    The user's ACTIVE team memberships, less those whose only role is billing
    manager. An org billing manager who doesn't also own or administer the
    org gets none.
    """
    org_member, memberships = await asyncio.gather(
        prisma.orgmember.find_first(
            where={"userId": user_id, "orgId": organization_id, "status": "ACTIVE"}
        ),
        prisma.teammember.find_many(
            where={
                "userId": user_id,
                "status": "ACTIVE",
                "Team": {"is": {"orgId": organization_id}},
            }
        ),
    )
    if org_member is None or not org_role_grants_resources(org_member):
        return []
    return [m.teamId for m in memberships if team_role_grants_resources(m)]


def org_role_grants_resources(member: OrgMember) -> bool:
    """Whether an org membership's role covers the org's workflows and runs."""
    return member.isOwner or member.isAdmin or not member.isBillingManager


def team_role_grants_resources(member: TeamMember) -> bool:
    """Whether a team membership's role covers the team's workflows and runs."""
    return member.isAdmin or not member.isBillingManager


BILLING_ONLY_ORG_ROLE = (
    "Your role in this organization covers its billing, not its workflows, runs "
    "or library"
)
BILLING_ONLY_TEAM_ROLE = (
    "Your role in team {team_id} covers its billing, not its workflows, runs or "
    "library"
)


async def resource_role_refusal(
    user_id: str, organization_id: str, team_id: str | None
) -> str | None:
    """Why the user's role keeps them from the workflows, runs and library of
    *organization_id*, and of team *team_id* in it when one is given, or None
    when it doesn't.

    The roles come from the user's ACTIVE memberships there, read on every
    call. A membership the user doesn't hold refuses nothing here: membership
    itself is checked where the org and team are resolved.
    """
    org_member, team_member = await asyncio.gather(
        prisma.orgmember.find_first(
            where={"userId": user_id, "orgId": organization_id, "status": "ACTIVE"}
        ),
        _active_team_membership(user_id, organization_id, team_id),
    )
    if org_member is not None and not org_role_grants_resources(org_member):
        return BILLING_ONLY_ORG_ROLE
    if team_member is not None and not team_role_grants_resources(team_member):
        return BILLING_ONLY_TEAM_ROLE.format(team_id=team_id)
    return None


async def _active_team_membership(
    user_id: str, organization_id: str, team_id: str | None
) -> TeamMember | None:
    if not team_id:
        return None
    return await prisma.teammember.find_first(
        where={
            "userId": user_id,
            "teamId": team_id,
            "status": "ACTIVE",
            "Team": {"is": {"orgId": organization_id}},
        }
    )


async def require_role_where_resource_lives(
    user_id: str,
    organization_id: str | None,
    team_id: str | None,
    org_action: OrgAction,
    team_action: TeamAction,
) -> None:
    """Refuse an action on an existing workflow, run or preset that the user's
    role in the resource's own org and team withholds.

    ``requires_resource_permission`` checks the org and team a request names;
    a resource can live in another team than that, so a route acting on one
    checks this too. The roles come from the user's ACTIVE memberships there
    and the role matrix decides. Without such a membership nothing is refused
    here: the route's ownership checks decide, as before.

    Raises:
        NotAuthorizedError: the user's role there withholds the action.
    """
    if not organization_id:
        return
    org_member = await prisma.orgmember.find_first(
        where={"userId": user_id, "orgId": organization_id, "status": "ACTIVE"}
    )
    if org_member is None:
        return
    team_member = (
        await prisma.teammember.find_first(
            where={
                "userId": user_id,
                "teamId": team_id,
                "status": "ACTIVE",
                "Team": {"is": {"orgId": organization_id}},
            }
        )
        if team_id
        else None
    )
    ctx = RequestContext(
        user_id=user_id,
        org_id=organization_id,
        team_id=team_member.teamId if team_member else None,
        is_org_owner=org_member.isOwner,
        is_org_admin=org_member.isAdmin,
        is_org_billing_manager=org_member.isBillingManager,
        is_team_admin=team_member is not None and team_member.isAdmin,
        is_team_billing_manager=(
            team_member is not None and team_member.isBillingManager
        ),
        seat_status="ACTIVE",
    )
    if not check_org_permission(ctx, org_action):
        raise NotAuthorizedError("Your role in this organization doesn't allow this.")
    if team_member is not None and not check_team_permission(ctx, team_action):
        raise NotAuthorizedError("Your role in this workspace doesn't allow this.")


def visibility_filter(
    user_id: str,
    organization_id: str | None,
    team_ids: list[str],
    *,
    user_field: str = "userId",
    org_field: str = "organizationId",
    team_field: str = "teamId",
) -> dict:
    """Build a Prisma OR-clause implementing the visibility rules above.

    With no org context (``organization_id`` is None) this degrades to
    plain personal ownership, preserving pre-org behaviour for internal
    callers that don't resolve a RequestContext.
    """
    if organization_id is None:
        return {user_field: user_id}

    return {
        "OR": [
            # Own rows in this org.
            {user_field: user_id, org_field: organization_id},
            # Org-home rows: visible to every member of the org.
            {org_field: organization_id, team_field: None},
            # Team rows: visible to members of those teams.
            *(
                [{org_field: organization_id, team_field: {"in": team_ids}}]
                if team_ids
                else []
            ),
        ]
    }
