"""Shared org/team visibility predicate for tenancy-scoped reads.

One definition of "what can this user see in this org" so every list
and fetch surface applies identical semantics:

- org-home rows (``teamId`` NULL) are visible to every org member
- team rows are visible to members of that team
- a user's own rows are always visible to them within the org
- untagged rows (created before org tagging, not yet backfilled) stay
  visible to their owning user

``organization_id`` must come from a membership-verified RequestContext
(or an equally trusted source such as ExecutionContext) — this module
does not re-verify org membership.
"""

import logging
from collections.abc import Collection

from backend.data.db import prisma

logger = logging.getLogger(__name__)


async def get_user_team_ids(user_id: str, organization_id: str) -> list[str]:
    """IDs of teams in *organization_id* where *user_id* is an ACTIVE member."""
    memberships = await prisma.teammember.find_many(
        where={
            "userId": user_id,
            "status": "ACTIVE",
            "Team": {"is": {"orgId": organization_id}},
        }
    )
    return [m.teamId for m in memberships]


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
            # Own rows in this org, and untagged (pre-backfill) rows.
            {
                user_field: user_id,
                "OR": [
                    {org_field: organization_id},
                    {org_field: None},
                ],
            },
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


async def writable_organization_ids(
    user_id: str, organization_id: str
) -> frozenset[str | None]:
    """The organizations whose rows a call acting in *organization_id* may change.

    Rows tagged with it, and untagged rows only when it is the user's personal
    org: they predate org tagging, and the backfill gives them the personal
    org, so changing them from another org would change what the personal org
    runs. ``None`` in the result stands for untagged rows.
    """
    personal = await prisma.orgmember.find_first(
        where={
            "userId": user_id,
            "orgId": organization_id,
            "isOwner": True,
            "Org": {"isPersonal": True, "deletedAt": None},
        }
    )
    if personal is None:
        return frozenset({organization_id})
    return frozenset({organization_id, None})


def organization_filter(
    organization_ids: Collection[str | None], *, org_field: str = "organizationId"
) -> dict:
    """A Prisma clause matching rows tagged with one of *organization_ids*.

    ``None`` among them matches untagged rows.
    """
    clauses: list[dict] = [
        {org_field: {"in": sorted(o for o in organization_ids if o is not None)}}
    ]
    if None in organization_ids:
        clauses.append({org_field: None})
    return {"OR": clauses}
