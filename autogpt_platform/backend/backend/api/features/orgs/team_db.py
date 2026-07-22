"""Database operations for workspace management."""

import logging

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from .team_model import TeamMemberResponse, TeamResponse

logger = logging.getLogger(__name__)


async def create_team(
    org_id: str,
    name: str,
    user_id: str,
    description: str | None = None,
    join_policy: str = "OPEN",
) -> TeamResponse:
    """Create a workspace and make the creator an admin."""
    ws = await prisma.team.create(
        data={
            "name": name,
            "orgId": org_id,
            "description": description,
            "joinPolicy": join_policy,
            "createdByUserId": user_id,
        }
    )

    # Creator becomes admin
    await prisma.teammember.create(
        data={
            "teamId": ws.id,
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    return TeamResponse.from_db(ws, member_count=1, is_member=True)


async def _member_facts(
    team_ids: list[str], user_id: str
) -> tuple[set[str], dict[str, int]]:
    """Compute per-caller membership and active member counts in bulk.

    Returns the subset of ``team_ids`` the caller is an ACTIVE member of, plus a
    ``team_id -> active member count`` map. Two queries total regardless of team
    count, so callers avoid an N+1 over the returned teams.
    """
    if not team_ids:
        return set(), {}

    caller_rows = await prisma.teammember.find_many(
        where={"teamId": {"in": team_ids}, "userId": user_id, "status": "ACTIVE"}
    )
    member_of = {row.teamId for row in caller_rows}

    count_rows = await prisma.teammember.group_by(
        by=["teamId"],
        where={"teamId": {"in": team_ids}, "status": "ACTIVE"},
        count=True,
    )
    count_by_team = {
        row["teamId"]: (row.get("_count") or {}).get("_all") or 0 for row in count_rows
    }
    return member_of, count_by_team


async def list_teams(
    org_id: str, user_id: str, can_manage_workspaces: bool = False
) -> list[TeamResponse]:
    """List workspaces visible to the caller, with a per-caller is_member flag.

    Regular members see all OPEN workspaces plus PRIVATE ones they belong to.
    Org admins (can_manage_workspaces) additionally see PRIVATE workspaces they
    are not in — as name + member count only, with the description redacted
    ("governance without surveillance").
    """
    where: dict = {"orgId": org_id, "archivedAt": None}
    if not can_manage_workspaces:
        where["OR"] = [
            {"joinPolicy": "OPEN"},
            {"Members": {"some": {"userId": user_id, "status": "ACTIVE"}}},
        ]

    workspaces = await prisma.team.find_many(where=where, order={"createdAt": "asc"})
    member_of, count_by_team = await _member_facts(
        [ws.id for ws in workspaces], user_id
    )
    return [
        TeamResponse.from_db(
            ws,
            member_count=count_by_team.get(ws.id, 0),
            is_member=ws.id in member_of,
            redact_description=(
                can_manage_workspaces
                and ws.joinPolicy != "OPEN"
                and ws.id not in member_of
            ),
        )
        for ws in workspaces
    ]


async def get_team(ws_id: str, expected_org_id: str | None = None) -> TeamResponse:
    """Get workspace details. Validates org ownership if expected_org_id is given."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if expected_org_id and ws.orgId != expected_org_id:
        raise NotFoundError(f"Workspace {ws_id} not found in org {expected_org_id}")
    return TeamResponse.from_db(ws)


async def get_team_for_viewer(
    ws_id: str,
    org_id: str,
    user_id: str,
    can_manage_workspaces: bool = False,
) -> TeamResponse:
    """Get workspace details with the same visibility rules as list_teams.

    OPEN workspaces and workspaces the caller belongs to are returned in full.
    Org admins (can_manage_workspaces) may view a PRIVATE workspace they are not
    in, but with the description redacted. A regular member asking for a PRIVATE
    workspace they don't belong to gets NotFoundError — the details route mirrors
    list visibility rather than exposing every team by id.
    """
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.orgId != org_id:
        raise NotFoundError(f"Workspace {ws_id} not found in org {org_id}")

    member_of, count_by_team = await _member_facts([ws_id], user_id)
    is_member = ws_id in member_of
    is_open = ws.joinPolicy == "OPEN"
    if not is_open and not is_member and not can_manage_workspaces:
        raise NotFoundError(f"Workspace {ws_id} not found")

    return TeamResponse.from_db(
        ws,
        member_count=count_by_team.get(ws_id, 0),
        is_member=is_member,
        redact_description=not is_open and not is_member,
    )


async def update_team(ws_id: str, data: dict) -> TeamResponse:
    """Update workspace fields. Guards the default workspace join policy."""
    update_data = {k: v for k, v in data.items() if v is not None}
    if not update_data:
        return await get_team(ws_id)

    # Guard: default workspace joinPolicy cannot be changed
    if "joinPolicy" in update_data:
        ws = await prisma.team.find_unique(where={"id": ws_id})
        if ws and ws.isDefault:
            raise ValueError("Cannot change the default workspace's join policy")

    await prisma.team.update(where={"id": ws_id}, data=update_data)
    return await get_team(ws_id)


async def delete_team(ws_id: str) -> None:
    """Delete a workspace. Cannot delete the default workspace."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.isDefault:
        raise ValueError("Cannot delete the default workspace")

    await prisma.team.delete(where={"id": ws_id})


async def join_team(ws_id: str, user_id: str, org_id: str) -> TeamResponse:
    """Self-join an OPEN workspace. User must be an org member."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.orgId != org_id:
        raise ValueError("Workspace does not belong to this organization")
    if ws.joinPolicy != "OPEN":
        raise ValueError("Cannot self-join a PRIVATE workspace. Request an invite.")

    # Verify user is actually an org member
    org_member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if org_member is None:
        raise ValueError(f"User {user_id} is not a member of the organization")

    # Check not already a member
    existing = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
    )
    if existing:
        return TeamResponse.from_db(ws, is_member=True)

    await prisma.teammember.create(
        data={
            "teamId": ws_id,
            "userId": user_id,
            "status": "ACTIVE",
        }
    )
    return TeamResponse.from_db(ws, is_member=True)


async def leave_team(ws_id: str, user_id: str) -> None:
    """Leave a workspace. Cannot leave the default workspace."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.isDefault:
        raise ValueError("Cannot leave the default workspace")

    await prisma.teammember.delete_many(where={"teamId": ws_id, "userId": user_id})


async def list_team_members(ws_id: str) -> list[TeamMemberResponse]:
    """List all active members of a workspace."""
    members = await prisma.teammember.find_many(
        where={"teamId": ws_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [TeamMemberResponse.from_db(m) for m in members]


async def is_team_admin(ws_id: str, user_id: str) -> bool:
    """Return True if the user is an active admin of the workspace."""
    member = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
    )
    return bool(member and member.isAdmin and member.status == "ACTIVE")


async def add_team_member(
    ws_id: str,
    user_id: str,
    org_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> TeamMemberResponse:
    """Add a member to a workspace. Must be an org member, workspace must belong to org."""
    # Verify workspace belongs to the org
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None or ws.orgId != org_id:
        raise ValueError(f"Workspace {ws_id} does not belong to org {org_id}")

    # Verify user is in the org
    org_member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if org_member is None:
        raise ValueError(f"User {user_id} is not a member of the organization")

    member = await prisma.teammember.create(
        data={
            "teamId": ws_id,
            "userId": user_id,
            "isAdmin": is_admin,
            "isBillingManager": is_billing_manager,
            "status": "ACTIVE",
            "invitedByUserId": invited_by,
        },
        include={"User": True},
    )
    return TeamMemberResponse.from_db(member)


async def update_team_member(
    ws_id: str,
    user_id: str,
    is_admin: bool | None,
    is_billing_manager: bool | None,
) -> TeamMemberResponse:
    """Update a workspace member's role flags."""
    update_data: dict = {}
    if is_admin is not None:
        update_data["isAdmin"] = is_admin
    if is_billing_manager is not None:
        update_data["isBillingManager"] = is_billing_manager

    if update_data:
        await prisma.teammember.update(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}},
            data=update_data,
        )

    members = await list_team_members(ws_id)
    return next(m for m in members if m.user_id == user_id)


async def remove_team_member(ws_id: str, user_id: str) -> None:
    """Remove a member from a workspace.

    Guards against removing the last admin — workspace would become unmanageable.
    """
    # Check if this would remove the last admin
    member = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
    )
    if member and member.isAdmin:
        admin_count = await prisma.teammember.count(
            where={"teamId": ws_id, "isAdmin": True, "status": "ACTIVE"}
        )
        if admin_count <= 1:
            raise ValueError(
                "Cannot remove the last workspace admin. "
                "Promote another member to admin first."
            )

    await prisma.teammember.delete(
        where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
    )
