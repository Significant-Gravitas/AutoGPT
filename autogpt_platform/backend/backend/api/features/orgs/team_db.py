"""Database operations for workspace management."""

import logging

from prisma import Prisma
from prisma.errors import UniqueViolationError
from prisma.models import TeamMember
from prisma.types import TeamMemberWhereUniqueInput

from backend.data.db import prisma, transaction
from backend.util.exceptions import ConflictError, NotFoundError

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

    return TeamResponse.from_db(ws, member_count=1)


async def list_teams(org_id: str, user_id: str) -> list[TeamResponse]:
    """List workspaces: all OPEN workspaces + PRIVATE ones the user belongs to."""
    workspaces = await prisma.team.find_many(
        where={
            "orgId": org_id,
            "archivedAt": None,
            "OR": [
                {"joinPolicy": "OPEN"},
                {"Members": {"some": {"userId": user_id, "status": "ACTIVE"}}},
            ],
        },
        order={"createdAt": "asc"},
    )
    return [TeamResponse.from_db(ws) for ws in workspaces]


async def get_team(ws_id: str, expected_org_id: str | None = None) -> TeamResponse:
    """Get workspace details. Validates org ownership if expected_org_id is given."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if expected_org_id and ws.orgId != expected_org_id:
        raise NotFoundError(f"Workspace {ws_id} not found in org {expected_org_id}")
    return TeamResponse.from_db(ws)


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
        return TeamResponse.from_db(ws)

    await prisma.teammember.create(
        data={
            "teamId": ws_id,
            "userId": user_id,
            "status": "ACTIVE",
        }
    )
    return TeamResponse.from_db(ws)


async def leave_team(ws_id: str, user_id: str) -> None:
    """Leave a workspace. Cannot leave the default workspace."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.isDefault:
        raise ValueError("Cannot leave the default workspace")

    async with transaction() as tx:
        await _lock_team_admins(tx, ws_id)
        member = await tx.teammember.find_unique(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
        )
        await _assert_not_last_admin(
            tx,
            ws_id,
            member,
            "The last workspace admin cannot leave. "
            "Promote another member to admin first.",
        )
        await tx.teammember.delete_many(where={"teamId": ws_id, "userId": user_id})


async def list_team_members(ws_id: str) -> list[TeamMemberResponse]:
    """List all active members of a workspace."""
    members = await prisma.teammember.find_many(
        where={"teamId": ws_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [TeamMemberResponse.from_db(m) for m in members]


async def is_admin_of_team(ws_id: str, user_id: str) -> bool:
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

    already_member = ConflictError(
        f"User {user_id} is already a member of workspace {ws_id}"
    )
    existing = await prisma.teammember.find_unique(
        where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
    )
    if existing:
        raise already_member

    try:
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
    except UniqueViolationError as e:
        # Lost a race with a concurrent add of the same user.
        raise already_member from e
    return TeamMemberResponse.from_db(member)


async def update_team_member(
    ws_id: str,
    user_id: str,
    is_admin: bool | None,
    is_billing_manager: bool | None,
) -> TeamMemberResponse:
    """Update a workspace member's role flags."""
    where: TeamMemberWhereUniqueInput = {
        "teamId_userId": {"teamId": ws_id, "userId": user_id}
    }
    update_data: dict = {}
    if is_admin is not None:
        update_data["isAdmin"] = is_admin
    if is_billing_manager is not None:
        update_data["isBillingManager"] = is_billing_manager

    async with transaction() as tx:
        if is_admin is False:
            await _lock_team_admins(tx, ws_id)
        member = await tx.teammember.find_unique(where=where, include={"User": True})
        if member is None or member.status != "ACTIVE":
            raise NotFoundError(f"User {user_id} is not a member of workspace {ws_id}")

        if is_admin is False:
            await _assert_not_last_admin(
                tx,
                ws_id,
                member,
                "Cannot demote the last workspace admin. "
                "Promote another member to admin first.",
            )

        if update_data:
            member = await tx.teammember.update(
                where=where, data=update_data, include={"User": True}
            )
            if member is None:
                raise NotFoundError(
                    f"User {user_id} is not a member of workspace {ws_id}"
                )

    return TeamMemberResponse.from_db(member)


async def remove_team_member(ws_id: str, user_id: str) -> None:
    """Remove a member from a workspace.

    Guards against removing the last admin — workspace would become unmanageable.
    """
    where: TeamMemberWhereUniqueInput = {
        "teamId_userId": {"teamId": ws_id, "userId": user_id}
    }
    async with transaction() as tx:
        await _lock_team_admins(tx, ws_id)
        member = await tx.teammember.find_unique(where=where)
        await _assert_not_last_admin(
            tx,
            ws_id,
            member,
            "Cannot remove the last workspace admin. "
            "Promote another member to admin first.",
        )
        await tx.teammember.delete(where=where)


async def _lock_team_admins(tx: Prisma, ws_id: str) -> None:
    """Hold the workspace's admin-roster lock until the transaction ends.

    Leave, remove and demote each count the remaining admins and then write.
    Without a shared lock two of them can both count two admins and both
    proceed, leaving none. Take it before reading the member row so the
    count that follows sees every earlier writer's commit. That relies on
    READ COMMITTED, where each statement gets a fresh snapshot, so it is
    pinned rather than left to the database default. It has to be the
    transaction's first statement, so call this before any other query.

    execute_raw, not query_raw: pg_advisory_xact_lock returns void, which
    Prisma cannot deserialize as a result column.
    """
    await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
    await tx.execute_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        f"team-admins:{ws_id}",
    )


async def _count_active_admins(tx: Prisma, ws_id: str) -> int:
    return await tx.teammember.count(
        where={"teamId": ws_id, "isAdmin": True, "status": "ACTIVE"}
    )


async def _assert_not_last_admin(
    tx: Prisma, ws_id: str, member: TeamMember | None, message: str
) -> None:
    """Raise ValueError (400) if ``member`` is the workspace's only active admin.

    Call with ``_lock_team_admins`` held, in the same transaction as the write.
    """
    if member is None or not member.isAdmin or member.status != "ACTIVE":
        return
    if await _count_active_admins(tx, ws_id) <= 1:
        raise ValueError(message)
