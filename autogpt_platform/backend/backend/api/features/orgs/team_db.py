"""Database operations for workspace management."""

import logging
from datetime import datetime, timezone

from autogpt_libs.auth.permissions import OrgAction, TeamAction
from prisma import Prisma

from backend.data.db import TRANSACTION_TIMEOUT, execute_raw_with_schema, prisma
from backend.data.tenancy import (
    lock_live_org_membership_scopes,
    lock_live_org_or_team_permission_scope,
    lock_live_org_permission_scope,
    lock_live_team_scope,
)
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from .db import (
    assert_no_owned_resources,
    assert_no_owned_schedules,
    lock_org_membership,
)
from .team_model import TeamMemberResponse, TeamResponse

logger = logging.getLogger(__name__)


async def _lock_team_manager(
    client: Prisma,
    org_id: str,
    ws_id: str,
    actor_user_id: str,
    team_action: TeamAction,
    related_user_ids: list[str] | None = None,
) -> None:
    if (
        await lock_live_org_or_team_permission_scope(
            client,
            actor_user_id,
            org_id,
            ws_id,
            OrgAction.MANAGE_WORKSPACES,
            team_action,
            related_user_ids,
        )
        is None
    ):
        raise NotAuthorizedError("Workspace management access was revoked")


async def create_team(
    org_id: str,
    name: str,
    user_id: str,
    description: str | None = None,
    join_policy: str = "OPEN",
    require_live_permission: bool = False,
) -> TeamResponse:
    """Create a workspace and make the creator an admin."""
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        await lock_org_membership(tx, org_id, user_id)
        member = await tx.orgmember.find_first(
            where={
                "orgId": org_id,
                "userId": user_id,
                "status": "ACTIVE",
                "Org": {"is": {"deletedAt": None}},
            }
        )
        if member is None or (
            require_live_permission
            and not (member.isOwner or member.isAdmin or member.isBillingManager)
        ):
            raise NotAuthorizedError("Workspace creation access was revoked")
        ws = await tx.team.create(
            data={
                "name": name,
                "orgId": org_id,
                "description": description,
                "joinPolicy": join_policy,
                "createdByUserId": user_id,
            }
        )
        await tx.teammember.create(
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
    if ws is None or ws.archivedAt is not None:
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
    if ws is None or ws.archivedAt is not None:
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


async def update_team(
    ws_id: str,
    data: dict,
    *,
    org_id: str | None = None,
    actor_user_id: str | None = None,
) -> TeamResponse:
    """Update workspace fields. Guards the default workspace join policy."""
    update_data = {k: v for k, v in data.items() if v is not None}
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        if org_id is not None and actor_user_id is not None:
            await _lock_team_manager(
                tx,
                org_id,
                ws_id,
                actor_user_id,
                TeamAction.MANAGE_SETTINGS,
            )
        else:
            await lock_live_team_scope(tx, ws_id)
        current = await tx.team.find_unique(where={"id": ws_id})
        if (
            current is None
            or current.archivedAt is not None
            or (org_id is not None and current.orgId != org_id)
        ):
            raise NotFoundError(f"Workspace {ws_id} not found")
        if not update_data:
            return TeamResponse.from_db(current)
        if "joinPolicy" in update_data and current.isDefault:
            raise ValueError("Cannot change the default workspace's join policy")
        updated = await tx.team.update(where={"id": ws_id}, data=update_data)
        return TeamResponse.from_db(updated)


async def delete_team(
    ws_id: str,
    *,
    org_id: str | None = None,
    actor_user_id: str | None = None,
) -> None:
    """Archive a workspace without widening its resources to org-home."""
    ws = await prisma.team.find_unique(where={"id": ws_id})
    if ws is None or ws.archivedAt is not None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        effective_org_id = org_id or ws.orgId
        if actor_user_id is not None:
            if (
                await lock_live_org_permission_scope(
                    tx,
                    actor_user_id,
                    effective_org_id,
                    OrgAction.MANAGE_WORKSPACES,
                )
                is None
            ):
                raise NotAuthorizedError("Workspace management access was revoked")
        await _lock_team(tx, ws_id)
        current = await tx.team.find_unique(where={"id": ws_id})
        if (
            current is None
            or current.archivedAt is not None
            or current.orgId != effective_org_id
        ):
            raise NotFoundError(f"Workspace {ws_id} not found")
        if current.isDefault:
            raise ValueError("Cannot delete the default workspace")

        await tx.team.update(
            where={"id": ws_id},
            data={
                "archivedAt": datetime.now(timezone.utc),
                "name": f"{current.name} [archived {current.id[:8]}]",
                "slug": None,
            },
        )


async def join_team(ws_id: str, user_id: str, org_id: str) -> TeamResponse:
    """Self-join an OPEN workspace. User must be an org member."""
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        if (
            await lock_live_org_permission_scope(
                tx, user_id, org_id, OrgAction.VIEW_RESOURCES
            )
            is None
        ):
            raise ValueError(
                f"User {user_id} is not an active member of the organization"
            )
        await _lock_team(tx, ws_id)
        ws = await tx.team.find_unique(where={"id": ws_id})
        if ws is None or ws.archivedAt is not None:
            raise NotFoundError(f"Workspace {ws_id} not found")
        if ws.orgId != org_id:
            raise ValueError("Workspace does not belong to this organization")
        if ws.joinPolicy != "OPEN":
            raise ValueError("Cannot self-join a PRIVATE workspace. Request an invite.")
        existing = await tx.teammember.find_unique(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
        )
        if existing is not None:
            if existing.status != "ACTIVE":
                raise ValueError("Workspace membership is being removed")
            return TeamResponse.from_db(ws, is_member=True)
        await tx.teammember.create(
            data={"teamId": ws_id, "userId": user_id, "status": "ACTIVE"}
        )
    return TeamResponse.from_db(ws, is_member=True)


async def _lock_team(client: Prisma, ws_id: str) -> None:
    await lock_live_team_scope(client, ws_id)
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"Team" SET "updatedAt" = "updatedAt" ' 'WHERE "id" = $1',
        ws_id,
        client=client,
    )


async def _start_team_member_removal(
    ws_id: str,
    user_id: str,
    org_id: str | None,
    reject_default: bool,
    requesting_user_id: str,
    require_manage_permission: bool,
) -> str:
    team = await prisma.team.find_unique(where={"id": ws_id})
    if team is None or team.archivedAt is not None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if org_id is not None and team.orgId != org_id:
        raise NotFoundError(f"Workspace {ws_id} not found")
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        if require_manage_permission:
            await _lock_team_manager(
                tx,
                team.orgId,
                ws_id,
                requesting_user_id,
                TeamAction.MANAGE_MEMBERS,
                [user_id],
            )
        else:
            if (
                await lock_live_org_permission_scope(
                    tx,
                    requesting_user_id,
                    team.orgId,
                    OrgAction.VIEW_RESOURCES,
                    [user_id],
                )
                is None
            ):
                raise NotAuthorizedError("Workspace access was revoked")
            await _lock_team(tx, ws_id)
        current_team = await tx.team.find_unique(where={"id": ws_id})
        if current_team is None or current_team.archivedAt is not None:
            raise NotFoundError(f"Workspace {ws_id} not found")
        if reject_default and current_team.isDefault:
            raise ValueError("Cannot leave the default workspace")
        member = await tx.teammember.find_unique(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}},
            include={"User": True},
        )
        if member is None or member.status != "ACTIVE":
            raise NotFoundError(f"Workspace membership for {user_id} not found")
        if member.isAdmin:
            admin_count = await tx.teammember.count(
                where={"teamId": ws_id, "isAdmin": True, "status": "ACTIVE"}
            )
            if admin_count <= 1:
                raise ValueError(
                    "Cannot remove the last workspace admin. "
                    "Promote another member to admin first."
                )
        await tx.teammember.update(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}},
            data={"status": "SUSPENDED"},
        )
    return team.orgId


async def _finalize_team_member_removal(
    ws_id: str,
    user_id: str,
    org_id: str,
    requesting_user_id: str,
    require_manage_permission: bool,
) -> None:
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        if require_manage_permission:
            await _lock_team_manager(
                tx,
                org_id,
                ws_id,
                requesting_user_id,
                TeamAction.MANAGE_MEMBERS,
                [user_id],
            )
        else:
            if (
                await lock_live_org_permission_scope(
                    tx,
                    requesting_user_id,
                    org_id,
                    OrgAction.VIEW_RESOURCES,
                    [user_id],
                )
                is None
            ):
                raise NotAuthorizedError("Workspace access was revoked")
            await _lock_team(tx, ws_id)
        member = await tx.teammember.find_unique(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
        )
        if member is None or member.status != "SUSPENDED":
            raise NotFoundError(f"Workspace membership for {user_id} not found")
        await assert_no_owned_resources(tx, org_id, user_id, ws_id)
        await tx.teaminvite.update_many(
            where={
                "teamId": ws_id,
                "targetUserId": user_id,
                "acceptedAt": None,
                "revokedAt": None,
            },
            data={"revokedAt": datetime.now(timezone.utc)},
        )
        await tx.teammember.delete(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
        )


async def _remove_team_membership(
    ws_id: str,
    user_id: str,
    org_id: str | None,
    reject_default: bool,
    requesting_user_id: str,
    require_manage_permission: bool,
) -> None:
    resolved_org_id = await _start_team_member_removal(
        ws_id,
        user_id,
        org_id,
        reject_default,
        requesting_user_id,
        require_manage_permission,
    )
    try:
        await assert_no_owned_resources(prisma, resolved_org_id, user_id, ws_id)
        await assert_no_owned_schedules(
            resolved_org_id, user_id, [ws_id], team_id=ws_id
        )
        await _finalize_team_member_removal(
            ws_id,
            user_id,
            resolved_org_id,
            requesting_user_id,
            require_manage_permission,
        )
    except Exception:
        async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
            await lock_live_org_membership_scopes(
                tx, resolved_org_id, [requesting_user_id, user_id]
            )
            org = await tx.organization.find_first(
                where={"id": resolved_org_id, "deletedAt": None}
            )
            if org is not None:
                await tx.teammember.update_many(
                    where={
                        "teamId": ws_id,
                        "userId": user_id,
                        "status": "SUSPENDED",
                    },
                    data={"status": "ACTIVE"},
                )
        raise


async def leave_team(ws_id: str, user_id: str, org_id: str) -> None:
    await _remove_team_membership(
        ws_id,
        user_id,
        org_id,
        reject_default=True,
        requesting_user_id=user_id,
        require_manage_permission=False,
    )


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
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        if invited_by is not None:
            await _lock_team_manager(
                tx,
                org_id,
                ws_id,
                invited_by,
                TeamAction.MANAGE_MEMBERS,
                [user_id],
            )
        else:
            await lock_org_membership(tx, org_id, user_id)
            await _lock_team(tx, ws_id)
        ws = await tx.team.find_unique(where={"id": ws_id})
        if ws is None or ws.archivedAt is not None or ws.orgId != org_id:
            raise ValueError(f"Workspace {ws_id} does not belong to org {org_id}")
        org_member = await tx.orgmember.find_unique(
            where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
        )
        if org_member is None or org_member.status != "ACTIVE":
            raise ValueError(
                f"User {user_id} is not an active member of the organization"
            )
        member = await tx.teammember.create(
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
    *,
    org_id: str | None = None,
    requesting_user_id: str | None = None,
) -> TeamMemberResponse:
    """Update a workspace member's role flags."""
    update_data: dict[str, bool] = {}
    if is_admin is not None:
        update_data["isAdmin"] = is_admin
    if is_billing_manager is not None:
        update_data["isBillingManager"] = is_billing_manager

    team = await prisma.team.find_unique(where={"id": ws_id})
    if team is None or team.archivedAt is not None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    async with prisma.tx(timeout=TRANSACTION_TIMEOUT) as tx:
        effective_org_id = org_id or team.orgId
        if requesting_user_id is not None:
            await _lock_team_manager(
                tx,
                effective_org_id,
                ws_id,
                requesting_user_id,
                TeamAction.MANAGE_MEMBERS,
                [user_id],
            )
        else:
            await lock_org_membership(tx, effective_org_id, user_id)
            await _lock_team(tx, ws_id)
        current_team = await tx.team.find_unique(where={"id": ws_id})
        if (
            current_team is None
            or current_team.archivedAt is not None
            or current_team.orgId != effective_org_id
        ):
            raise NotFoundError(f"Workspace {ws_id} not found")
        member = await tx.teammember.find_unique(
            where={"teamId_userId": {"teamId": ws_id, "userId": user_id}}
        )
        if member is None or member.status != "ACTIVE":
            raise NotFoundError(f"Workspace membership for {user_id} not found")
        if is_admin is False and member.isAdmin:
            other_admins = await tx.teammember.count(
                where={
                    "teamId": ws_id,
                    "userId": {"not": user_id},
                    "isAdmin": True,
                    "status": "ACTIVE",
                }
            )
            if other_admins == 0:
                raise ValueError(
                    "Cannot demote the last workspace admin. "
                    "Promote another member to admin first."
                )
        if update_data:
            member = await tx.teammember.update(
                where={"teamId_userId": {"teamId": ws_id, "userId": user_id}},
                data=update_data,
                include={"User": True},
            )

    return TeamMemberResponse.from_db(member)


async def remove_team_member(
    ws_id: str,
    user_id: str,
    *,
    org_id: str | None = None,
    requesting_user_id: str | None = None,
) -> None:
    await _remove_team_membership(
        ws_id,
        user_id,
        org_id,
        reject_default=False,
        requesting_user_id=requesting_user_id or user_id,
        require_manage_permission=requesting_user_id is not None,
    )
