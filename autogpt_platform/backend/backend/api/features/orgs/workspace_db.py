"""Database operations for workspace management."""

import logging

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


def _ws_to_dict(ws, member_count: int = 0) -> dict:
    return {
        "id": ws.id,
        "name": ws.name,
        "slug": ws.slug,
        "description": ws.description,
        "isDefault": ws.isDefault,
        "joinPolicy": ws.joinPolicy,
        "orgId": ws.orgId,
        "memberCount": member_count,
        "createdAt": ws.createdAt,
    }


async def create_workspace(
    org_id: str,
    name: str,
    user_id: str,
    description: str | None = None,
    join_policy: str = "OPEN",
) -> dict:
    """Create a workspace and make the creator an admin."""
    ws = await prisma.orgworkspace.create(
        data={
            "name": name,
            "orgId": org_id,
            "description": description,
            "joinPolicy": join_policy,
            "createdByUserId": user_id,
        }
    )

    # Creator becomes admin
    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": ws.id,
            "userId": user_id,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )

    return _ws_to_dict(ws, member_count=1)


async def list_workspaces(org_id: str, user_id: str) -> list[dict]:
    """List workspaces: all OPEN workspaces + PRIVATE ones the user belongs to."""
    workspaces = await prisma.orgworkspace.find_many(
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
    return [_ws_to_dict(ws) for ws in workspaces]


async def get_workspace(ws_id: str) -> dict:
    """Get workspace details."""
    ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    return _ws_to_dict(ws)


async def update_workspace(ws_id: str, data: dict) -> dict:
    """Update workspace fields."""
    update_data = {k: v for k, v in data.items() if v is not None}
    if not update_data:
        return await get_workspace(ws_id)

    await prisma.orgworkspace.update(where={"id": ws_id}, data=update_data)
    return await get_workspace(ws_id)


async def delete_workspace(ws_id: str) -> None:
    """Delete a workspace. Cannot delete the default workspace."""
    ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.isDefault:
        raise ValueError("Cannot delete the default workspace")

    await prisma.orgworkspace.delete(where={"id": ws_id})


async def join_workspace(ws_id: str, user_id: str, org_id: str) -> dict:
    """Self-join an OPEN workspace. User must be an org member."""
    ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.orgId != org_id:
        raise ValueError("Workspace does not belong to this organization")
    if ws.joinPolicy != "OPEN":
        raise ValueError("Cannot self-join a PRIVATE workspace. Request an invite.")

    # Check not already a member
    existing = await prisma.orgworkspacemember.find_unique(
        where={"workspaceId_userId": {"workspaceId": ws_id, "userId": user_id}}
    )
    if existing:
        return _ws_to_dict(ws)

    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": ws_id,
            "userId": user_id,
            "status": "ACTIVE",
        }
    )
    return _ws_to_dict(ws)


async def leave_workspace(ws_id: str, user_id: str) -> None:
    """Leave a workspace. Cannot leave the default workspace."""
    ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if ws.isDefault:
        raise ValueError("Cannot leave the default workspace")

    await prisma.orgworkspacemember.delete_many(
        where={"workspaceId": ws_id, "userId": user_id}
    )


async def list_workspace_members(ws_id: str) -> list[dict]:
    """List all active members of a workspace."""
    members = await prisma.orgworkspacemember.find_many(
        where={"workspaceId": ws_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [
        {
            "id": m.id,
            "userId": m.userId,
            "email": m.User.email if m.User else "",
            "name": m.User.name if m.User else None,
            "isAdmin": m.isAdmin,
            "isBillingManager": m.isBillingManager,
            "joinedAt": m.joinedAt,
        }
        for m in members
    ]


async def add_workspace_member(
    ws_id: str,
    user_id: str,
    org_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> dict:
    """Add a member to a workspace. Must be an org member."""
    # Verify user is in the org
    org_member = await prisma.orgmember.find_unique(
        where={"orgId_userId": {"orgId": org_id, "userId": user_id}}
    )
    if org_member is None:
        raise ValueError(f"User {user_id} is not a member of the organization")

    member = await prisma.orgworkspacemember.create(
        data={
            "workspaceId": ws_id,
            "userId": user_id,
            "isAdmin": is_admin,
            "isBillingManager": is_billing_manager,
            "status": "ACTIVE",
            "invitedByUserId": invited_by,
        },
        include={"User": True},
    )
    return {
        "id": member.id,
        "userId": member.userId,
        "email": member.User.email if member.User else "",
        "name": member.User.name if member.User else None,
        "isAdmin": member.isAdmin,
        "isBillingManager": member.isBillingManager,
        "joinedAt": member.joinedAt,
    }


async def update_workspace_member(
    ws_id: str,
    user_id: str,
    is_admin: bool | None,
    is_billing_manager: bool | None,
) -> dict:
    """Update a workspace member's role flags."""
    update_data: dict = {}
    if is_admin is not None:
        update_data["isAdmin"] = is_admin
    if is_billing_manager is not None:
        update_data["isBillingManager"] = is_billing_manager

    if update_data:
        await prisma.orgworkspacemember.update(
            where={"workspaceId_userId": {"workspaceId": ws_id, "userId": user_id}},
            data=update_data,
        )

    members = await list_workspace_members(ws_id)
    return next(m for m in members if m["userId"] == user_id)


async def remove_workspace_member(ws_id: str, user_id: str) -> None:
    """Remove a member from a workspace."""
    await prisma.orgworkspacemember.delete(
        where={"workspaceId_userId": {"workspaceId": ws_id, "userId": user_id}}
    )
