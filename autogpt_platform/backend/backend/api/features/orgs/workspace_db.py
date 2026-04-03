"""Database operations for workspace management."""

import logging

from backend.data.db import prisma
from backend.util.exceptions import NotFoundError

from .workspace_model import WorkspaceMemberResponse, WorkspaceResponse

logger = logging.getLogger(__name__)


async def create_workspace(
    org_id: str,
    name: str,
    user_id: str,
    description: str | None = None,
    join_policy: str = "OPEN",
) -> WorkspaceResponse:
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

    return WorkspaceResponse.from_db(ws, member_count=1)


async def list_workspaces(org_id: str, user_id: str) -> list[WorkspaceResponse]:
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
    return [WorkspaceResponse.from_db(ws) for ws in workspaces]


async def get_workspace(
    ws_id: str, expected_org_id: str | None = None
) -> WorkspaceResponse:
    """Get workspace details. Validates org ownership if expected_org_id is given."""
    ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
    if ws is None:
        raise NotFoundError(f"Workspace {ws_id} not found")
    if expected_org_id and ws.orgId != expected_org_id:
        raise NotFoundError(f"Workspace {ws_id} not found in org {expected_org_id}")
    return WorkspaceResponse.from_db(ws)


async def update_workspace(ws_id: str, data: dict) -> WorkspaceResponse:
    """Update workspace fields. Guards the default workspace join policy."""
    update_data = {k: v for k, v in data.items() if v is not None}
    if not update_data:
        return await get_workspace(ws_id)

    # Guard: default workspace joinPolicy cannot be changed
    if "joinPolicy" in update_data:
        ws = await prisma.orgworkspace.find_unique(where={"id": ws_id})
        if ws and ws.isDefault:
            raise ValueError("Cannot change the default workspace's join policy")

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


async def join_workspace(ws_id: str, user_id: str, org_id: str) -> WorkspaceResponse:
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
        return WorkspaceResponse.from_db(ws)

    await prisma.orgworkspacemember.create(
        data={
            "workspaceId": ws_id,
            "userId": user_id,
            "status": "ACTIVE",
        }
    )
    return WorkspaceResponse.from_db(ws)


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


async def list_workspace_members(ws_id: str) -> list[WorkspaceMemberResponse]:
    """List all active members of a workspace."""
    members = await prisma.orgworkspacemember.find_many(
        where={"workspaceId": ws_id, "status": "ACTIVE"},
        include={"User": True},
    )
    return [WorkspaceMemberResponse.from_db(m) for m in members]


async def add_workspace_member(
    ws_id: str,
    user_id: str,
    org_id: str,
    is_admin: bool = False,
    is_billing_manager: bool = False,
    invited_by: str | None = None,
) -> WorkspaceMemberResponse:
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
    return WorkspaceMemberResponse.from_db(member)


async def update_workspace_member(
    ws_id: str,
    user_id: str,
    is_admin: bool | None,
    is_billing_manager: bool | None,
) -> WorkspaceMemberResponse:
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
    return next(m for m in members if m.user_id == user_id)


async def remove_workspace_member(ws_id: str, user_id: str) -> None:
    """Remove a member from a workspace.

    Guards against removing the last admin — workspace would become unmanageable.
    """
    # Check if this would remove the last admin
    member = await prisma.orgworkspacemember.find_unique(
        where={"workspaceId_userId": {"workspaceId": ws_id, "userId": user_id}}
    )
    if member and member.isAdmin:
        admin_count = await prisma.orgworkspacemember.count(
            where={"workspaceId": ws_id, "isAdmin": True, "status": "ACTIVE"}
        )
        if admin_count <= 1:
            raise ValueError(
                "Cannot remove the last workspace admin. "
                "Promote another member to admin first."
            )

    await prisma.orgworkspacemember.delete(
        where={"workspaceId_userId": {"workspaceId": ws_id, "userId": user_id}}
    )
