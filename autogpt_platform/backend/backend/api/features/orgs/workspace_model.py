"""Pydantic request/response models for workspace management."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    join_policy: str = "OPEN"  # OPEN or PRIVATE


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    join_policy: str | None = None  # OPEN or PRIVATE


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str | None
    description: str | None
    is_default: bool
    join_policy: str
    org_id: str
    member_count: int
    created_at: datetime

    @staticmethod
    def from_db(ws, member_count: int = 0) -> "WorkspaceResponse":
        return WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            description=ws.description,
            is_default=ws.isDefault,
            join_policy=ws.joinPolicy,
            org_id=ws.orgId,
            member_count=member_count,
            created_at=ws.createdAt,
        )


class WorkspaceMemberResponse(BaseModel):
    id: str
    user_id: str
    email: str
    name: str | None
    is_admin: bool
    is_billing_manager: bool
    joined_at: datetime

    @staticmethod
    def from_db(member) -> "WorkspaceMemberResponse":
        return WorkspaceMemberResponse(
            id=member.id,
            user_id=member.userId,
            email=member.User.email if member.User else "",
            name=member.User.name if member.User else None,
            is_admin=member.isAdmin,
            is_billing_manager=member.isBillingManager,
            joined_at=member.joinedAt,
        )


class AddWorkspaceMemberRequest(BaseModel):
    user_id: str
    is_admin: bool = False
    is_billing_manager: bool = False


class UpdateWorkspaceMemberRequest(BaseModel):
    is_admin: bool | None = None
    is_billing_manager: bool | None = None
