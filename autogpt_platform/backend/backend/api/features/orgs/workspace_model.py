"""Pydantic request/response models for workspace management."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    joinPolicy: str = "OPEN"  # OPEN or PRIVATE


class UpdateWorkspaceRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    joinPolicy: str | None = None  # OPEN or PRIVATE


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str | None
    description: str | None
    isDefault: bool
    joinPolicy: str
    orgId: str
    memberCount: int
    createdAt: datetime


class WorkspaceMemberResponse(BaseModel):
    id: str
    userId: str
    email: str
    name: str | None
    isAdmin: bool
    isBillingManager: bool
    joinedAt: datetime


class AddWorkspaceMemberRequest(BaseModel):
    userId: str
    isAdmin: bool = False
    isBillingManager: bool = False


class UpdateWorkspaceMemberRequest(BaseModel):
    isAdmin: bool | None = None
    isBillingManager: bool | None = None
