"""Pydantic request/response models for organization management."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateOrgRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    slug: str = Field(
        ..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$"
    )
    description: str | None = None


class UpdateOrgRequest(BaseModel):
    name: str | None = None
    slug: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9-]*$")
    description: str | None = None
    avatarUrl: str | None = None


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    avatarUrl: str | None
    description: str | None
    isPersonal: bool
    memberCount: int
    createdAt: datetime


class OrgMemberResponse(BaseModel):
    id: str
    userId: str
    email: str
    name: str | None
    isOwner: bool
    isAdmin: bool
    isBillingManager: bool
    joinedAt: datetime


class AddMemberRequest(BaseModel):
    userId: str
    isAdmin: bool = False
    isBillingManager: bool = False


class UpdateMemberRequest(BaseModel):
    isAdmin: bool | None = None
    isBillingManager: bool | None = None


class TransferOwnershipRequest(BaseModel):
    newOwnerId: str


class OrgAliasResponse(BaseModel):
    id: str
    aliasSlug: str
    aliasType: str
    createdAt: datetime


class CreateAliasRequest(BaseModel):
    aliasSlug: str = Field(
        ..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$"
    )
