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
    avatar_url: str | None = None


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    avatar_url: str | None
    description: str | None
    is_personal: bool
    member_count: int
    created_at: datetime

    @staticmethod
    def from_db(org, member_count: int = 0) -> "OrgResponse":
        return OrgResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            avatar_url=org.avatarUrl,
            description=org.description,
            is_personal=org.isPersonal,
            member_count=member_count,
            created_at=org.createdAt,
        )


class OrgMemberResponse(BaseModel):
    id: str
    user_id: str
    email: str
    name: str | None
    is_owner: bool
    is_admin: bool
    is_billing_manager: bool
    joined_at: datetime

    @staticmethod
    def from_db(member) -> "OrgMemberResponse":
        return OrgMemberResponse(
            id=member.id,
            user_id=member.userId,
            email=member.User.email if member.User else "",
            name=member.User.name if member.User else None,
            is_owner=member.isOwner,
            is_admin=member.isAdmin,
            is_billing_manager=member.isBillingManager,
            joined_at=member.joinedAt,
        )


class AddMemberRequest(BaseModel):
    user_id: str
    is_admin: bool = False
    is_billing_manager: bool = False


class UpdateMemberRequest(BaseModel):
    is_admin: bool | None = None
    is_billing_manager: bool | None = None


class TransferOwnershipRequest(BaseModel):
    new_owner_id: str


class OrgAliasResponse(BaseModel):
    id: str
    alias_slug: str
    alias_type: str
    created_at: datetime

    @staticmethod
    def from_db(alias) -> "OrgAliasResponse":
        return OrgAliasResponse(
            id=alias.id,
            alias_slug=alias.aliasSlug,
            alias_type=alias.aliasType,
            created_at=alias.createdAt,
        )


class CreateAliasRequest(BaseModel):
    alias_slug: str = Field(
        ..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$"
    )


class CreateInvitationRequest(BaseModel):
    email: str
    is_admin: bool = False
    is_billing_manager: bool = False
    workspace_ids: list[str] = Field(default_factory=list)


class InvitationResponse(BaseModel):
    id: str
    email: str
    is_admin: bool
    is_billing_manager: bool
    token: str
    expires_at: datetime
    created_at: datetime
    workspace_ids: list[str]

    @staticmethod
    def from_db(inv) -> "InvitationResponse":
        return InvitationResponse(
            id=inv.id,
            email=inv.email,
            is_admin=inv.isAdmin,
            is_billing_manager=inv.isBillingManager,
            token=inv.token,
            expires_at=inv.expiresAt,
            created_at=inv.createdAt,
            workspace_ids=inv.workspaceIds,
        )
