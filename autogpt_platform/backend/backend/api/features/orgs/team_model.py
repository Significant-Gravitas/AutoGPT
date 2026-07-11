"""Pydantic request/response models for workspace management."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateTeamRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    join_policy: str = "OPEN"  # OPEN or PRIVATE


class UpdateTeamRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    join_policy: str | None = None  # OPEN or PRIVATE


class TeamResponse(BaseModel):
    id: str
    name: str
    slug: str | None
    description: str | None
    is_default: bool
    join_policy: str
    org_id: str
    member_count: int
    is_member: bool
    created_at: datetime

    @staticmethod
    def from_db(
        ws,
        member_count: int = 0,
        is_member: bool = False,
        redact_description: bool = False,
    ) -> "TeamResponse":
        # redact_description hides the description of PRIVATE teams an org admin
        # can see for governance but isn't a member of ("governance without
        # surveillance"); name/member_count/join_policy stay visible.
        return TeamResponse(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            description=None if redact_description else ws.description,
            is_default=ws.isDefault,
            join_policy=ws.joinPolicy,
            org_id=ws.orgId,
            member_count=member_count,
            is_member=is_member,
            created_at=ws.createdAt,
        )


class TeamMemberResponse(BaseModel):
    id: str
    user_id: str
    email: str
    name: str | None
    is_admin: bool
    is_billing_manager: bool
    joined_at: datetime

    @staticmethod
    def from_db(member) -> "TeamMemberResponse":
        return TeamMemberResponse(
            id=member.id,
            user_id=member.userId,
            email=member.User.email if member.User else "",
            name=member.User.name if member.User else None,
            is_admin=member.isAdmin,
            is_billing_manager=member.isBillingManager,
            joined_at=member.joinedAt,
        )


class AddTeamMemberRequest(BaseModel):
    user_id: str
    is_admin: bool = False
    is_billing_manager: bool = False


class UpdateTeamMemberRequest(BaseModel):
    is_admin: bool | None = None
    is_billing_manager: bool | None = None
