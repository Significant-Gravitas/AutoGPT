from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

import prisma.enums
from pydantic import BaseModel, EmailStr

from backend.data.model import UserTransaction
from backend.util.models import Pagination

if TYPE_CHECKING:
    from backend.data.invited_user import BulkInvitedUsersResult, InvitedUserRecord


class UserHistoryResponse(BaseModel):
    """Response model for listings with version history"""

    history: list[UserTransaction]
    pagination: Pagination


class AddUserCreditsResponse(BaseModel):
    new_balance: int
    transaction_key: str


class CreateInvitedUserRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None


class InvitedUserResponse(BaseModel):
    id: str
    email: str
    status: prisma.enums.InvitedUserStatus
    auth_user_id: Optional[str] = None
    name: Optional[str] = None
    tally_understanding: Optional[dict[str, Any]] = None
    tally_status: prisma.enums.TallyComputationStatus
    tally_computed_at: Optional[datetime] = None
    tally_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, record: InvitedUserRecord) -> InvitedUserResponse:
        return cls.model_validate(record.model_dump())


class InvitedUsersResponse(BaseModel):
    invited_users: list[InvitedUserResponse]
    pagination: Pagination


class BulkInvitedUserRowResponse(BaseModel):
    row_number: int
    email: Optional[str] = None
    name: Optional[str] = None
    status: Literal["CREATED", "SKIPPED", "ERROR"]
    message: str
    invited_user: Optional[InvitedUserResponse] = None


class BulkInvitedUsersResponse(BaseModel):
    created_count: int
    skipped_count: int
    error_count: int
    results: list[BulkInvitedUserRowResponse]

    @classmethod
    def from_result(cls, result: BulkInvitedUsersResult) -> BulkInvitedUsersResponse:
        return cls(
            created_count=result.created_count,
            skipped_count=result.skipped_count,
            error_count=result.error_count,
            results=[
                BulkInvitedUserRowResponse(
                    row_number=row.row_number,
                    email=row.email,
                    name=row.name,
                    status=row.status,
                    message=row.message,
                    invited_user=(
                        InvitedUserResponse.from_record(row.invited_user)
                        if row.invited_user is not None
                        else None
                    ),
                )
                for row in result.results
            ],
        )
