from datetime import datetime
from typing import Any, Literal, Optional

import prisma.enums
from pydantic import BaseModel, EmailStr

from backend.data.model import UserTransaction
from backend.util.models import Pagination


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
