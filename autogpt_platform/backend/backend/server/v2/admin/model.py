from datetime import datetime

from backend.server.model import Pagination
from pydantic import BaseModel


class GrantHistory(BaseModel):
    user_id: str
    user_email: str
    amount: int
    date: datetime
    reason: str
    admin_email: str


class GrantHistoryResponse(BaseModel):
    """Response model for listings with version history"""

    grants: list[GrantHistory]
    pagination: Pagination


class UserBalance(BaseModel):
    user_id: str
    user_email: str
    balance: int


class UserBalanceResponse(BaseModel):
    """Response model for listings with version history"""

    balances: list[UserBalance]
    pagination: Pagination


class AddUserCreditsResponse(BaseModel):
    success: bool
    new_balance: int
    transaction_key: str
