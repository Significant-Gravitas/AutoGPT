from datetime import datetime

from prisma.enums import CreditTransactionType
from pydantic import BaseModel

from backend.server.model import Pagination


class UserHistory(BaseModel):
    user_id: str
    user_email: str
    amount: int
    date: datetime
    current_balance: int
    reason: str
    admin_email: str
    type: CreditTransactionType
    extra_data: str | None = None


class UserHistoryResponse(BaseModel):
    """Response model for listings with version history"""

    history: list[UserHistory]
    pagination: Pagination


class AddUserCreditsResponse(BaseModel):
    new_balance: int
    transaction_key: str
