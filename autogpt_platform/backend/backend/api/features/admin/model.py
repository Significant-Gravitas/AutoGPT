from pydantic import BaseModel

from backend.data.model import UserTransaction
from backend.util.models import Pagination


class UserHistoryResponse(BaseModel):
    """Response model for listings with version history"""

    history: list[UserTransaction]
    pagination: Pagination


class AddUserCreditsResponse(BaseModel):
    new_balance: int
    transaction_key: str
