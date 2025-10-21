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


class ExecutionDiagnosticsResponse(BaseModel):
    """Response model for execution diagnostics"""

    running_executions: int
    queued_executions_db: int
    queued_executions_rabbitmq: int
    timestamp: str


class AgentDiagnosticsResponse(BaseModel):
    """Response model for agent diagnostics"""

    total_agents: int
    active_agents: int
    agents_with_active_executions: int
    timestamp: str
