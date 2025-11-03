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

    # Current execution state
    running_executions: int
    queued_executions_db: int
    queued_executions_rabbitmq: int
    cancel_queue_depth: int

    # Orphaned execution detection
    orphaned_running: int
    orphaned_queued: int

    # Failure metrics
    failed_count_1h: int
    failed_count_24h: int
    failure_rate_24h: float

    # Long-running detection
    stuck_running_24h: int
    stuck_running_1h: int
    oldest_running_hours: float | None

    # Stuck queued detection
    stuck_queued_1h: int
    queued_never_started: int

    # Throughput metrics
    completed_1h: int
    completed_24h: int
    throughput_per_hour: float

    timestamp: str


class AgentDiagnosticsResponse(BaseModel):
    """Response model for agent diagnostics"""

    agents_with_active_executions: int
    timestamp: str
