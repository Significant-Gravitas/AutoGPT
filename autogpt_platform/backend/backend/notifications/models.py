from datetime import datetime
from enum import Enum
from typing import Annotated, Generic, Literal, Optional, TypeVar, Union, overload

from pydantic import BaseModel, Field


class BatchingStrategy(str, Enum):
    IMMEDIATE = "immediate"  # Send right away (errors, critical notifications)
    HOURLY = "hourly"  # Batch for up to an hour (usage reports)
    DAILY = "daily"  # Daily digest (summary notifications)
    BACKOFF = "backoff"  # Backoff strategy (exponential backoff)


class NotificationType(str, Enum):
    AGENT_RUN = "agent_run"
    ZERO_BALANCE = "zero_balance"
    LOW_BALANCE = "low_balance"
    BLOCK_EXECUTION_FAILED = "block_execution_failed"
    CONTINUOUS_AGENT_ERROR = "continuous_agent_error"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    MONTHLY_SUMMARY = "monthly_summary"

    @property
    def strategy(self) -> BatchingStrategy:
        BATCHING_RULES = {
            # These are batched by the notification service
            NotificationType.AGENT_RUN: BatchingStrategy.IMMEDIATE,
            # These are batched by the notification service, but with a backoff strategy
            NotificationType.ZERO_BALANCE: BatchingStrategy.BACKOFF,
            NotificationType.LOW_BALANCE: BatchingStrategy.BACKOFF,
            NotificationType.BLOCK_EXECUTION_FAILED: BatchingStrategy.BACKOFF,
            NotificationType.CONTINUOUS_AGENT_ERROR: BatchingStrategy.BACKOFF,
            # These aren't batched by the notification service, so we send them right away
            NotificationType.DAILY_SUMMARY: BatchingStrategy.IMMEDIATE,
            NotificationType.WEEKLY_SUMMARY: BatchingStrategy.IMMEDIATE,
            NotificationType.MONTHLY_SUMMARY: BatchingStrategy.IMMEDIATE,
        }
        return BATCHING_RULES.get(self, BatchingStrategy.HOURLY)

    @property
    def template(self) -> str:
        """Returns template name for this notification type"""
        return {
            NotificationType.AGENT_RUN: "agent_run.html",
            NotificationType.ZERO_BALANCE: "zero_balance.html",
            NotificationType.LOW_BALANCE: "low_balance.html",
            NotificationType.BLOCK_EXECUTION_FAILED: "block_failed.html",
            NotificationType.CONTINUOUS_AGENT_ERROR: "agent_error.html",
            NotificationType.DAILY_SUMMARY: "daily_summary.html",
            NotificationType.WEEKLY_SUMMARY: "weekly_summary.html",
            NotificationType.MONTHLY_SUMMARY: "monthly_summary.html",
        }[self]


T_co = TypeVar("T_co", bound="BaseNotificationData", covariant=True)


class BaseNotificationData(BaseModel):
    pass


class AgentRunData(BaseNotificationData):
    agent_name: str
    credits_used: float
    # remaining_balance: float
    execution_time: float
    graph_id: str
    node_count: int = Field(..., description="Number of nodes executed")


class ZeroBalanceData(BaseNotificationData):
    last_transaction: float
    last_transaction_time: datetime
    top_up_link: str


class LowBalanceData(BaseNotificationData):
    current_balance: float
    threshold_amount: float
    top_up_link: str
    recent_usage: float = Field(..., description="Usage in the last 24 hours")


class BlockExecutionFailedData(BaseNotificationData):
    block_name: str
    block_id: str
    error_message: str
    graph_id: str
    node_id: str
    execution_id: str


class ContinuousAgentErrorData(BaseNotificationData):
    agent_name: str
    error_message: str
    graph_id: str
    execution_id: str
    start_time: datetime
    error_time: datetime
    attempts: int = Field(..., description="Number of retry attempts made")


class BaseSummaryData(BaseNotificationData):
    total_credits_used: float
    total_executions: int
    most_used_agent: str
    total_execution_time: float
    successful_runs: int
    failed_runs: int
    average_execution_time: float
    cost_breakdown: dict[str, float]


class DailySummaryData(BaseSummaryData):
    date: datetime


class WeeklySummaryData(BaseSummaryData):
    start_date: datetime
    end_date: datetime
    week_number: int
    year: int


class MonthlySummaryData(BaseSummaryData):
    month: int
    year: int


NotificationData = Annotated[
    Union[
        AgentRunData,
        ZeroBalanceData,
        LowBalanceData,
        BlockExecutionFailedData,
        ContinuousAgentErrorData,
        MonthlySummaryData,
    ],
    Field(discriminator="type"),
]


class NotificationEvent(BaseModel, Generic[T_co]):
    user_id: str
    type: NotificationType
    data: T_co
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def strategy(self) -> BatchingStrategy:
        return self.type.strategy

    @property
    def template(self) -> str:
        return self.type.template


# Type-safe constructors
@overload
def create_notification(
    user_id: str, type: Literal[NotificationType.AGENT_RUN], data: AgentRunData
) -> NotificationEvent[AgentRunData]: ...


@overload
def create_notification(
    user_id: str, type: Literal[NotificationType.ZERO_BALANCE], data: ZeroBalanceData
) -> NotificationEvent[ZeroBalanceData]: ...


@overload
def create_notification(
    user_id: str, type: Literal[NotificationType.LOW_BALANCE], data: LowBalanceData
) -> NotificationEvent[LowBalanceData]: ...


@overload
def create_notification(
    user_id: str,
    type: Literal[NotificationType.BLOCK_EXECUTION_FAILED],
    data: BlockExecutionFailedData,
) -> NotificationEvent[BlockExecutionFailedData]: ...


@overload
def create_notification(
    user_id: str,
    type: Literal[NotificationType.CONTINUOUS_AGENT_ERROR],
    data: ContinuousAgentErrorData,
) -> NotificationEvent[ContinuousAgentErrorData]: ...


@overload
def create_notification(
    user_id: str,
    type: Literal[NotificationType.DAILY_SUMMARY],
    data: DailySummaryData,
) -> NotificationEvent[DailySummaryData]: ...


@overload
def create_notification(
    user_id: str,
    type: Literal[NotificationType.WEEKLY_SUMMARY],
    data: WeeklySummaryData,
) -> NotificationEvent[WeeklySummaryData]: ...


@overload
def create_notification(
    user_id: str,
    type: Literal[NotificationType.MONTHLY_SUMMARY],
    data: MonthlySummaryData,
) -> NotificationEvent[MonthlySummaryData]: ...


def create_notification(
    user_id: str, type: NotificationType, data: BaseNotificationData
) -> NotificationEvent[BaseNotificationData]:
    return NotificationEvent(user_id=user_id, type=type, data=data)


class NotificationBatch(BaseModel):
    user_id: str
    events: list[NotificationEvent]
    strategy: BatchingStrategy
    last_update: datetime = datetime.now()


class NotificationResult(BaseModel):
    success: bool
    message: Optional[str] = None
