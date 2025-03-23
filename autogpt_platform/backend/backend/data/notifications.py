import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any, Generic, Optional, TypeVar, Union

from prisma import Json
from prisma.enums import NotificationType
from prisma.models import NotificationEvent, UserNotificationBatch
from prisma.types import UserNotificationBatchWhereInput

# from backend.notifications.models import NotificationEvent
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.server.v2.store.exceptions import DatabaseError

from .db import transaction

logger = logging.getLogger(__name__)


NotificationDataType_co = TypeVar(
    "NotificationDataType_co", bound="BaseNotificationData", covariant=True
)
SummaryParamsType_co = TypeVar(
    "SummaryParamsType_co", bound="BaseSummaryParams", covariant=True
)


class QueueType(Enum):
    IMMEDIATE = "immediate"  # Send right away (errors, critical notifications)
    BATCH = "batch"  # Batch for up to an hour (usage reports)
    SUMMARY = "summary"  # Daily digest (summary notifications)
    BACKOFF = "backoff"  # Backoff strategy (exponential backoff)
    ADMIN = "admin"  # Admin notifications (errors, critical notifications)


class BaseNotificationData(BaseModel):
    class Config:
        extra = "allow"


class AgentRunData(BaseNotificationData):
    agent_name: str
    credits_used: float
    execution_time: float
    node_count: int = Field(..., description="Number of nodes executed")
    graph_id: str
    outputs: list[dict[str, Any]] = Field(..., description="Outputs of the agent")


class ZeroBalanceData(BaseNotificationData):
    last_transaction: float
    last_transaction_time: datetime
    top_up_link: str

    @field_validator("last_transaction_time")
    @classmethod
    def validate_timezone(cls, value: datetime):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class LowBalanceData(BaseNotificationData):
    agent_name: str = Field(..., description="Name of the agent")
    current_balance: float = Field(
        ..., description="Current balance in credits (100 = $1)"
    )
    billing_page_link: str = Field(..., description="Link to billing page")
    shortfall: float = Field(..., description="Amount of credits needed to continue")


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

    @field_validator("start_time", "error_time")
    @classmethod
    def validate_timezone(cls, value: datetime):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class BaseSummaryData(BaseNotificationData):
    total_credits_used: float
    total_executions: int
    most_used_agent: str
    total_execution_time: float
    successful_runs: int
    failed_runs: int
    average_execution_time: float
    cost_breakdown: dict[str, float]


class BaseSummaryParams(BaseModel):
    pass


class DailySummaryParams(BaseSummaryParams):
    date: datetime

    @field_validator("date")
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class WeeklySummaryParams(BaseSummaryParams):
    start_date: datetime
    end_date: datetime

    @field_validator("start_date", "end_date")
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class DailySummaryData(BaseSummaryData):
    date: datetime

    @field_validator("date")
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class WeeklySummaryData(BaseSummaryData):
    start_date: datetime
    end_date: datetime

    @field_validator("start_date", "end_date")
    def validate_timezone(cls, value):
        if value.tzinfo is None:
            raise ValueError("datetime must have timezone information")
        return value


class MonthlySummaryData(BaseNotificationData):
    month: int
    year: int


class RefundRequestData(BaseNotificationData):
    user_id: str
    user_name: str
    user_email: str
    transaction_id: str
    refund_request_id: str
    reason: str
    amount: float
    balance: int


NotificationData = Annotated[
    Union[
        AgentRunData,
        ZeroBalanceData,
        LowBalanceData,
        BlockExecutionFailedData,
        ContinuousAgentErrorData,
        MonthlySummaryData,
        WeeklySummaryData,
        DailySummaryData,
        RefundRequestData,
        BaseSummaryData,
    ],
    Field(discriminator="type"),
]


class NotificationEventDTO(BaseModel):
    user_id: str
    type: NotificationType
    data: dict
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    retry_count: int = 0


class SummaryParamsEventDTO(BaseModel):
    user_id: str
    type: NotificationType
    data: dict
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class NotificationEventModel(BaseModel, Generic[NotificationDataType_co]):
    user_id: str
    type: NotificationType
    data: NotificationDataType_co
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def strategy(self) -> QueueType:
        return NotificationTypeOverride(self.type).strategy

    @field_validator("type", mode="before")
    def uppercase_type(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v

    @property
    def template(self) -> str:
        return NotificationTypeOverride(self.type).template


class SummaryParamsEventModel(BaseModel, Generic[SummaryParamsType_co]):
    user_id: str
    type: NotificationType
    data: SummaryParamsType_co
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


def get_notif_data_type(
    notification_type: NotificationType,
) -> type[BaseNotificationData]:
    return {
        NotificationType.AGENT_RUN: AgentRunData,
        NotificationType.ZERO_BALANCE: ZeroBalanceData,
        NotificationType.LOW_BALANCE: LowBalanceData,
        NotificationType.BLOCK_EXECUTION_FAILED: BlockExecutionFailedData,
        NotificationType.CONTINUOUS_AGENT_ERROR: ContinuousAgentErrorData,
        NotificationType.DAILY_SUMMARY: DailySummaryData,
        NotificationType.WEEKLY_SUMMARY: WeeklySummaryData,
        NotificationType.MONTHLY_SUMMARY: MonthlySummaryData,
        NotificationType.REFUND_REQUEST: RefundRequestData,
        NotificationType.REFUND_PROCESSED: RefundRequestData,
    }[notification_type]


def get_summary_params_type(
    notification_type: NotificationType,
) -> type[BaseSummaryParams]:
    return {
        NotificationType.DAILY_SUMMARY: DailySummaryParams,
        NotificationType.WEEKLY_SUMMARY: WeeklySummaryParams,
    }[notification_type]


class NotificationBatch(BaseModel):
    user_id: str
    events: list[NotificationEvent]
    strategy: QueueType
    last_update: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class NotificationResult(BaseModel):
    success: bool
    message: Optional[str] = None


class NotificationTypeOverride:
    def __init__(self, notification_type: NotificationType):
        self.notification_type = notification_type

    @property
    def strategy(self) -> QueueType:
        BATCHING_RULES = {
            # These are batched by the notification service
            NotificationType.AGENT_RUN: QueueType.BATCH,
            # These are batched by the notification service, but with a backoff strategy
            NotificationType.ZERO_BALANCE: QueueType.BACKOFF,
            NotificationType.LOW_BALANCE: QueueType.IMMEDIATE,
            NotificationType.BLOCK_EXECUTION_FAILED: QueueType.BACKOFF,
            NotificationType.CONTINUOUS_AGENT_ERROR: QueueType.BACKOFF,
            NotificationType.DAILY_SUMMARY: QueueType.SUMMARY,
            NotificationType.WEEKLY_SUMMARY: QueueType.SUMMARY,
            NotificationType.MONTHLY_SUMMARY: QueueType.SUMMARY,
            NotificationType.REFUND_REQUEST: QueueType.ADMIN,
            NotificationType.REFUND_PROCESSED: QueueType.ADMIN,
        }
        return BATCHING_RULES.get(self.notification_type, QueueType.IMMEDIATE)

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
            NotificationType.REFUND_REQUEST: "refund_request.html",
            NotificationType.REFUND_PROCESSED: "refund_processed.html",
        }[self.notification_type]

    @property
    def subject(self) -> str:
        return {
            NotificationType.AGENT_RUN: "Agent Run Report",
            NotificationType.ZERO_BALANCE: "You're out of credits!",
            NotificationType.LOW_BALANCE: "Low Balance Warning!",
            NotificationType.BLOCK_EXECUTION_FAILED: "Uh oh! Block Execution Failed",
            NotificationType.CONTINUOUS_AGENT_ERROR: "Shoot! Continuous Agent Error",
            NotificationType.DAILY_SUMMARY: "Here's your daily summary!",
            NotificationType.WEEKLY_SUMMARY: "Look at all the cool stuff you did last week!",
            NotificationType.MONTHLY_SUMMARY: "We did a lot this month!",
            NotificationType.REFUND_REQUEST: "[ACTION REQUIRED] You got a ${{data.amount / 100}} refund request from {{data.user_name}}",
            NotificationType.REFUND_PROCESSED: "Refund for ${{data.amount / 100}} to {{data.user_name}} has been processed",
        }[self.notification_type]


class NotificationPreferenceDTO(BaseModel):
    email: EmailStr = Field(..., description="User's email address")
    preferences: dict[NotificationType, bool] = Field(
        ..., description="Which notifications the user wants"
    )
    daily_limit: int = Field(..., description="Max emails per day")


class NotificationPreference(BaseModel):
    user_id: str
    email: EmailStr
    preferences: dict[NotificationType, bool] = Field(
        default_factory=dict, description="Which notifications the user wants"
    )
    daily_limit: int = 10  # Max emails per day
    emails_sent_today: int = 0
    last_reset_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class UserNotificationEventDTO(BaseModel):
    type: NotificationType
    data: dict
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_db(model: NotificationEvent) -> "UserNotificationEventDTO":
        return UserNotificationEventDTO(
            type=model.type,
            data=dict(model.data),
            created_at=model.createdAt,
            updated_at=model.updatedAt,
        )


class UserNotificationBatchDTO(BaseModel):
    user_id: str
    type: NotificationType
    notifications: list[UserNotificationEventDTO]
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_db(model: UserNotificationBatch) -> "UserNotificationBatchDTO":
        return UserNotificationBatchDTO(
            user_id=model.userId,
            type=model.type,
            notifications=[
                UserNotificationEventDTO.from_db(notification)
                for notification in model.notifications or []
            ],
            created_at=model.createdAt,
            updated_at=model.updatedAt,
        )


def get_batch_delay(notification_type: NotificationType) -> timedelta:
    return {
        NotificationType.AGENT_RUN: timedelta(minutes=60),
        NotificationType.ZERO_BALANCE: timedelta(minutes=60),
        NotificationType.LOW_BALANCE: timedelta(minutes=60),
        NotificationType.BLOCK_EXECUTION_FAILED: timedelta(minutes=60),
        NotificationType.CONTINUOUS_AGENT_ERROR: timedelta(minutes=60),
    }[notification_type]


async def create_or_add_to_user_notification_batch(
    user_id: str,
    notification_type: NotificationType,
    notification_data: NotificationEventModel,
) -> UserNotificationBatchDTO:
    try:
        logger.info(
            f"Creating or adding to notification batch for {user_id} with type {notification_type} and data {notification_data}"
        )

        # Serialize the data
        json_data: Json = Json(notification_data.data.model_dump())

        # First try to find existing batch
        existing_batch = await UserNotificationBatch.prisma().find_unique(
            where={
                "userId_type": {
                    "userId": user_id,
                    "type": notification_type,
                }
            },
            include={"notifications": True},
        )

        if not existing_batch:
            async with transaction() as tx:
                notification_event = await tx.notificationevent.create(
                    data={
                        "type": notification_type,
                        "data": json_data,
                    }
                )

                # Create new batch
                resp = await tx.usernotificationbatch.create(
                    data={
                        "userId": user_id,
                        "type": notification_type,
                        "notifications": {"connect": [{"id": notification_event.id}]},
                    },
                    include={"notifications": True},
                )
                return UserNotificationBatchDTO.from_db(resp)
        else:
            async with transaction() as tx:
                notification_event = await tx.notificationevent.create(
                    data={
                        "type": notification_type,
                        "data": json_data,
                        "UserNotificationBatch": {"connect": {"id": existing_batch.id}},
                    }
                )
                # Add to existing batch
                resp = await tx.usernotificationbatch.update(
                    where={"id": existing_batch.id},
                    data={
                        "notifications": {"connect": [{"id": notification_event.id}]}
                    },
                    include={"notifications": True},
                )
            if not resp:
                raise DatabaseError(
                    f"Failed to add notification event {notification_event.id} to existing batch {existing_batch.id}"
                )
            return UserNotificationBatchDTO.from_db(resp)
    except Exception as e:
        raise DatabaseError(
            f"Failed to create or add to notification batch for user {user_id} and type {notification_type}: {e}"
        ) from e


async def get_user_notification_oldest_message_in_batch(
    user_id: str,
    notification_type: NotificationType,
) -> UserNotificationEventDTO | None:
    try:
        batch = await UserNotificationBatch.prisma().find_first(
            where={"userId": user_id, "type": notification_type},
            include={"notifications": True},
        )
        if not batch:
            return None
        if not batch.notifications:
            return None
        sorted_notifications = sorted(batch.notifications, key=lambda x: x.createdAt)

        return (
            UserNotificationEventDTO.from_db(sorted_notifications[0])
            if sorted_notifications
            else None
        )
    except Exception as e:
        raise DatabaseError(
            f"Failed to get user notification last message in batch for user {user_id} and type {notification_type}: {e}"
        ) from e


async def empty_user_notification_batch(
    user_id: str, notification_type: NotificationType
) -> None:
    try:
        async with transaction() as tx:
            await tx.notificationevent.delete_many(
                where={
                    "UserNotificationBatch": {
                        "is": {"userId": user_id, "type": notification_type}
                    }
                }
            )

            await tx.usernotificationbatch.delete_many(
                where=UserNotificationBatchWhereInput(
                    userId=user_id,
                    type=notification_type,
                )
            )
    except Exception as e:
        raise DatabaseError(
            f"Failed to empty user notification batch for user {user_id} and type {notification_type}: {e}"
        ) from e


async def get_user_notification_batch(
    user_id: str,
    notification_type: NotificationType,
) -> UserNotificationBatchDTO | None:
    try:
        batch = await UserNotificationBatch.prisma().find_first(
            where={"userId": user_id, "type": notification_type},
            include={"notifications": True},
        )
        return UserNotificationBatchDTO.from_db(batch) if batch else None
    except Exception as e:
        raise DatabaseError(
            f"Failed to get user notification batch for user {user_id} and type {notification_type}: {e}"
        ) from e


async def get_all_batches_by_type(
    notification_type: NotificationType,
) -> list[UserNotificationBatchDTO]:
    try:
        batches = await UserNotificationBatch.prisma().find_many(
            where={
                "type": notification_type,
                "notifications": {
                    "some": {}  # Only return batches with at least one notification
                },
            },
            include={"notifications": True},
        )
        return [UserNotificationBatchDTO.from_db(batch) for batch in batches]
    except Exception as e:
        raise DatabaseError(
            f"Failed to get all batches by type {notification_type}: {e}"
        ) from e
