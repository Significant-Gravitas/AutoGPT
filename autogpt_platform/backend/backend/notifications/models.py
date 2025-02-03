from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr


class BatchingStrategy(str, Enum):
    IMMEDIATE = "immediate"  # Send right away (errors, critical notifications)
    HOURLY = "hourly"  # Batch for up to an hour (usage reports)
    DAILY = "daily"  # Daily digest (summary notifications)


class NotificationType(str, Enum):
    AGENT_RUN = "agent_run"  # BatchingStrategy.HOURLY
    ZERO_BALANCE = "zero_balance"  # BatchingStrategy.IMMEDIATE
    LOW_BALANCE = "low_balance"  # BatchingStrategy.IMMEDIATE
    BLOCK_EXECUTION_FAILED = "block_execution_failed"  # BatchingStrategy.IMMEDIATE
    CONTINUOUS_AGENT_ERROR = "continuous_agent_error"  # BatchingStrategy.IMMEDIATE

    @property
    def strategy(self) -> BatchingStrategy:
        BATCHING_RULES = {
            NotificationType.AGENT_RUN: BatchingStrategy.HOURLY,
            NotificationType.ZERO_BALANCE: BatchingStrategy.IMMEDIATE,
            NotificationType.LOW_BALANCE: BatchingStrategy.IMMEDIATE,
            NotificationType.BLOCK_EXECUTION_FAILED: BatchingStrategy.IMMEDIATE,
            NotificationType.CONTINUOUS_AGENT_ERROR: BatchingStrategy.IMMEDIATE,
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
        }[self]


class NotificationEvent(BaseModel):
    user_id: str
    type: NotificationType
    data: dict
    created_at: datetime = datetime.now()

    @property
    def strategy(self) -> BatchingStrategy:
        return self.type.strategy

    @property
    def template(self) -> str:
        return self.type.template


class NotificationBatch(BaseModel):
    user_id: str
    events: list[NotificationEvent]
    strategy: BatchingStrategy
    last_update: datetime = datetime.now()


class NotificationResult(BaseModel):
    success: bool
    message: Optional[str] = None


class NotificationPreference(BaseModel):
    """User's notification preferences"""

    user_id: str
    email: EmailStr
    preferences: dict[NotificationType, bool] = {}  # Which notifications they want
    daily_limit: int = 10  # Max emails per day
    emails_sent_today: int = 0
    last_reset_date: datetime = datetime.now()
