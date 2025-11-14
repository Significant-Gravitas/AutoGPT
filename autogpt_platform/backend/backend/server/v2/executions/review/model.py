from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PendingHumanReviewResponse(BaseModel):
    """Response model for pending human review."""

    id: str = Field(description="Unique ID of the pending review")
    user_id: str = Field(description="User ID associated with the review")
    node_exec_id: str = Field(description="Node execution ID")
    graph_exec_id: str = Field(description="Graph execution ID")
    graph_id: str = Field(description="Graph ID")
    graph_version: int = Field(description="Graph version")
    data: Any = Field(description="Data waiting for review")
    status: Literal["WAITING", "APPROVED", "REJECTED"] = Field(
        description="Review status"
    )
    review_message: str | None = Field(
        description="Optional message from the reviewer", default=None
    )
    created_at: datetime = Field(description="When the review was created")
    updated_at: datetime | None = Field(
        description="When the review was last updated", default=None
    )
    reviewed_at: datetime | None = Field(
        description="When the review was completed", default=None
    )


class ReviewActionRequest(BaseModel):
    """Request model for reviewing data."""

    action: Literal["approve", "reject"] = Field(description="Action to take")
    reviewed_data: Any | None = Field(
        description="Modified data (only for approve action)", default=None
    )
    message: str | None = Field(
        description="Optional message from the reviewer", default=None
    )
