import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Union

from prisma.enums import ReviewStatus
from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from prisma.models import PendingHumanReview

# SafeJson-compatible type alias for review data
SafeJsonData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class PendingHumanReviewModel(BaseModel):
    """Response model for pending human review data.

    Represents a human review request that is awaiting user action.
    Contains all necessary information for a user to review and approve
    or reject data from a Human-in-the-Loop block execution.

    Attributes:
        id: Unique identifier for the review record
        user_id: ID of the user who must perform the review
        node_exec_id: ID of the node execution that created this review
        graph_exec_id: ID of the graph execution containing the node
        graph_id: ID of the graph template being executed
        graph_version: Version number of the graph template
        payload: The actual data payload awaiting review
        instructions: Instructions or message for the reviewer
        editable: Whether the reviewer can edit the data
        status: Current review status (WAITING, APPROVED, or REJECTED)
        review_message: Optional message from the reviewer
        created_at: Timestamp when review was created
        updated_at: Timestamp when review was last modified
        reviewed_at: Timestamp when review was completed (if applicable)
    """

    node_exec_id: str = Field(description="Node execution ID (primary key)")
    user_id: str = Field(description="User ID associated with the review")
    graph_exec_id: str = Field(description="Graph execution ID")
    graph_id: str = Field(description="Graph ID")
    graph_version: int = Field(description="Graph version")
    payload: SafeJsonData = Field(description="The actual data payload awaiting review")
    instructions: str | None = Field(
        description="Instructions or message for the reviewer", default=None
    )
    editable: bool = Field(description="Whether the reviewer can edit the data")
    status: ReviewStatus = Field(description="Review status")
    review_message: str | None = Field(
        description="Optional message from the reviewer", default=None
    )
    was_edited: bool | None = Field(
        description="Whether the data was modified during review", default=None
    )
    processed: bool = Field(
        description="Whether the review result has been processed by the execution engine",
        default=False,
    )
    created_at: datetime = Field(description="When the review was created")
    updated_at: datetime | None = Field(
        description="When the review was last updated", default=None
    )
    reviewed_at: datetime | None = Field(
        description="When the review was completed", default=None
    )

    @classmethod
    def from_db(cls, review: "PendingHumanReview") -> "PendingHumanReviewModel":
        """
        Convert a database model to a response model.

        Uses the new flat database structure with separate columns for
        payload, instructions, and editable flag.

        Handles invalid data gracefully by using safe defaults.
        """
        return cls(
            node_exec_id=review.nodeExecId,
            user_id=review.userId,
            graph_exec_id=review.graphExecId,
            graph_id=review.graphId,
            graph_version=review.graphVersion,
            payload=review.payload,
            instructions=review.instructions,
            editable=review.editable,
            status=review.status,
            review_message=review.reviewMessage,
            was_edited=review.wasEdited,
            processed=review.processed,
            created_at=review.createdAt,
            updated_at=review.updatedAt,
            reviewed_at=review.reviewedAt,
        )


class ReviewItem(BaseModel):
    """Single review item for processing."""

    node_exec_id: str = Field(description="Node execution ID to review")
    approved: bool = Field(
        description="Whether this review is approved (True) or rejected (False)"
    )
    message: str | None = Field(
        None, description="Optional review message", max_length=2000
    )
    reviewed_data: SafeJsonData | None = Field(
        None, description="Optional edited data (ignored if approved=False)"
    )

    @field_validator("reviewed_data")
    @classmethod
    def validate_reviewed_data(cls, v):
        """Validate that reviewed_data is safe and properly structured."""
        if v is None:
            return v

        # Validate SafeJson compatibility
        def validate_safejson_type(obj):
            """Ensure object only contains SafeJson compatible types."""
            if obj is None:
                return True
            elif isinstance(obj, (str, int, float, bool)):
                return True
            elif isinstance(obj, dict):
                return all(
                    isinstance(k, str) and validate_safejson_type(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, list):
                return all(validate_safejson_type(item) for item in obj)
            else:
                return False

        if not validate_safejson_type(v):
            raise ValueError("reviewed_data contains non-SafeJson compatible types")

        # Validate data size to prevent DoS attacks
        try:
            json_str = json.dumps(v)
            if len(json_str) > 1000000:  # 1MB limit
                raise ValueError("reviewed_data is too large (max 1MB)")
        except (TypeError, ValueError) as e:
            raise ValueError(f"reviewed_data must be JSON serializable: {str(e)}")

        # Ensure no dangerous nested structures (prevent infinite recursion)
        def check_depth(obj, max_depth=10, current_depth=0):
            """Recursively check object nesting depth to prevent stack overflow attacks."""
            if current_depth > max_depth:
                raise ValueError("reviewed_data has excessive nesting depth")

            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, max_depth, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, max_depth, current_depth + 1)

        check_depth(v)
        return v

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate and sanitize review message."""
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class ReviewRequest(BaseModel):
    """Request model for processing ALL pending reviews for an execution.

    This request must include ALL pending reviews for a graph execution.
    Each review will be either approved (with optional data modifications)
    or rejected (data ignored). The execution will resume only after ALL reviews are processed.
    """

    reviews: List[ReviewItem] = Field(
        description="All reviews with their approval status, data, and messages"
    )

    @model_validator(mode="after")
    def validate_review_completeness(self):
        """Validate that we have at least one review to process and no duplicates."""
        if not self.reviews:
            raise ValueError("At least one review must be provided")

        # Ensure no duplicate node_exec_ids
        node_ids = [review.node_exec_id for review in self.reviews]
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in set(node_ids) if node_ids.count(nid) > 1]
            raise ValueError(f"Duplicate review IDs found: {', '.join(duplicates)}")

        return self


class ReviewResponse(BaseModel):
    """Response from review endpoint."""

    approved_count: int = Field(description="Number of reviews successfully approved")
    rejected_count: int = Field(description="Number of reviews successfully rejected")
    failed_count: int = Field(description="Number of reviews that failed processing")
    error: str | None = Field(None, description="Error message if operation failed")
