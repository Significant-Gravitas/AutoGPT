import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from prisma.models import PendingHumanReview

# SafeJson-compatible type alias for review data
SafeJsonData = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


class PendingReviewData(BaseModel):
    """Data structure for pending human review stored in database.

    This represents the structured format of the 'data' field in PendingHumanReviewResponse
    when it contains review-specific metadata along with the actual data payload.

    Attributes:
        data: The actual data payload awaiting review
        message: Instructions or context message for the reviewer
        editable: Whether the reviewer is allowed to modify the data
    """

    data: SafeJsonData = Field(description="The actual data payload awaiting review")
    message: str = Field(description="Instructions or context message for the reviewer")
    editable: bool = Field(
        description="Whether the reviewer is allowed to modify the data"
    )


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
        data: The data payload awaiting review (can be any JSON structure)
        status: Current review status (WAITING, APPROVED, or REJECTED)
        review_message: Optional message from the reviewer
        created_at: Timestamp when review was created
        updated_at: Timestamp when review was last modified
        reviewed_at: Timestamp when review was completed (if applicable)
    """

    id: str = Field(description="Unique ID of the pending review")
    user_id: str = Field(description="User ID associated with the review")
    node_exec_id: str = Field(description="Node execution ID")
    graph_exec_id: str = Field(description="Graph execution ID")
    graph_id: str = Field(description="Graph ID")
    graph_version: int = Field(description="Graph version")
    data: PendingReviewData = Field(description="Structured data awaiting human review")
    status: Literal["WAITING", "APPROVED", "REJECTED"] = Field(
        description="Review status"
    )
    review_message: str | None = Field(
        description="Optional message from the reviewer", default=None
    )
    was_edited: bool | None = Field(
        description="Whether the data was modified during review", default=None
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

        Handles parsing of the data field from Json to PendingReviewData,
        with fallback for legacy data formats.
        """
        # Parse data as PendingReviewData
        try:
            parsed_data = PendingReviewData.model_validate(review.data)
        except Exception:
            # Fallback for legacy data - create PendingReviewData wrapper
            parsed_data = PendingReviewData(
                data=review.data,
                message="",  # Legacy data has no message
                editable=True,  # Default to editable for backward compatibility
            )

        # Convert status to literal type
        from typing import cast

        from prisma.enums import ReviewStatus

        status_map = {
            ReviewStatus.WAITING: "WAITING",
            ReviewStatus.APPROVED: "APPROVED",
            ReviewStatus.REJECTED: "REJECTED",
        }
        converted_status = cast(
            Literal["WAITING", "APPROVED", "REJECTED"], status_map[review.status]
        )

        return cls(
            id=review.id,
            user_id=review.userId,
            node_exec_id=review.nodeExecId,
            graph_exec_id=review.graphExecId,
            graph_id=review.graphId,
            graph_version=review.graphVersion,
            data=parsed_data,
            status=converted_status,
            review_message=review.reviewMessage,
            was_edited=review.wasEdited,
            created_at=review.createdAt,
            updated_at=review.updatedAt,
            reviewed_at=review.reviewedAt,
        )


class ReviewActionRequest(BaseModel):
    """Request model for reviewing data in a Human-in-the-Loop workflow.

    Represents a user's decision and optional modifications for pending review data.
    Supports both approval (with optional data modifications) and rejection actions.

    Validation Features:
    - Ensures reviewed_data is only provided for approve actions
    - Validates JSON serializability of reviewed data
    - Enforces size limits (1MB) to prevent DoS attacks
    - Checks nesting depth to prevent infinite recursion
    - Sanitizes empty review messages

    Attributes:
        action: The review decision - either "approve" or "reject"
        reviewed_data: Optional modified data (only valid for approve action)
        message: Optional message from the reviewer (max 2000 chars)
    """

    action: Literal["approve", "reject"] = Field(description="Action to take")
    reviewed_data: SafeJsonData | None = Field(
        description="Modified data (only for approve action, must be SafeJson compatible)",
        default=None,
    )
    message: str | None = Field(
        description="Optional message from the reviewer", default=None, max_length=2000
    )

    @field_validator("reviewed_data")
    @classmethod
    def validate_reviewed_data(cls, v):
        """Validate that reviewed_data is safe and properly structured.

        Performs comprehensive validation to prevent security issues and ensure
        data integrity in the Human-in-the-Loop workflow.

        Security Checks:
        - Validates JSON serializability to prevent injection attacks
        - Enforces 1MB size limit to prevent DoS attacks
        - Limits nesting depth to 10 levels to prevent stack overflow
        - Ensures data conforms to SafeJson compatible types

        Args:
            v: The reviewed_data value to validate

        Returns:
            The validated reviewed_data value

        Raises:
            ValueError: If validation fails for any security or format reason

        Note:
            The action consistency check is handled by model_validator
        """
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

    @model_validator(mode="before")
    def validate_action_data_consistency(cls, values):
        """Validate consistency between action and reviewed_data.

        Ensures that reviewed_data is only provided when action is 'approve',
        which makes the API more predictable and prevents confusion.

        Args:
            values: All field values from the model

        Returns:
            The validated field values

        Raises:
            ValueError: If reviewed_data is provided with reject action
        """
        action = values.get("action")
        reviewed_data = values.get("reviewed_data")

        if action == "reject" and reviewed_data is not None:
            raise ValueError(
                "reviewed_data should not be provided when action is 'reject'"
            )

        return values

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        """Validate and sanitize review message.

        Ensures review messages are properly formatted and not just whitespace.
        Empty or whitespace-only messages are converted to None for consistency.

        Args:
            v: The message value to validate

        Returns:
            Sanitized message string or None if empty/whitespace
        """
        if v is not None and len(v.strip()) == 0:
            return None
        return v


class ReviewActionResponse(BaseModel):
    """Response model for review action completion.

    Confirms that a review action (approve/reject) was successfully processed.

    Attributes:
        status: Always "success" to indicate successful processing
        action: The action that was performed ("approve" or "reject")
    """

    status: Literal["success"] = Field(
        description="Operation status", default="success"
    )
    action: Literal["approve", "reject"] = Field(
        description="The action that was performed"
    )
