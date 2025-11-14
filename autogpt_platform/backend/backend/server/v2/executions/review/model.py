import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PendingHumanReviewResponse(BaseModel):
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
    data: Any = Field(description="Data waiting for review")
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
    reviewed_data: Any | None = Field(
        description="Modified data (only for approve action)", default=None
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
