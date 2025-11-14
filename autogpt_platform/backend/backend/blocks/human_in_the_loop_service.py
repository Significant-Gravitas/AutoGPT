"""
Service layer for Human In The Loop (HITL) functionality.
Separates business logic from database operations and block implementation.
"""

from typing import Any, Optional

from prisma.models import PendingHumanReview

from backend.data.human_review import (
    ReviewResult,
    extract_approved_data,
    get_pending_review_by_node,
    handle_review_workflow,
    process_approved_review,
    process_rejected_review,
)

# Re-export for backward compatibility
# These are now defined in the data layer


class HumanInTheLoopService:
    """
    Service class for managing Human In The Loop workflows.

    This service encapsulates all business logic related to HITL reviews,
    keeping database operations separate from the block implementation.
    """

    @staticmethod
    async def get_existing_review(
        node_exec_id: str, user_id: str
    ) -> Optional[PendingHumanReview]:
        """
        Get an existing review for a node execution.

        Args:
            node_exec_id: The node execution ID to check
            user_id: The user ID to validate ownership

        Returns:
            The existing review if found and owned by the user, None otherwise
        """
        return await get_pending_review_by_node(node_exec_id, user_id)

    @staticmethod
    def extract_approved_data(review: PendingHumanReview) -> Any:
        """Extract approved data from a review record (delegated to data layer)."""
        return extract_approved_data(review)

    @staticmethod
    async def process_approved_review(
        review: PendingHumanReview, expected_data_type: type
    ) -> ReviewResult:
        """Process an approved review (delegated to data layer)."""
        return await process_approved_review(review, expected_data_type)

    @staticmethod
    async def process_rejected_review(review: PendingHumanReview) -> ReviewResult:
        """Process a rejected review (delegated to data layer)."""
        return await process_rejected_review(review)

    # Removed: create_or_update_review is now handled directly by the data layer

    @staticmethod
    async def handle_review_workflow(
        user_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        input_data: Any,
        message: str,
        editable: bool,
        expected_data_type: type,
    ) -> Optional[ReviewResult]:
        """Handle the complete review workflow (delegated to data layer)."""
        return await handle_review_workflow(
            user_id=user_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            input_data=input_data,
            message=message,
            editable=editable,
            expected_data_type=expected_data_type,
        )
