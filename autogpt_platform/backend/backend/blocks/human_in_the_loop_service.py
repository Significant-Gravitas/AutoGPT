"""
Service layer for Human In The Loop (HITL) functionality.
Separates business logic from database operations and block implementation.
"""

from typing import Any, Optional

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview
from pydantic import BaseModel

from backend.data.db import transaction
from backend.util.json import SafeJson


class ReviewDataStructure(BaseModel):
    """Structured representation of review data stored in the database."""

    data: Any
    message: str
    editable: bool


class ReviewResult(BaseModel):
    """Result of a review operation."""

    data: Any
    status: str
    message: str = ""


class HumanInTheLoopService:
    """
    Service class for managing Human In The Loop workflows.

    This service encapsulates all business logic related to HITL reviews,
    keeping database operations separate from the block implementation.
    """

    @staticmethod
    async def get_existing_review(node_exec_id: str) -> Optional[PendingHumanReview]:
        """
        Get an existing review for a node execution.

        Args:
            node_exec_id: The node execution ID to check

        Returns:
            The existing review if found, None otherwise
        """
        return await PendingHumanReview.prisma().find_unique(
            where={"nodeExecId": node_exec_id}
        )

    @staticmethod
    def extract_approved_data(review: PendingHumanReview) -> Any:
        """
        Extract approved data from a review record.

        Handles both structured and legacy data formats.

        Args:
            review: The approved review record

        Returns:
            The extracted data
        """
        try:
            # Try to parse as structured format
            review_structure = ReviewDataStructure.model_validate(review.data)
            return review_structure.data
        except Exception:
            # Fallback for legacy data format or corrupted data
            if isinstance(review.data, dict) and "data" in review.data:
                return review.data["data"]
            else:
                return review.data

    @staticmethod
    async def process_approved_review(
        review: PendingHumanReview, expected_data_type: type
    ) -> ReviewResult:
        """
        Process an approved review and clean up the database.

        Args:
            review: The approved review to process
            expected_data_type: The expected type for the data

        Returns:
            ReviewResult with the processed data
        """
        from backend.util.type import convert

        approved_data = HumanInTheLoopService.extract_approved_data(review)
        approved_data = convert(approved_data, expected_data_type)

        # Clean up the review record atomically
        async with transaction() as tx:
            await tx.pendinghumanreview.delete(where={"id": review.id})

        return ReviewResult(
            data=approved_data, status="approved", message=review.reviewMessage or ""
        )

    @staticmethod
    async def process_rejected_review(review: PendingHumanReview) -> ReviewResult:
        """
        Process a rejected review and clean up the database.

        Args:
            review: The rejected review to process

        Returns:
            ReviewResult with rejection details
        """
        # Clean up the review record atomically
        async with transaction() as tx:
            await tx.pendinghumanreview.delete(where={"id": review.id})

        return ReviewResult(
            data=None, status="rejected", message=review.reviewMessage or ""
        )

    @staticmethod
    async def create_or_update_review(
        user_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        data: Any,
        message: str,
        editable: bool,
    ) -> None:
        """
        Create a new review or update an existing one.

        Args:
            user_id: ID of the user who owns this review
            node_exec_id: ID of the node execution
            graph_exec_id: ID of the graph execution
            graph_id: ID of the graph template
            graph_version: Version of the graph template
            data: The data to be reviewed
            message: Instructions for the reviewer
            editable: Whether the data can be edited
        """
        review_data = {
            "data": data,
            "message": message,
            "editable": editable,
        }

        await PendingHumanReview.prisma().upsert(
            where={"nodeExecId": node_exec_id},
            data={
                "create": {
                    "userId": user_id,
                    "nodeExecId": node_exec_id,
                    "graphExecId": graph_exec_id,
                    "graphId": graph_id,
                    "graphVersion": graph_version,
                    "data": SafeJson(review_data),
                    "status": ReviewStatus.WAITING,
                },
                "update": {
                    "data": SafeJson(review_data),
                    "status": ReviewStatus.WAITING,
                },
            },
        )

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
        """
        Handle the complete review workflow.

        This is the main entry point that orchestrates the entire HITL process:
        1. Check for existing reviews
        2. Process approved/rejected reviews
        3. Create new reviews if needed

        Args:
            user_id: ID of the user who owns this review
            node_exec_id: ID of the node execution
            graph_exec_id: ID of the graph execution
            graph_id: ID of the graph template
            graph_version: Version of the graph template
            input_data: The data to be reviewed
            message: Instructions for the reviewer
            editable: Whether the data can be edited
            expected_data_type: Expected type for the output data

        Returns:
            ReviewResult if the review is complete, None if waiting for human input
        """
        # Check if there's already a review for this node execution
        existing_review = await HumanInTheLoopService.get_existing_review(node_exec_id)

        if existing_review:
            if existing_review.status == ReviewStatus.APPROVED:
                # Process the approved review
                return await HumanInTheLoopService.process_approved_review(
                    existing_review, expected_data_type
                )
            elif existing_review.status == ReviewStatus.REJECTED:
                # Process the rejected review
                return await HumanInTheLoopService.process_rejected_review(
                    existing_review
                )
            # If status is WAITING, continue to create/update logic

        # Create or update the pending review
        await HumanInTheLoopService.create_or_update_review(
            user_id=user_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            data=input_data,
            message=message,
            editable=editable,
        )

        # Return None to indicate that we're waiting for human input
        return None


class HITLValidationError(Exception):
    """Exception raised when HITL validation fails."""

    def __init__(self, message: str, review_message: str = ""):
        super().__init__(message)
        self.review_message = review_message
