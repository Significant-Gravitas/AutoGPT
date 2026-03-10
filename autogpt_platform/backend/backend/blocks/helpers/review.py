"""
Shared helpers for Human-In-The-Loop (HITL) review functionality.
Used by both the dedicated HumanInTheLoopBlock and blocks that require human review.
"""

import logging
from typing import Any, Optional

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.data.human_review import ReviewResult
from backend.util.clients import get_database_manager_async_client

logger = logging.getLogger(__name__)


class ReviewDecision(BaseModel):
    """Result of a review decision."""

    should_proceed: bool
    message: str
    review_result: ReviewResult


class HITLReviewHelper:
    """Helper class for Human-In-The-Loop review operations."""

    @staticmethod
    async def check_approval(**kwargs) -> Optional[ReviewResult]:
        """Check if there's an existing approval for this node execution."""
        return await get_database_manager_async_client().check_approval(**kwargs)

    @staticmethod
    async def get_or_create_human_review(**kwargs) -> Optional[ReviewResult]:
        """Create or retrieve a human review from the database."""
        return await get_database_manager_async_client().get_or_create_human_review(
            **kwargs
        )

    @staticmethod
    async def update_node_execution_status(**kwargs) -> None:
        """Update the execution status of a node."""
        from backend.executor.manager import async_update_node_execution_status

        await async_update_node_execution_status(
            db_client=get_database_manager_async_client(), **kwargs
        )

    @staticmethod
    async def update_review_processed_status(
        node_exec_id: str, processed: bool
    ) -> None:
        """Update the processed status of a review."""
        return await get_database_manager_async_client().update_review_processed_status(
            node_exec_id, processed
        )

    @staticmethod
    async def _handle_review_request(
        input_data: Any,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        block_name: str = "Block",
        editable: bool = False,
    ) -> Optional[ReviewResult]:
        """
        Handle a review request for a block that requires human review.

        Args:
            input_data: The input data to be reviewed
            user_id: ID of the user requesting the review
            node_id: ID of the node in the graph definition
            node_exec_id: ID of the node execution
            graph_exec_id: ID of the graph execution
            graph_id: ID of the graph
            graph_version: Version of the graph
            block_name: Name of the block requesting review
            editable: Whether the reviewer can edit the data

        Returns:
            ReviewResult if review is complete, None if waiting for human input

        Raises:
            Exception: If review creation or status update fails
        """
        from backend.data.execution import ExecutionStatus

        # Note: Safe mode checks (human_in_the_loop_safe_mode, sensitive_action_safe_mode)
        # are handled by the caller:
        # - HITL blocks check human_in_the_loop_safe_mode in their run() method
        # - Sensitive action blocks check sensitive_action_safe_mode in is_block_exec_need_review()
        # This function only handles checking for existing approvals.
        # Check if this node has already been approved (normal or auto-approval)
        if approval_result := await HITLReviewHelper.check_approval(
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            node_id=node_id,
            user_id=user_id,
            input_data=input_data,
        ):
            logger.info(
                f"Block {block_name} skipping review for node {node_exec_id} - "
                f"found existing approval"
            )
            # Return a new ReviewResult with the current node_exec_id but approved status
            # For auto-approvals, always use current input_data
            # For normal approvals, use approval_result.data unless it's None
            is_auto_approval = approval_result.node_exec_id != node_exec_id
            approved_data = (
                input_data
                if is_auto_approval
                else (
                    approval_result.data
                    if approval_result.data is not None
                    else input_data
                )
            )
            return ReviewResult(
                data=approved_data,
                status=ReviewStatus.APPROVED,
                message=approval_result.message,
                processed=True,
                node_exec_id=node_exec_id,
            )

        result = await HITLReviewHelper.get_or_create_human_review(
            user_id=user_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            input_data=input_data,
            message=block_name,  # Use block_name directly as the message
            editable=editable,
        )

        if result is None:
            logger.info(
                f"Block {block_name} pausing execution for node {node_exec_id} - awaiting human review"
            )
            await HITLReviewHelper.update_node_execution_status(
                exec_id=node_exec_id,
                status=ExecutionStatus.REVIEW,
            )
            return None  # Signal that execution should pause

        # Mark review as processed if not already done
        if not result.processed:
            await HITLReviewHelper.update_review_processed_status(
                node_exec_id=node_exec_id, processed=True
            )

        return result

    @staticmethod
    async def handle_review_decision(
        input_data: Any,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: str,
        graph_id: str,
        graph_version: int,
        block_name: str = "Block",
        editable: bool = False,
    ) -> Optional[ReviewDecision]:
        """
        Handle a review request and return the decision in a single call.

        Args:
            input_data: The input data to be reviewed
            user_id: ID of the user requesting the review
            node_id: ID of the node in the graph definition
            node_exec_id: ID of the node execution
            graph_exec_id: ID of the graph execution
            graph_id: ID of the graph
            graph_version: Version of the graph
            block_name: Name of the block requesting review
            editable: Whether the reviewer can edit the data

        Returns:
            ReviewDecision if review is complete (approved/rejected),
            None if execution should pause (awaiting review)
        """
        review_result = await HITLReviewHelper._handle_review_request(
            input_data=input_data,
            user_id=user_id,
            node_id=node_id,
            node_exec_id=node_exec_id,
            graph_exec_id=graph_exec_id,
            graph_id=graph_id,
            graph_version=graph_version,
            block_name=block_name,
            editable=editable,
        )

        if review_result is None:
            # Still awaiting review - return None to pause execution
            return None

        # Review is complete, determine outcome
        should_proceed = review_result.status == ReviewStatus.APPROVED
        message = review_result.message or (
            "Execution approved by reviewer"
            if should_proceed
            else "Execution rejected by reviewer"
        )

        return ReviewDecision(
            should_proceed=should_proceed, message=message, review_result=review_result
        )
