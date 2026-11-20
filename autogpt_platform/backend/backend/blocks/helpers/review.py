"""
Shared helpers for Human-In-The-Loop (HITL) review functionality.
Used by both the dedicated HumanInTheLoopBlock and blocks that require human review.
"""

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.data.db_accessors import review_db
from backend.data.human_review import ReviewResult
from backend.util.clients import get_database_manager_async_client

if TYPE_CHECKING:
    from backend.blocks._base import Block
    from backend.data.execution import ExecutionContext

logger = logging.getLogger(__name__)

SAFE_MODE_OFF_MESSAGE = "Auto-approved (safe mode disabled)"


class ReviewReadError(RuntimeError):
    """The review saved for a step could not be read, so the step stopped
    before running."""


class ReviewDecision(BaseModel):
    """Result of a review decision."""

    should_proceed: bool
    message: str
    review_result: ReviewResult

    @classmethod
    def from_result(cls, review_result: ReviewResult) -> "ReviewDecision":
        should_proceed = review_result.status == ReviewStatus.APPROVED
        return cls(
            should_proceed=should_proceed,
            message=review_result.message
            or (
                "Execution approved by reviewer"
                if should_proceed
                else "Execution rejected by reviewer"
            ),
            review_result=review_result,
        )


class HITLReviewHelper:
    """Helper class for Human-In-The-Loop review operations."""

    @staticmethod
    async def check_approval(**kwargs) -> Optional[ReviewResult]:
        """Check if there's an existing approval for this node execution."""
        return await get_database_manager_async_client().check_approval(**kwargs)

    @staticmethod
    async def get_stored_review(
        node_exec_id: str, user_id: str
    ) -> Optional[ReviewResult]:
        """The review stored on this step, answered or still waiting, or None
        if it was never asked.

        A read error fails the step: a review that can't be read is never
        taken for none, which would run a step its reviewer may have rejected.
        """
        try:
            reviews = await review_db().get_reviews_by_node_exec_ids(
                [node_exec_id], user_id
            )
        except Exception as e:
            logger.warning(
                f"Could not read the review saved for step {node_exec_id}; "
                f"it does not run: {e!r}"
            )
            raise ReviewReadError(
                f"Could not read the review saved for step {node_exec_id}, "
                "so it did not run"
            ) from e
        review = reviews.get(node_exec_id)
        if review is None:
            return None
        return ReviewResult(
            data=review.payload,
            status=review.status,
            message=review.review_message or "",
            processed=review.processed,
            node_exec_id=node_exec_id,
        )

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
    async def record_gate_approval(**kwargs) -> Optional[ReviewResult]:
        """Record the chat gate's yes on a workflow step."""
        return await get_database_manager_async_client().record_gate_approval(**kwargs)

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
        graph_exec_id: Optional[str],
        graph_id: Optional[str],
        graph_version: Optional[int],
        block_name: str = "Block",
        editable: bool = False,
        is_graph_execution: bool = True,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        block: "Block | None" = None,
        execution_context: "ExecutionContext | None" = None,
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
            chat_session_id: The chat asking, when not a graph execution
            block_name: Name of the block requesting review
            editable: Whether the reviewer can edit the data
            block, execution_context: an irreversible block's own pause, which
                the approval gate of the chat that started the run may answer

        Returns:
            ReviewResult if review is complete, None if waiting for human input

        Raises:
            Exception: If review creation or status update fails
        """
        from backend.data.execution import ExecutionStatus

        # Reached only for a step with no review of its own and its safe mode
        # on (handle_review_decision). An "approve future" stands in for an
        # answer here, so it can never outrank one the step was given.
        if approval_result := await HITLReviewHelper.check_approval(
            node_exec_id=node_exec_id,
            node_id=node_id,
            user_id=user_id,
            input_data=input_data,
            graph_exec_id=graph_exec_id,
            chat_session_id=chat_session_id,
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

        review = {
            "user_id": user_id,
            "node_exec_id": node_exec_id,
            "graph_exec_id": graph_exec_id,
            "graph_id": graph_id,
            "graph_version": graph_version,
            "chat_session_id": chat_session_id,
            "input_data": input_data,
            "message": block_name,  # Use block_name directly as the message
            "editable": editable,
            "organization_id": organization_id,
            "team_id": team_id,
        }
        gated = (
            await HITLReviewHelper._gate_review(block, execution_context, review)
            if block is not None
            and execution_context is not None
            and is_graph_execution
            else None
        )
        if isinstance(gated, ReviewResult):
            result: Optional[ReviewResult] = gated
        elif gated == "parked":
            result = None
        else:
            result = await HITLReviewHelper.get_or_create_human_review(**review)

        if result is None:
            logger.info(
                f"Block {block_name} pausing execution for node {node_exec_id} - awaiting human review"
            )
            if is_graph_execution:
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
    async def _gate_review(
        block: "Block", ctx: "ExecutionContext", review: dict[str, Any]
    ) -> ReviewResult | Literal["parked"] | None:
        """The chat's approval gate answers the pause: a decision to act on,
        ``"parked"`` once its card waits in the chat, or None for a human review.
        """
        # Deferred: the gate imports the block registry, which imports this module.
        from backend.copilot.gate import workflow

        session = await workflow.review_route(ctx, review["user_id"])
        step = (
            await workflow.workflow_step(
                review["graph_exec_id"],
                review["graph_id"],
                review["graph_version"],
                review["node_exec_id"],
            )
            if session is not None
            else None
        )
        if session is None or step is None:
            return None
        try:
            ruling = await workflow.judge_step(
                session, block, review["input_data"], ctx, step
            )
            if ruling.run:
                return (
                    await HITLReviewHelper.record_gate_approval(
                        user_id=review["user_id"],
                        node_exec_id=step.node_exec_id,
                        graph_exec_id=step.graph_exec_id,
                        graph_id=step.graph_id,
                        graph_version=step.graph_version,
                        input_data=review["input_data"],
                        block_name=review["message"],
                        message=ruling.reason,
                        organization_id=review["organization_id"],
                        team_id=review["team_id"],
                    )
                    or "parked"
                )
            waiting = await HITLReviewHelper.get_or_create_human_review(
                **review, alert=False
            )
        except Exception:
            logger.warning(
                f"Gate could not answer step {review['node_exec_id']}; "
                "it waits for a human review",
                exc_info=True,
            )
            return None
        if waiting is not None:
            return waiting
        try:
            opened = await workflow.open_step_card(
                session, block, review["input_data"], step, ruling
            )
        except Exception:
            logger.warning(
                f"Gate could not open the card of step {step.node_exec_id}",
                exc_info=True,
            )
            opened = False
        if not opened:
            # No card, so the row is an ordinary human review and alerts as one.
            try:
                await HITLReviewHelper.get_or_create_human_review(**review)
            except Exception:
                logger.warning(f"Could not alert review {step.node_exec_id}")
        return "parked"

    @staticmethod
    async def handle_review_decision(
        input_data: Any,
        user_id: str,
        node_id: str,
        node_exec_id: str,
        graph_exec_id: Optional[str],
        graph_id: Optional[str],
        graph_version: Optional[int],
        block_name: str = "Block",
        editable: bool = False,
        is_graph_execution: bool = True,
        organization_id: Optional[str] = None,
        team_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        block: "Block | None" = None,
        execution_context: "ExecutionContext | None" = None,
        ask: bool = True,
    ) -> Optional[ReviewDecision]:
        """
        Decide a step that may need a person's review, in this order:

        1. The review already stored on this step. An answer decides it, and
           a review still waiting holds it for that answer, whatever the safe
           mode or the run's queue message says now: a resume, or a late or
           duplicate delivery of the run, may carry another one.
        2. With ``ask`` off (the step's safe mode), an approval of
           ``input_data`` as it is.
        3. An "approve future" for this node, or else a review to wait for.

        Args:
            input_data: The input data to be reviewed
            user_id: ID of the user requesting the review
            node_id: ID of the node in the graph definition
            node_exec_id: ID of the node execution
            graph_exec_id: ID of the graph execution
            graph_id: ID of the graph
            graph_version: Version of the graph
            chat_session_id: The chat asking, when not a graph execution
            block_name: Name of the block requesting review
            editable: Whether the reviewer can edit the data
            ask: Whether a step with no answer stops for review

        Returns:
            ReviewDecision if review is complete (approved/rejected),
            None if execution should pause (awaiting review)
        """
        stored = await HITLReviewHelper.get_stored_review(
            node_exec_id=node_exec_id, user_id=user_id
        )
        if stored is not None and stored.status == ReviewStatus.WAITING:
            logger.info(
                f"Block {block_name} holds node {node_exec_id}: its review is "
                "still waiting for an answer"
            )
            if is_graph_execution:
                from backend.data.execution import ExecutionStatus

                await HITLReviewHelper.update_node_execution_status(
                    exec_id=node_exec_id, status=ExecutionStatus.REVIEW
                )
            return None
        if stored is not None:
            if not stored.processed:
                await HITLReviewHelper.update_review_processed_status(
                    node_exec_id=node_exec_id, processed=True
                )
            if stored.data is None:
                stored = stored.model_copy(update={"data": input_data})
            return ReviewDecision.from_result(stored)

        if not ask:
            logger.debug(
                "Block %s runs node %s unreviewed: safe mode off",
                block_name,
                node_exec_id,
            )
            return ReviewDecision.from_result(
                ReviewResult(
                    data=input_data,
                    status=ReviewStatus.APPROVED,
                    message=SAFE_MODE_OFF_MESSAGE,
                    processed=True,
                    node_exec_id=node_exec_id,
                )
            )

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
            is_graph_execution=is_graph_execution,
            organization_id=organization_id,
            team_id=team_id,
            chat_session_id=chat_session_id,
            block=block,
            execution_context=execution_context,
        )

        if review_result is None:
            # Still awaiting review - return None to pause execution
            return None
        return ReviewDecision.from_result(review_result)
