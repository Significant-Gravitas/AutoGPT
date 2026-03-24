"""Tool for continuing block execution after human review approval."""

import logging
from typing import Any

from prisma.enums import ReviewStatus

from backend.blocks import get_block
from backend.copilot.constants import (
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
    parse_node_id_from_exec_id,
)
from backend.copilot.model import ChatSession
from backend.data.db_accessors import review_db

from .base import BaseTool
from .helpers import execute_block, resolve_block_credentials
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class ContinueRunBlockTool(BaseTool):
    """Tool for continuing a block execution after human review approval."""

    @property
    def name(self) -> str:
        return "continue_run_block"

    @property
    def description(self) -> str:
        return "Resume block execution after a run_block call returned review_required. Pass the review_id."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "review_id": {
                    "type": "string",
                    "description": "review_id from the review_required response.",
                },
            },
            "required": ["review_id"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        review_id = (
            kwargs.get("review_id", "").strip() if kwargs.get("review_id") else ""
        )
        session_id = session.session_id

        if not review_id:
            return ErrorResponse(
                message="Please provide a review_id", session_id=session_id
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        # Look up and validate the review record via adapter
        reviews = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
        review = reviews.get(review_id)

        if not review:
            return ErrorResponse(
                message=(
                    f"Review '{review_id}' not found or already executed. "
                    "It may have been consumed by a previous continue_run_block call."
                ),
                session_id=session_id,
            )

        # Validate the review belongs to this session
        expected_graph_exec_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
        if review.graph_exec_id != expected_graph_exec_id:
            return ErrorResponse(
                message="Review does not belong to this session.",
                session_id=session_id,
            )

        if review.status == ReviewStatus.WAITING:
            return ErrorResponse(
                message="Review has not been approved yet. "
                "Please wait for the user to approve the review first.",
                session_id=session_id,
            )

        if review.status == ReviewStatus.REJECTED:
            return ErrorResponse(
                message="Review was rejected. The block will not execute.",
                session_id=session_id,
            )

        # Extract block_id from review_id: copilot-node-{block_id}:{random_hex}
        block_id = parse_node_id_from_exec_id(review_id).removeprefix(
            COPILOT_NODE_PREFIX
        )
        block = get_block(block_id)
        if not block:
            return ErrorResponse(
                message=f"Block '{block_id}' not found", session_id=session_id
            )

        input_data: dict[str, Any] = (
            review.payload if isinstance(review.payload, dict) else {}
        )

        logger.info(
            "Continuing block %s (%s) for user %s with review_id=%s",
            block.name,
            block_id,
            user_id,
            review_id,
        )

        matched_creds, missing_creds = await resolve_block_credentials(
            user_id, block, input_data
        )
        if missing_creds:
            return ErrorResponse(
                message=f"Block '{block.name}' requires credentials that are not configured.",
                session_id=session_id,
            )

        # dry_run=False is safe here: run_block's dry-run fast-path (line ~241)
        # skips HITL entirely, so continue_run_block is never called during a
        # dry run — only real executions reach the human review gate.
        result = await execute_block(
            block=block,
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            node_exec_id=review_id,
            matched_credentials=matched_creds,
            dry_run=False,
        )

        # Delete review record after successful execution (one-time use)
        if result.type != "error":
            await review_db().delete_review_by_node_exec_id(review_id, user_id)

        return result
