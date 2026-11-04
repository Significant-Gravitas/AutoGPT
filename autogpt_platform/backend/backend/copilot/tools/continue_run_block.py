"""Tool for continuing block execution after human review approval."""

import logging
from typing import Any, Awaitable, Callable

from prisma.enums import ReviewStatus

from backend.blocks import get_block
from backend.copilot.constants import COPILOT_NODE_PREFIX, parse_node_id_from_exec_id
from backend.copilot.model import ChatSession
from backend.copilot.tool_display import emit_tool_display_name
from backend.data.db_accessors import review_db

from .base import BaseTool
from .helpers import block_run_refusal, execute_block, resolve_block_credentials
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class ContinueRunBlockTool(BaseTool):
    """Tool for continuing a block execution after human review approval."""

    # Returns execute_block's result, same as run_capability.
    digest_large_output = True

    @property
    def name(self) -> str:
        return "continue_run_block"

    @property
    def description(self) -> str:
        return "Resume block execution after a run_capability call returned review_required. Pass the review_id."

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
        review_id: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        review_id = review_id.strip() if review_id else ""
        session_id = session.session_id

        if not review_id:
            return ErrorResponse(
                message="Please provide a review_id", session_id=session_id
            ).from_outside()

        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            ).from_outside()

        # Look up and validate the review record via adapter
        reviews = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
        review = reviews.get(review_id)

        if not review:
            return ErrorResponse(
                message=(
                    f"Review '{review_id}' not found or already executed. "
                    "It may have been consumed by a previous resume_capability call."
                ),
                session_id=session_id,
            ).from_outside()

        # Validate the review belongs to this session
        if review.session_id != session_id:
            return ErrorResponse(
                message="Review does not belong to this session.",
                session_id=session_id,
            ).from_outside()

        if review.status == ReviewStatus.WAITING:
            return ErrorResponse(
                message="Review has not been approved yet. "
                "Please wait for the user to approve the review first.",
                session_id=session_id,
            ).from_outside()

        if review.status == ReviewStatus.REJECTED:
            return ErrorResponse(
                message="Review was rejected. The block will not execute.",
                session_id=session_id,
            ).from_outside()

        # Extract block_id from review_id: copilot-node-{block_id}:{random_hex}
        block_id = parse_node_id_from_exec_id(review_id).removeprefix(
            COPILOT_NODE_PREFIX
        )
        block = get_block(block_id)
        if not block:
            return ErrorResponse(
                message=f"Block '{block_id}' not found", session_id=session_id
            ).from_outside()
        if block.disabled:
            return await refuse_disabled_block(block_id, review_id, user_id, session_id)
        if refusal := block_run_refusal(block, block_id, session_id):
            return refusal

        emit_tool_display_name(block.name)

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
            user_id, block, input_data, session.expert_id, session_id=session_id
        )
        if missing_creds:
            return ErrorResponse(
                message=f"Block '{block.name}' requires credentials that are not configured.",
                session_id=session_id,
            ).from_outside()

        # dry_run=False is safe here: run_capability's dry-run fast-path skips
        # HITL entirely, so resume_capability is never called during a
        # dry run — only real executions reach the human review gate.
        return await run_claimed(
            review_id,
            user_id,
            session,
            lambda: execute_block(
                block=block,
                block_id=block_id,
                input_data=input_data,
                user_id=user_id,
                session_id=session_id,
                node_exec_id=review_id,
                matched_credentials=matched_creds,
                dry_run=False,
                organization_id=session.organization_id,
                team_id=session.team_id,
                expert_id=session.expert_id,
            ),
        )


async def run_claimed(
    review_id: str,
    user_id: str,
    session: ChatSession,
    run: Callable[[], Awaitable[ToolResponseBase]],
) -> ToolResponseBase:
    """Run an approved call once. Calls resuming the same review can arrive
    together, so the approval is claimed before the call runs; it is spent
    once the call has run, and given back when it did not (an error, a
    sign-in card or a fresh review), for the retry that follows."""
    if not await review_db().claim_approved_review(review_id, user_id):
        return ErrorResponse(
            message=(
                f"Review '{review_id}' not found or already executed. It may have "
                "been consumed by an earlier resume_capability call."
            ),
            session_id=session.session_id,
        ).from_outside()
    try:
        result = await run()
    except Exception:
        await review_db().release_review_claim(review_id, user_id)
        raise
    if result.type in ("error", "setup_requirements", "review_required"):
        await review_db().release_review_claim(review_id, user_id)
    else:
        await review_db().delete_review_by_node_exec_id(review_id, user_id)
    return result


async def refuse_disabled_block(
    block_id: str, review_id: str, user_id: str, session_id: str
) -> ErrorResponse:
    """Refuse an approved review whose block has since been disabled, and use
    the approval up: it outlives the turn that asked for it, and must not run
    the block once the block is enabled again."""
    await review_db().delete_review_by_node_exec_id(review_id, user_id)
    return ErrorResponse(
        message=f"Block '{block_id}' is disabled", session_id=session_id
    ).from_outside()
