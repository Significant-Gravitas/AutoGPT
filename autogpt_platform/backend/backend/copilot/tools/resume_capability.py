"""Resume a ``run_capability`` call that paused for human review.

Block reviews replay through the existing ``ContinueRunBlockTool`` path; MCP
reviews (synthetic ``copilot-mcp-`` node ids) replay the stored call.  With
``input_overrides`` the call is re-issued as a fresh run with the merged
input, so every gate (credentials, review, spend) is evaluated again rather
than trusting the stored approval for different data.
"""

import logging
from typing import Any

from prisma.enums import ReviewStatus

from backend.copilot.capabilities.mcp_review import MCPReviewPayload, is_mcp_review_id
from backend.copilot.constants import (
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
    parse_node_id_from_exec_id,
)
from backend.copilot.model import ChatSession
from backend.data.db_accessors import review_db

from .base import BaseTool
from .continue_run_block import ContinueRunBlockTool
from .models import ErrorResponse, ToolResponseBase
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool

logger = logging.getLogger(__name__)


class ResumeCapabilityTool(BaseTool):
    digest_large_output = True

    @property
    def name(self) -> str:
        return "resume_capability"

    @property
    def description(self) -> str:
        return (
            "Resume a run_capability call that returned review_required, after the "
            "user approved it. Pass the review_id. input_overrides re-runs the call "
            "with changed input instead (it may ask for approval again)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "review_id": {
                    "type": "string",
                    "description": "review_id from the review_required response.",
                },
                "input_overrides": {
                    "type": "object",
                    "description": "Fields to change before re-running.",
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
        input_overrides: dict[str, Any] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        review_id = (review_id or "").strip()
        session_id = session.session_id
        if not review_id:
            return ErrorResponse(
                message="Please provide a review_id", session_id=session_id
            )
        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )
        if input_overrides is not None and not isinstance(input_overrides, dict):
            return ErrorResponse(
                message="input_overrides must be an object", session_id=session_id
            )
        if is_mcp_review_id(review_id):
            return await _resume_mcp(review_id, user_id, session, input_overrides)
        if input_overrides:
            return await _rerun_block(review_id, user_id, session, input_overrides)
        return await ContinueRunBlockTool()._execute(
            user_id, session, review_id=review_id
        )


async def _load_approved_review(
    review_id: str, user_id: str, session: ChatSession
) -> Any | ErrorResponse:
    session_id = session.session_id
    reviews = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    review = reviews.get(review_id)
    if not review:
        return ErrorResponse(
            message=(
                f"Review '{review_id}' not found or already executed. It may have "
                "been consumed by an earlier resume_capability call."
            ),
            session_id=session_id,
        )
    if review.graph_exec_id != f"{COPILOT_SESSION_PREFIX}{session_id}":
        return ErrorResponse(
            message="Review does not belong to this session.", session_id=session_id
        )
    if review.status == ReviewStatus.WAITING:
        return ErrorResponse(
            message="Review has not been approved yet. Wait for the user to approve it.",
            session_id=session_id,
        )
    if review.status == ReviewStatus.REJECTED:
        return ErrorResponse(
            message="Review was rejected. The capability will not run.",
            session_id=session_id,
        )
    return review


async def _resume_mcp(
    review_id: str,
    user_id: str,
    session: ChatSession,
    input_overrides: dict[str, Any] | None,
) -> ToolResponseBase:
    review = await _load_approved_review(review_id, user_id, session)
    if isinstance(review, ErrorResponse):
        return review
    stored = review.payload if isinstance(review.payload, dict) else {}
    try:
        payload = MCPReviewPayload.model_validate(stored)
    except ValueError:
        return ErrorResponse(
            message="Stored review payload is not an MCP call.",
            session_id=session.session_id,
        )
    arguments = {**payload.arguments, **(input_overrides or {})}
    result = await RunMCPToolTool()._execute(
        user_id,
        session,
        server_url=payload.server_url,
        tool_name=payload.tool,
        tool_arguments=arguments,
    )
    if result.type != "error":
        await review_db().delete_review_by_node_exec_id(review_id, user_id)
    return result


async def _rerun_block(
    review_id: str,
    user_id: str,
    session: ChatSession,
    input_overrides: dict[str, Any],
) -> ToolResponseBase:
    review = await _load_approved_review(review_id, user_id, session)
    if isinstance(review, ErrorResponse):
        return review
    block_id = parse_node_id_from_exec_id(review_id).removeprefix(COPILOT_NODE_PREFIX)
    stored = review.payload if isinstance(review.payload, dict) else {}
    await review_db().delete_review_by_node_exec_id(review_id, user_id)
    return await RunBlockTool()._execute(
        user_id, session, block_id=block_id, input_data={**stored, **input_overrides}
    )
