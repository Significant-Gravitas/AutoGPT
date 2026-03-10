"""Tool for continuing block execution after human review approval."""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.blocks import get_block
from backend.copilot.constants import COPILOT_NODE_EXEC_ID_SEPARATOR
from backend.copilot.model import ChatSession
from backend.data.db_accessors import workspace_db
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import BlockError

from .base import BaseTool
from .models import BlockOutputResponse, ErrorResponse, ToolResponseBase
from .run_block import COPILOT_NODE_PREFIX, COPILOT_SESSION_PREFIX, RunBlockTool

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema

logger = logging.getLogger(__name__)


class ContinueRunBlockTool(BaseTool):
    """Tool for continuing a block execution after human review approval."""

    @property
    def name(self) -> str:
        return "continue_run_block"

    @property
    def description(self) -> str:
        return (
            "Continue executing a block after human review approval. "
            "Use this after a run_block call returned review_required. "
            "Pass the review_id from the review_required response. "
            "The block will execute with the original pre-approved input data."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "review_id": {
                    "type": "string",
                    "description": (
                        "The review_id from a previous review_required response. "
                        "This resumes execution with the pre-approved input data."
                    ),
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
                message="Please provide a review_id",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        # Look up the review record
        review = await PendingHumanReview.prisma().find_first(
            where={
                "nodeExecId": review_id,
                "userId": user_id,
            }
        )

        if not review:
            return ErrorResponse(
                message=(
                    f"Review '{review_id}' not found or already executed. "
                    "It may have been consumed by a previous continue_run_block call."
                ),
                session_id=session_id,
            )

        if review.status == ReviewStatus.WAITING:
            return ErrorResponse(
                message=(
                    "Review has not been approved yet. "
                    "Please wait for the user to approve the review first."
                ),
                session_id=session_id,
            )

        if review.status == ReviewStatus.REJECTED:
            return ErrorResponse(
                message="Review was rejected. The block will not execute.",
                session_id=session_id,
            )

        # Status is APPROVED — extract block_id from review_id format:
        # copilot-node-{block_id}:{random_hex}
        node_id = review_id.rsplit(COPILOT_NODE_EXEC_ID_SEPARATOR, 1)[0]
        block_id = node_id.removeprefix(COPILOT_NODE_PREFIX)

        block = get_block(block_id)
        if not block:
            return ErrorResponse(
                message=f"Block '{block_id}' not found",
                session_id=session_id,
            )

        # Use the original input data from the review record
        payload = review.payload
        input_data: dict[str, Any] = payload if isinstance(payload, dict) else {}

        logger.info(
            f"Continuing block {block.name} ({block_id}) for user {user_id} "
            f"with review_id={review_id}"
        )

        try:
            workspace = await workspace_db().get_or_create_workspace(user_id)

            synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session.session_id}"
            synthetic_graph_exec_id = f"{COPILOT_SESSION_PREFIX}{session.session_id}"
            synthetic_node_id = f"{COPILOT_NODE_PREFIX}{block_id}"

            execution_context = ExecutionContext(
                user_id=user_id,
                graph_id=synthetic_graph_id,
                graph_exec_id=synthetic_graph_exec_id,
                graph_version=1,
                node_id=synthetic_node_id,
                node_exec_id=review_id,
                workspace_id=workspace.id,
                session_id=session.session_id,
                # Disable review — already approved
                sensitive_action_safe_mode=False,
            )

            exec_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "execution_context": execution_context,
                "workspace_id": workspace.id,
                "graph_exec_id": synthetic_graph_exec_id,
                "node_exec_id": review_id,
                "node_id": synthetic_node_id,
                "graph_version": 1,
                "graph_id": synthetic_graph_id,
            }

            # Resolve credentials
            creds_manager = IntegrationCredentialsManager()
            matched_credentials, missing_credentials = (
                await self._resolve_block_credentials(user_id, block, input_data)
            )

            if missing_credentials:
                return ErrorResponse(
                    message=f"Block '{block.name}' requires credentials that are not configured.",
                    session_id=session_id,
                )

            for field_name, cred_meta in matched_credentials.items():
                if field_name not in input_data:
                    input_data[field_name] = cred_meta.model_dump()

                actual_credentials = await creds_manager.get(
                    user_id, cred_meta.id, lock=False
                )
                if actual_credentials:
                    exec_kwargs[field_name] = actual_credentials
                else:
                    return ErrorResponse(
                        message=f"Failed to retrieve credentials for {field_name}",
                        session_id=session_id,
                    )

            # Execute the block
            outputs: dict[str, list[Any]] = defaultdict(list)
            async for output_name, output_data in block.execute(
                input_data,
                **exec_kwargs,
            ):
                outputs[output_name].append(output_data)

            # Delete the review record to enforce one-time use
            await PendingHumanReview.prisma().delete_many(
                where={"nodeExecId": review_id, "userId": user_id}
            )

            return BlockOutputResponse(
                message=f"Block '{block.name}' executed successfully",
                block_id=block_id,
                block_name=block.name,
                outputs=dict(outputs),
                success=True,
                session_id=session_id,
            )

        except BlockError as e:
            logger.warning(f"Block execution failed: {e}")
            return ErrorResponse(
                message=f"Block execution failed: {e}",
                error=str(e),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Unexpected error executing block: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Failed to execute block: {str(e)}",
                error=str(e),
                session_id=session_id,
            )

    # Reuse RunBlockTool's credential resolution (handles discriminated credentials)
    _run_block_tool = RunBlockTool()

    async def _resolve_block_credentials(
        self,
        user_id: str,
        block: "AnyBlockSchema",
        input_data: dict[str, Any],
    ) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
        """Resolve credentials for the block (delegates to RunBlockTool)."""
        return await self._run_block_tool._resolve_block_credentials(
            user_id, block, input_data
        )
