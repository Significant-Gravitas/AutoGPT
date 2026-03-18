"""Tool for executing blocks directly."""

import logging
from typing import Any

from backend.blocks._base import AnyBlockSchema
from backend.copilot.model import ChatSession

from .base import BaseTool
from .helpers import (
    BlockPreparation,
    check_hitl_review,
    execute_block,
    get_inputs_from_schema,
    prepare_block_for_execution,
)
from .models import BlockDetails, BlockDetailsResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class RunBlockTool(BaseTool):
    """Tool for executing a block and returning its outputs."""

    @property
    def name(self) -> str:
        return "run_block"

    @property
    def description(self) -> str:
        return (
            "Execute a block. IMPORTANT: Always get block_id from find_block first "
            "— do NOT guess or fabricate IDs. "
            "Call with empty input_data to see schema, then with data to execute. "
            "If review_required, use continue_run_block."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_id": {
                    "type": "string",
                    "description": "Block ID from find_block results.",
                },
                "input_data": {
                    "type": "object",
                    "description": "Input values. Use {} first to see schema.",
                },
            },
            "required": ["block_id", "input_data"],
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
        """Execute a block with the given input data.

        Args:
            user_id: User ID (required)
            session: Chat session
            block_id: Block UUID to execute
            input_data: Input values for the block

        Returns:
            BlockOutputResponse: Block execution outputs
            SetupRequirementsResponse: Missing credentials
            ErrorResponse: Error message
        """
        block_id = kwargs.get("block_id", "").strip()
        input_data = kwargs.get("input_data", {})
        session_id = session.session_id

        if not block_id:
            return ErrorResponse(
                message="Please provide a block_id",
                session_id=session_id,
            )

        if not isinstance(input_data, dict):
            return ErrorResponse(
                message="input_data must be an object",
                session_id=session_id,
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required",
                session_id=session_id,
            )

        logger.info(f"Executing block {block_id} for user {user_id}")

        prep_or_err = await prepare_block_for_execution(
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session=session,
            session_id=session_id,
        )
        if isinstance(prep_or_err, ToolResponseBase):
            return prep_or_err
        prep: BlockPreparation = prep_or_err

        # Show block details when required inputs are not yet provided.
        # This is run_block's two-step UX: first call returns the schema,
        # second call (with inputs) actually executes.
        if not (prep.required_non_credential_keys <= prep.provided_input_keys):
            try:
                output_schema: dict[str, Any] = prep.block.output_schema.jsonschema()
            except Exception as e:
                logger.warning(
                    "Failed to generate output schema for block %s: %s", block_id, e
                )
                return ErrorResponse(
                    message=f"Block '{prep.block.name}' has an invalid output schema",
                    error=str(e),
                    session_id=session_id,
                )

            credentials_meta = list(prep.matched_credentials.values())
            return BlockDetailsResponse(
                message=(
                    f"Block '{prep.block.name}' details. "
                    "Provide input_data matching the inputs schema to execute the block."
                ),
                session_id=session_id,
                block=BlockDetails(
                    id=block_id,
                    name=prep.block.name,
                    description=prep.block.description or "",
                    inputs=prep.input_schema,
                    outputs=output_schema,
                    credentials=credentials_meta,
                ),
                user_authenticated=True,
            )

        hitl_or_err = await check_hitl_review(prep, user_id, session_id)
        if isinstance(hitl_or_err, ToolResponseBase):
            return hitl_or_err
        synthetic_node_exec_id, input_data = hitl_or_err

        return await execute_block(
            block=prep.block,
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            node_exec_id=synthetic_node_exec_id,
            matched_credentials=prep.matched_credentials,
        )

    def _get_inputs_list(self, block: AnyBlockSchema) -> list[dict[str, Any]]:
        """Extract non-credential inputs from block schema."""
        schema = block.input_schema.jsonschema()
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        return get_inputs_from_schema(schema, exclude_fields=credentials_fields)
