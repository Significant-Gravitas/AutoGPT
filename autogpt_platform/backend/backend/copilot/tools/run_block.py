"""Tool for executing blocks directly."""

import logging
import uuid
from typing import Any

from backend.blocks import BlockType, get_block
from backend.blocks._base import AnyBlockSchema
from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import FileRefExpansionError, expand_file_refs_in_args
from backend.data.db_accessors import review_db
from backend.data.execution import ExecutionContext

from .base import BaseTool
from .find_block import COPILOT_EXCLUDED_BLOCK_IDS, COPILOT_EXCLUDED_BLOCK_TYPES
from .helpers import execute_block, get_inputs_from_schema, resolve_block_credentials
from .models import (
    BlockDetails,
    BlockDetailsResponse,
    ErrorResponse,
    InputValidationErrorResponse,
    ReviewRequiredResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .utils import build_missing_credentials_from_field_info

logger = logging.getLogger(__name__)


class RunBlockTool(BaseTool):
    """Tool for executing a block and returning its outputs."""

    @property
    def name(self) -> str:
        return "run_block"

    @property
    def description(self) -> str:
        return (
            "Execute a specific block with the provided input data. "
            "IMPORTANT: You MUST call find_block first to get the block's 'id' - "
            "do NOT guess or make up block IDs. "
            "On first attempt (without input_data), returns detailed schema showing "
            "required inputs and outputs. Then call again with proper input_data to execute. "
            "If a block requires human review, use continue_run_block with the "
            "review_id after the user approves."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_id": {
                    "type": "string",
                    "description": (
                        "The block's 'id' field from find_block results. "
                        "NEVER guess this - always get it from find_block first."
                    ),
                },
                "block_name": {
                    "type": "string",
                    "description": (
                        "The block's human-readable name from find_block results. "
                        "Used for display purposes in the UI."
                    ),
                },
                "input_data": {
                    "type": "object",
                    "description": (
                        "Input values for the block. "
                        "First call with empty {} to see the block's schema, "
                        "then call again with proper values to execute."
                    ),
                },
            },
            "required": ["block_id", "block_name", "input_data"],
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

        # Get the block
        block = get_block(block_id)
        if not block:
            return ErrorResponse(
                message=f"Block '{block_id}' not found",
                session_id=session_id,
            )
        if block.disabled:
            return ErrorResponse(
                message=f"Block '{block_id}' is disabled",
                session_id=session_id,
            )

        # Check if block is excluded from CoPilot (graph-only blocks)
        if (
            block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
            or block.id in COPILOT_EXCLUDED_BLOCK_IDS
        ):
            # Provide actionable guidance for blocks with dedicated tools
            if block.block_type == BlockType.MCP_TOOL:
                hint = (
                    " Use the `run_mcp_tool` tool instead — it handles "
                    "MCP server discovery, authentication, and execution."
                )
            elif block.block_type == BlockType.AGENT:
                hint = " Use the `run_agent` tool instead."
            else:
                hint = " This block is designed for use within graphs only."
            return ErrorResponse(
                message=f"Block '{block.name}' cannot be run directly.{hint}",
                session_id=session_id,
            )

        logger.info(f"Executing block {block.name} ({block_id}) for user {user_id}")

        (
            matched_credentials,
            missing_credentials,
        ) = await resolve_block_credentials(user_id, block, input_data)

        # Get block schemas for details/validation
        try:
            input_schema: dict[str, Any] = block.input_schema.jsonschema()
        except Exception as e:
            logger.warning(
                "Failed to generate input schema for block %s: %s",
                block_id,
                e,
            )
            return ErrorResponse(
                message=f"Block '{block.name}' has an invalid input schema",
                error=str(e),
                session_id=session_id,
            )
        try:
            output_schema: dict[str, Any] = block.output_schema.jsonschema()
        except Exception as e:
            logger.warning(
                "Failed to generate output schema for block %s: %s",
                block_id,
                e,
            )
            return ErrorResponse(
                message=f"Block '{block.name}' has an invalid output schema",
                error=str(e),
                session_id=session_id,
            )

        # Expand @@agptfile: refs in input_data with the block's input
        # schema.  The generic _truncating wrapper skips opaque object
        # properties (input_data has no declared inner properties in the
        # tool schema), so file ref tokens are still intact here.
        # Using the block's schema lets us return raw text for string-typed
        # fields and parsed structures for list/dict-typed fields.
        if input_data:
            try:
                input_data = await expand_file_refs_in_args(
                    input_data,
                    user_id,
                    session,
                    input_schema=input_schema,
                )
            except FileRefExpansionError as exc:
                return ErrorResponse(
                    message=(
                        f"Failed to resolve file reference: {exc}. "
                        "Ensure the file exists before referencing it."
                    ),
                    session_id=session_id,
                )

        if missing_credentials:
            # Return setup requirements response with missing credentials
            credentials_fields_info = block.input_schema.get_credentials_fields_info()
            missing_creds_dict = build_missing_credentials_from_field_info(
                credentials_fields_info, set(matched_credentials.keys())
            )
            missing_creds_list = list(missing_creds_dict.values())

            return SetupRequirementsResponse(
                message=(
                    f"Block '{block.name}' requires credentials that are not configured. "
                    "Please set up the required credentials before running this block."
                ),
                session_id=session_id,
                setup_info=SetupInfo(
                    agent_id=block_id,
                    agent_name=block.name,
                    user_readiness=UserReadiness(
                        has_all_credentials=False,
                        missing_credentials=missing_creds_dict,
                        ready_to_run=False,
                    ),
                    requirements={
                        "credentials": missing_creds_list,
                        "inputs": self._get_inputs_list(block),
                        "execution_modes": ["immediate"],
                    },
                ),
                graph_id=None,
                graph_version=None,
            )

        # Check if this is a first attempt (required inputs missing)
        # Return block details so user can see what inputs are needed
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        required_keys = set(input_schema.get("required", []))
        required_non_credential_keys = required_keys - credentials_fields
        provided_input_keys = set(input_data.keys()) - credentials_fields

        # Check for unknown input fields
        valid_fields = (
            set(input_schema.get("properties", {}).keys()) - credentials_fields
        )
        unrecognized_fields = provided_input_keys - valid_fields
        if unrecognized_fields:
            return InputValidationErrorResponse(
                message=(
                    f"Unknown input field(s) provided: {', '.join(sorted(unrecognized_fields))}. "
                    f"Block was not executed. Please use the correct field names from the schema."
                ),
                session_id=session_id,
                unrecognized_fields=sorted(unrecognized_fields),
                inputs=input_schema,
            )

        # Show details when not all required non-credential inputs are provided
        if not (required_non_credential_keys <= provided_input_keys):
            # Get credentials info for the response
            credentials_meta = []
            for field_name, cred_meta in matched_credentials.items():
                credentials_meta.append(cred_meta)

            return BlockDetailsResponse(
                message=(
                    f"Block '{block.name}' details. "
                    "Provide input_data matching the inputs schema to execute the block."
                ),
                session_id=session_id,
                block=BlockDetails(
                    id=block_id,
                    name=block.name,
                    description=block.description or "",
                    inputs=input_schema,
                    outputs=output_schema,
                    credentials=credentials_meta,
                ),
                user_authenticated=True,
            )

        # Generate synthetic IDs for CoPilot context.
        # Encode node_id in node_exec_id so it can be extracted later
        # (e.g. for auto-approve, where we need node_id but have no NodeExecution row).
        synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session.session_id}"
        synthetic_node_id = f"{COPILOT_NODE_PREFIX}{block_id}"

        # Check for an existing WAITING review for this block with the same input.
        # If the LLM retries run_block with identical input, we reuse the existing
        # review instead of creating duplicates. Different inputs = new execution.
        existing_reviews = await review_db().get_pending_reviews_for_execution(
            synthetic_graph_id, user_id
        )
        existing_review = next(
            (
                r
                for r in existing_reviews
                if r.node_id == synthetic_node_id
                and r.status.value == "WAITING"
                and r.payload == input_data
            ),
            None,
        )
        if existing_review:
            return ReviewRequiredResponse(
                message=(
                    f"Block '{block.name}' requires human review. "
                    f"After the user approves, call continue_run_block with "
                    f"review_id='{existing_review.node_exec_id}' to execute."
                ),
                session_id=session_id,
                block_id=block_id,
                block_name=block.name,
                review_id=existing_review.node_exec_id,
                graph_exec_id=synthetic_graph_id,
                input_data=input_data,
            )

        synthetic_node_exec_id = (
            f"{synthetic_node_id}{COPILOT_NODE_EXEC_ID_SEPARATOR}"
            f"{uuid.uuid4().hex[:8]}"
        )

        # Check for HITL review before execution.
        # This creates the review record in the DB for CoPilot flows.
        review_context = ExecutionContext(
            user_id=user_id,
            graph_id=synthetic_graph_id,
            graph_exec_id=synthetic_graph_id,
            graph_version=1,
            node_id=synthetic_node_id,
            node_exec_id=synthetic_node_exec_id,
            sensitive_action_safe_mode=True,
        )
        should_pause, input_data = await block.is_block_exec_need_review(
            input_data,
            user_id=user_id,
            node_id=synthetic_node_id,
            node_exec_id=synthetic_node_exec_id,
            graph_exec_id=synthetic_graph_id,
            graph_id=synthetic_graph_id,
            graph_version=1,
            execution_context=review_context,
            is_graph_execution=False,
        )
        if should_pause:
            return ReviewRequiredResponse(
                message=(
                    f"Block '{block.name}' requires human review. "
                    f"After the user approves, call continue_run_block with "
                    f"review_id='{synthetic_node_exec_id}' to execute."
                ),
                session_id=session_id,
                block_id=block_id,
                block_name=block.name,
                review_id=synthetic_node_exec_id,
                graph_exec_id=synthetic_graph_id,
                input_data=input_data,
            )

        return await execute_block(
            block=block,
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            node_exec_id=synthetic_node_exec_id,
            matched_credentials=matched_credentials,
        )

    def _get_inputs_list(self, block: AnyBlockSchema) -> list[dict[str, Any]]:
        """Extract non-credential inputs from block schema."""
        schema = block.input_schema.jsonschema()
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        return get_inputs_from_schema(schema, exclude_fields=credentials_fields)
