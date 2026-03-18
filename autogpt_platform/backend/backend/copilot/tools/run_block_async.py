"""Tool for starting block execution as a background task."""

import asyncio
import logging
import uuid
from typing import Any

from backend.blocks import BlockType, get_block
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
    BlockJobStartedResponse,
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


class RunBlockAsyncTool(BaseTool):
    """Start block execution as a background task and return a job_id immediately.

    This allows the caller to issue multiple run_block_async calls in quick
    succession (e.g. to run several blocks in parallel) and then collect the
    results with get_block_result once they are needed.
    """

    @property
    def name(self) -> str:
        return "run_block_async"

    @property
    def description(self) -> str:
        return (
            "Start a block execution in the background and return a job_id immediately. "
            "Use this instead of run_block when you want to run multiple blocks in parallel. "
            "IMPORTANT: You MUST call find_block first to get the block's 'id'. "
            "Call run_block_async for each block you want to run concurrently, then call "
            "get_block_result with each job_id to retrieve the outputs. "
            "If a block requires human review, use continue_run_block after approval."
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
                    "description": "The block's human-readable name from find_block results.",
                },
                "input_data": {
                    "type": "object",
                    "description": "Input values for the block.",
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
        block_id = kwargs.get("block_id", "").strip()
        input_data = kwargs.get("input_data", {})
        session_id = session.session_id

        if not block_id:
            return ErrorResponse(
                message="Please provide a block_id", session_id=session_id
            )

        if not isinstance(input_data, dict):
            return ErrorResponse(
                message="input_data must be an object", session_id=session_id
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        block = get_block(block_id)
        if not block:
            return ErrorResponse(
                message=f"Block '{block_id}' not found", session_id=session_id
            )
        if block.disabled:
            return ErrorResponse(
                message=f"Block '{block_id}' is disabled", session_id=session_id
            )

        if (
            block.block_type in COPILOT_EXCLUDED_BLOCK_TYPES
            or block.id in COPILOT_EXCLUDED_BLOCK_IDS
        ):
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

        logger.info(
            f"Starting async block execution: {block.name} ({block_id}) for user {user_id}"
        )

        (matched_credentials, missing_credentials) = await resolve_block_credentials(
            user_id, block, input_data
        )

        try:
            input_schema: dict[str, Any] = block.input_schema.jsonschema()
        except Exception as e:
            logger.warning(
                "Failed to generate input schema for block %s: %s", block_id, e
            )
            return ErrorResponse(
                message=f"Block '{block.name}' has an invalid input schema",
                error=str(e),
                session_id=session_id,
            )

        if input_data:
            try:
                input_data = await expand_file_refs_in_args(
                    input_data, user_id, session, input_schema=input_schema
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
                        "execution_modes": ["background"],
                    },
                ),
                graph_id=None,
                graph_version=None,
            )

        # Validate required fields and field names
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        required_keys = set(input_schema.get("required", []))
        required_non_credential_keys = required_keys - credentials_fields
        provided_input_keys = set(input_data.keys()) - credentials_fields

        valid_fields = (
            set(input_schema.get("properties", {}).keys()) - credentials_fields
        )
        unrecognized_fields = provided_input_keys - valid_fields
        if unrecognized_fields:
            return InputValidationErrorResponse(
                message=(
                    f"Unknown input field(s) provided: {', '.join(sorted(unrecognized_fields))}. "
                    "Block was not executed. Please use the correct field names from the schema."
                ),
                session_id=session_id,
                unrecognized_fields=sorted(unrecognized_fields),
                inputs=input_schema,
            )

        if not (required_non_credential_keys <= provided_input_keys):
            return ErrorResponse(
                message=(
                    f"Block '{block.name}' is missing required inputs: "
                    f"{', '.join(sorted(required_non_credential_keys - provided_input_keys))}. "
                    "Please provide all required inputs."
                ),
                session_id=session_id,
            )

        # Generate synthetic IDs for CoPilot context
        synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session.session_id}"
        synthetic_node_id = f"{COPILOT_NODE_PREFIX}{block_id}"

        # Reuse existing WAITING review if LLM retries with identical input
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

        # Check for HITL review
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

        # Start execution as a background task and return the job_id immediately
        job_id = uuid.uuid4().hex

        async def _run() -> Any:
            return await execute_block(
                block=block,
                block_id=block_id,
                input_data=input_data,
                user_id=user_id,
                session_id=session_id,
                node_exec_id=synthetic_node_exec_id,
                matched_credentials=matched_credentials,
            )

        task = asyncio.create_task(_run())
        from backend.copilot.sdk.tool_adapter import (
            get_background_jobs,  # avoid circular import at module level
        )

        jobs = get_background_jobs()
        if jobs is None:
            logger.warning(
                "Background job store not initialised for session %s; "
                "run_block_async results will not be retrievable.",
                session_id,
            )
        else:
            jobs[job_id] = task

        return BlockJobStartedResponse(
            message=(
                f"Block '{block.name}' started in the background. "
                f"Call get_block_result with job_id='{job_id}' to retrieve the output."
            ),
            session_id=session_id,
            job_id=job_id,
            block_id=block_id,
            block_name=block.name,
        )

    def _get_inputs_list(self, block: Any) -> list[dict[str, Any]]:
        schema = block.input_schema.jsonschema()
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        return get_inputs_from_schema(schema, exclude_fields=credentials_fields)
