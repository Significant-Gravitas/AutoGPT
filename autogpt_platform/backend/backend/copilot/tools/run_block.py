"""Tool for executing blocks directly."""

import logging
import uuid
from collections import defaultdict
from typing import Any

from pydantic_core import PydanticUndefined

from backend.blocks import get_block
from backend.blocks._base import AnyBlockSchema
from backend.copilot.model import ChatSession
from backend.data.db_accessors import workspace_db
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import BlockError

from .base import BaseTool
from .find_block import COPILOT_EXCLUDED_BLOCK_IDS, COPILOT_EXCLUDED_BLOCK_TYPES
from .helpers import get_inputs_from_schema
from .models import (
    BlockDetails,
    BlockDetailsResponse,
    BlockOutputResponse,
    ErrorResponse,
    InputValidationErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from .utils import (
    build_missing_credentials_from_field_info,
    match_credentials_to_requirements,
)

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
            "required inputs and outputs. Then call again with proper input_data to execute."
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
            return ErrorResponse(
                message=(
                    f"Block '{block.name}' cannot be run directly in CoPilot. "
                    "This block is designed for use within graphs only."
                ),
                session_id=session_id,
            )

        logger.info(f"Executing block {block.name} ({block_id}) for user {user_id}")

        creds_manager = IntegrationCredentialsManager()
        (
            matched_credentials,
            missing_credentials,
        ) = await self._resolve_block_credentials(user_id, block, input_data)

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

        try:
            # Get or create user's workspace for CoPilot file operations
            workspace = await workspace_db().get_or_create_workspace(user_id)

            # Generate synthetic IDs for CoPilot context
            # Each chat session is treated as its own agent with one continuous run
            # This means:
            # - graph_id (agent) = session (memories scoped to session when limit_to_agent=True)
            # - graph_exec_id (run) = session (memories scoped to session when limit_to_run=True)
            # - node_exec_id = unique per block execution
            synthetic_graph_id = f"copilot-session-{session.session_id}"
            synthetic_graph_exec_id = f"copilot-session-{session.session_id}"
            synthetic_node_id = f"copilot-node-{block_id}"
            synthetic_node_exec_id = (
                f"copilot-{session.session_id}-{uuid.uuid4().hex[:8]}"
            )

            # Create unified execution context with all required fields
            execution_context = ExecutionContext(
                # Execution identity
                user_id=user_id,
                graph_id=synthetic_graph_id,
                graph_exec_id=synthetic_graph_exec_id,
                graph_version=1,  # Versions are 1-indexed
                node_id=synthetic_node_id,
                node_exec_id=synthetic_node_exec_id,
                # Workspace with session scoping
                workspace_id=workspace.id,
                session_id=session.session_id,
            )

            # Prepare kwargs for block execution
            # Keep individual kwargs for backwards compatibility with existing blocks
            exec_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "execution_context": execution_context,
                # Legacy: individual kwargs for blocks not yet using execution_context
                "workspace_id": workspace.id,
                "graph_exec_id": synthetic_graph_exec_id,
                "node_exec_id": synthetic_node_exec_id,
                "node_id": synthetic_node_id,
                "graph_version": 1,  # Versions are 1-indexed
                "graph_id": synthetic_graph_id,
            }

            for field_name, cred_meta in matched_credentials.items():
                # Inject metadata into input_data (for validation)
                if field_name not in input_data:
                    input_data[field_name] = cred_meta.model_dump()

                # Fetch actual credentials and pass as kwargs (for execution)
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

            # Execute the block and collect outputs
            outputs: dict[str, list[Any]] = defaultdict(list)
            async for output_name, output_data in block.execute(
                input_data,
                **exec_kwargs,
            ):
                outputs[output_name].append(output_data)

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

    async def _resolve_block_credentials(
        self,
        user_id: str,
        block: AnyBlockSchema,
        input_data: dict[str, Any] | None = None,
    ) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
        """
        Resolve credentials for a block by matching user's available credentials.

        Args:
            user_id: User ID
            block: Block to resolve credentials for
            input_data: Input data for the block (used to determine provider via discriminator)

        Returns:
            tuple of (matched_credentials, missing_credentials) - matched credentials
            are used for block execution, missing ones indicate setup requirements.
        """
        input_data = input_data or {}
        requirements = self._resolve_discriminated_credentials(block, input_data)

        if not requirements:
            return {}, []

        return await match_credentials_to_requirements(user_id, requirements)

    def _get_inputs_list(self, block: AnyBlockSchema) -> list[dict[str, Any]]:
        """Extract non-credential inputs from block schema."""
        schema = block.input_schema.jsonschema()
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        return get_inputs_from_schema(schema, exclude_fields=credentials_fields)

    def _resolve_discriminated_credentials(
        self,
        block: AnyBlockSchema,
        input_data: dict[str, Any],
    ) -> dict[str, CredentialsFieldInfo]:
        """Resolve credential requirements, applying discriminator logic where needed."""
        credentials_fields_info = block.input_schema.get_credentials_fields_info()
        if not credentials_fields_info:
            return {}

        resolved: dict[str, CredentialsFieldInfo] = {}

        for field_name, field_info in credentials_fields_info.items():
            effective_field_info = field_info

            if field_info.discriminator and field_info.discriminator_mapping:
                discriminator_value = input_data.get(field_info.discriminator)
                if discriminator_value is None:
                    field = block.input_schema.model_fields.get(
                        field_info.discriminator
                    )
                    if field and field.default is not PydanticUndefined:
                        discriminator_value = field.default

                if (
                    discriminator_value
                    and discriminator_value in field_info.discriminator_mapping
                ):
                    effective_field_info = field_info.discriminate(discriminator_value)
                    # For host-scoped credentials, add the discriminator value
                    # (e.g., URL) so _credential_is_for_host can match it
                    effective_field_info.discriminator_values.add(discriminator_value)
                    logger.debug(
                        f"Discriminated provider for {field_name}: "
                        f"{discriminator_value} -> {effective_field_info.provider}"
                    )

            resolved[field_name] = effective_field_info

        return resolved
