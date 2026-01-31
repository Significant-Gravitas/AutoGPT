"""Tool for executing blocks directly."""

import logging
from collections import defaultdict
from typing import Any

from pydantic_core import PydanticUndefined

from backend.api.features.chat.model import ChatSession
from backend.data.block import get_block
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import BlockError

from .base import BaseTool
from .models import (
    BlockOutputResponse,
    ErrorResponse,
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
            "Use the 'id' from find_block results and provide input_data "
            "matching the block's required_inputs."
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
                "input_data": {
                    "type": "object",
                    "description": (
                        "Input values for the block. Use the 'required_inputs' field "
                        "from find_block to see what fields are needed."
                    ),
                },
            },
            "required": ["block_id", "input_data"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _check_block_credentials(
        self,
        user_id: str,
        block: Any,
        input_data: dict[str, Any] | None = None,
    ) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
        """
        Check if user has required credentials for a block.

        Args:
            user_id: User ID
            block: Block to check credentials for
            input_data: Input data for the block (used to determine provider via discriminator)

        Returns:
            tuple[matched_credentials, missing_credentials]
        """
        matched_credentials: dict[str, CredentialsMetaInput] = {}
        missing_credentials: list[CredentialsMetaInput] = []
        input_data = input_data or {}

        # Get credential field info from block's input schema
        credentials_fields_info = block.input_schema.get_credentials_fields_info()

        if not credentials_fields_info:
            return matched_credentials, missing_credentials

        # Get user's available credentials
        creds_manager = IntegrationCredentialsManager()
        available_creds = await creds_manager.store.get_all_creds(user_id)

        for field_name, field_info in credentials_fields_info.items():
            effective_field_info = field_info
            if field_info.discriminator and field_info.discriminator_mapping:
                # Get discriminator from input, falling back to schema default
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
                    logger.debug(
                        f"Discriminated provider for {field_name}: "
                        f"{discriminator_value} -> {effective_field_info.provider}"
                    )

            matching_cred = next(
                (
                    cred
                    for cred in available_creds
                    if cred.provider in effective_field_info.provider
                    and cred.type in effective_field_info.supported_types
                ),
                None,
            )

            if matching_cred:
                matched_credentials[field_name] = CredentialsMetaInput(
                    id=matching_cred.id,
                    provider=matching_cred.provider,  # type: ignore
                    type=matching_cred.type,
                    title=matching_cred.title,
                )
            else:
                # Create a placeholder for the missing credential
                provider = next(iter(effective_field_info.provider), "unknown")
                cred_type = next(iter(effective_field_info.supported_types), "api_key")
                missing_credentials.append(
                    CredentialsMetaInput(
                        id=field_name,
                        provider=provider,  # type: ignore
                        type=cred_type,  # type: ignore
                        title=field_name.replace("_", " ").title(),
                    )
                )

        return matched_credentials, missing_credentials

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

        logger.info(f"Executing block {block.name} ({block_id}) for user {user_id}")

        creds_manager = IntegrationCredentialsManager()
        matched_credentials, missing_credentials = await self._check_block_credentials(
            user_id, block, input_data
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

        try:
            # Fetch actual credentials and prepare kwargs for block execution
            # Create execution context with defaults (blocks may require it)
            exec_kwargs: dict[str, Any] = {
                "user_id": user_id,
                "execution_context": ExecutionContext(),
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

    def _get_inputs_list(self, block: Any) -> list[dict[str, Any]]:
        """Extract non-credential inputs from block schema."""
        inputs_list = []
        schema = block.input_schema.jsonschema()
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        # Get credential field names to exclude
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())

        for field_name, field_schema in properties.items():
            # Skip credential fields
            if field_name in credentials_fields:
                continue

            inputs_list.append(
                {
                    "name": field_name,
                    "title": field_schema.get("title", field_name),
                    "type": field_schema.get("type", "string"),
                    "description": field_schema.get("description", ""),
                    "required": field_name in required_fields,
                }
            )

        return inputs_list
