"""Tool for executing blocks directly."""

import logging
from collections import defaultdict
from typing import Any

from backend.data.block import get_block
from backend.data.model import CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.server.v2.chat.model import ChatSession
from backend.server.v2.chat.tools.base import BaseTool
from backend.server.v2.chat.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    SetupInfo,
    SetupRequirementsResponse,
    ToolResponseBase,
    UserReadiness,
)
from backend.util.exceptions import BlockError

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
            "Use find_block to discover available blocks and their input schemas. "
            "The block will run and return its outputs once complete."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_id": {
                    "type": "string",
                    "description": "The UUID of the block to execute",
                },
                "input_data": {
                    "type": "object",
                    "description": (
                        "Input values for the block. Must match the block's input schema. "
                        "Check the block's input_schema from find_block for required fields."
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
    ) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
        """
        Check if user has required credentials for a block.

        Returns:
            tuple[matched_credentials, missing_credentials]
        """
        matched_credentials: dict[str, CredentialsMetaInput] = {}
        missing_credentials: list[CredentialsMetaInput] = []

        # Get credential field info from block's input schema
        credentials_fields_info = block.input_schema.get_credentials_fields_info()

        if not credentials_fields_info:
            return matched_credentials, missing_credentials

        # Get user's available credentials
        creds_manager = IntegrationCredentialsManager()
        available_creds = await creds_manager.store.get_all_creds(user_id)

        for field_name, field_info in credentials_fields_info.items():
            # field_info.provider is a frozenset of acceptable providers
            # field_info.supported_types is a frozenset of acceptable types
            matching_cred = next(
                (
                    cred
                    for cred in available_creds
                    if cred.provider in field_info.provider
                    and cred.type in field_info.supported_types
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
                provider = next(iter(field_info.provider), "unknown")
                cred_type = next(iter(field_info.supported_types), "api_key")
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

        logger.info(f"Executing block {block.name} ({block_id}) for user {user_id}")

        # Check credentials
        matched_credentials, missing_credentials = await self._check_block_credentials(
            user_id, block
        )

        if missing_credentials:
            # Return setup requirements response with missing credentials
            missing_creds_dict = {c.id: c.model_dump() for c in missing_credentials}

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
                        "credentials": [c.model_dump() for c in missing_credentials],
                        "inputs": self._get_inputs_list(block),
                        "execution_modes": ["immediate"],
                    },
                ),
                graph_id=None,
                graph_version=None,
            )

        try:
            # Inject matched credentials into input_data
            for field_name, cred_meta in matched_credentials.items():
                if field_name not in input_data:
                    input_data[field_name] = cred_meta.model_dump()

            # Execute the block and collect outputs
            outputs: dict[str, list[Any]] = defaultdict(list)
            async for output_name, output_data in block.execute(
                input_data,
                user_id=user_id,
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
