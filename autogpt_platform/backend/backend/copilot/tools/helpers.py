"""Shared helpers for chat tools."""

import logging
from collections import defaultdict
from typing import Any

from pydantic_core import PydanticUndefined

from backend.blocks._base import AnyBlockSchema
from backend.copilot.constants import COPILOT_NODE_PREFIX, COPILOT_SESSION_PREFIX
from backend.data.credit import UsageTransactionMetadata
from backend.data.db_accessors import credit_db, workspace_db
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
from backend.executor.simulator import simulate_block
from backend.executor.utils import block_usage_cost
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import BlockError, InsufficientBalanceError
from backend.util.type import coerce_inputs_to_schema

from .models import BlockOutputResponse, ErrorResponse, ToolResponseBase
from .utils import match_credentials_to_requirements

logger = logging.getLogger(__name__)


def get_inputs_from_schema(
    input_schema: dict[str, Any],
    exclude_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract input field info from JSON schema."""
    if not isinstance(input_schema, dict):
        return []

    exclude = exclude_fields or set()
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    return [
        {
            "name": name,
            "title": schema.get("title", name),
            "type": schema.get("type", "string"),
            "description": schema.get("description", ""),
            "required": name in required,
            "default": schema.get("default"),
        }
        for name, schema in properties.items()
        if name not in exclude
    ]


async def execute_block(
    *,
    block: AnyBlockSchema,
    block_id: str,
    input_data: dict[str, Any],
    user_id: str,
    session_id: str,
    node_exec_id: str,
    matched_credentials: dict[str, CredentialsMetaInput],
    sensitive_action_safe_mode: bool = False,
    dry_run: bool = False,
) -> ToolResponseBase:
    """Execute a block with full context setup, credential injection, and error handling.

    This is the shared execution path used by both ``run_block`` (after review
    check) and ``continue_run_block`` (after approval).

    Returns:
        BlockOutputResponse on success, ErrorResponse on failure.
    """
    # Dry-run path: simulate the block with an LLM, no real execution
    if dry_run:
        try:
            outputs: dict[str, list[Any]] = defaultdict(list)
            async for output_name, output_data in simulate_block(block, input_data):
                outputs[output_name].append(output_data)
            return BlockOutputResponse(
                message=(
                    f"[DRY RUN] Block '{block.name}' simulated successfully "
                    "— no real execution occurred."
                ),
                block_id=block_id,
                block_name=block.name,
                outputs=dict(outputs),
                success=True,
                session_id=session_id,
            )
        except Exception as e:
            logger.error("Dry-run simulation failed: %s", e, exc_info=True)
            return ErrorResponse(
                message=f"Dry-run simulation failed: {e}",
                error=str(e),
                session_id=session_id,
            )

    try:
        workspace = await workspace_db().get_or_create_workspace(user_id)

        synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
        synthetic_node_id = f"{COPILOT_NODE_PREFIX}{block_id}"

        execution_context = ExecutionContext(
            user_id=user_id,
            graph_id=synthetic_graph_id,
            graph_exec_id=synthetic_graph_id,
            graph_version=1,
            node_id=synthetic_node_id,
            node_exec_id=node_exec_id,
            workspace_id=workspace.id,
            session_id=session_id,
            sensitive_action_safe_mode=sensitive_action_safe_mode,
        )

        exec_kwargs: dict[str, Any] = {
            "user_id": user_id,
            "execution_context": execution_context,
            "workspace_id": workspace.id,
            "graph_exec_id": synthetic_graph_id,
            "node_exec_id": node_exec_id,
            "node_id": synthetic_node_id,
            "graph_version": 1,
            "graph_id": synthetic_graph_id,
        }

        # Inject credentials
        creds_manager = IntegrationCredentialsManager()
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

        # Coerce non-matching data types to the expected input schema.
        coerce_inputs_to_schema(input_data, block.input_schema)

        # Pre-execution credit check (courtesy; spend_credits is atomic)
        cost, cost_filter = block_usage_cost(block, input_data)
        has_cost = cost > 0
        _credit_db = credit_db()
        if has_cost:
            balance = await _credit_db.get_credits(user_id)
            if balance < cost:
                return ErrorResponse(
                    message=(
                        f"Insufficient credits to run '{block.name}'. "
                        "Please top up your credits to continue."
                    ),
                    session_id=session_id,
                )

        # Execute the block and collect outputs
        outputs: dict[str, list[Any]] = defaultdict(list)
        async for output_name, output_data in block.execute(
            input_data,
            **exec_kwargs,
        ):
            outputs[output_name].append(output_data)

        # Charge credits for block execution
        if has_cost:
            try:
                await _credit_db.spend_credits(
                    user_id=user_id,
                    cost=cost,
                    metadata=UsageTransactionMetadata(
                        graph_exec_id=synthetic_graph_id,
                        graph_id=synthetic_graph_id,
                        node_id=synthetic_node_id,
                        node_exec_id=node_exec_id,
                        block_id=block_id,
                        block=block.name,
                        input=cost_filter,
                        reason="copilot_block_execution",
                    ),
                )
            except Exception as e:
                # Block already executed (with possible side effects). Never
                # return ErrorResponse here — the user received output and
                # deserves it. Log the billing failure for reconciliation.
                leak_type = (
                    "INSUFFICIENT_BALANCE"
                    if isinstance(e, InsufficientBalanceError)
                    else "UNEXPECTED_ERROR"
                )
                logger.error(
                    "BILLING_LEAK[%s]: block executed but credit charge failed — "
                    "user_id=%s, block_id=%s, node_exec_id=%s, cost=%s: %s",
                    leak_type,
                    user_id,
                    block_id,
                    node_exec_id,
                    cost,
                    e,
                    extra={
                        "json_fields": {
                            "billing_leak": True,
                            "leak_type": leak_type,
                            "user_id": user_id,
                            "cost": str(cost),
                        }
                    },
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
        logger.warning("Block execution failed: %s", e)
        return ErrorResponse(
            message=f"Block execution failed: {e}",
            error=str(e),
            session_id=session_id,
        )
    except Exception as e:
        logger.error("Unexpected error executing block: %s", e, exc_info=True)
        return ErrorResponse(
            message=f"Failed to execute block: {str(e)}",
            error=str(e),
            session_id=session_id,
        )


async def resolve_block_credentials(
    user_id: str,
    block: AnyBlockSchema,
    input_data: dict[str, Any] | None = None,
) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
    """Resolve credentials for a block by matching user's available credentials.

    Handles discriminated credentials (e.g. provider selection based on model).

    Returns:
        (matched_credentials, missing_credentials)
    """
    input_data = input_data or {}
    requirements = _resolve_discriminated_credentials(block, input_data)

    if not requirements:
        return {}, []

    return await match_credentials_to_requirements(user_id, requirements)


def _resolve_discriminated_credentials(
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
                field = block.input_schema.model_fields.get(field_info.discriminator)
                if field and field.default is not PydanticUndefined:
                    discriminator_value = field.default

            if (
                discriminator_value
                and discriminator_value in field_info.discriminator_mapping
            ):
                effective_field_info = field_info.discriminate(discriminator_value)
                effective_field_info.discriminator_values.add(discriminator_value)
                logger.debug(
                    f"Discriminated provider for {field_name}: "
                    f"{discriminator_value} -> {effective_field_info.provider}"
                )

        resolved[field_name] = effective_field_info

    return resolved
