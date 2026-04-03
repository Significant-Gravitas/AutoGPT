"""Shared helpers for chat tools."""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pydantic_core import PydanticUndefined

from backend.blocks import BlockType, get_block
from backend.blocks._base import AnyBlockSchema
from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_NODE_PREFIX,
    COPILOT_SESSION_PREFIX,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import FileRefExpansionError, expand_file_refs_in_args
from backend.data.credit import UsageTransactionMetadata
from backend.data.db_accessors import credit_db, review_db, workspace_db
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
from backend.executor.simulator import simulate_block
from backend.executor.utils import block_usage_cost
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import BlockError, InsufficientBalanceError
from backend.util.type import coerce_inputs_to_schema

from .models import (
    BlockOutputResponse,
    ErrorResponse,
    InputValidationErrorResponse,
    ReviewRequiredResponse,
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
    dry_run: bool,
) -> ToolResponseBase:
    """Execute a block with full context setup, credential injection, and error handling.

    This is the shared execution path used by both ``run_block`` (after review
    check) and ``continue_run_block`` (after approval).

    Returns:
        BlockOutputResponse on success, ErrorResponse on failure.
    """
    # Dry-run path: simulate the block with an LLM, no real execution.
    # HITL review is intentionally skipped — no real execution occurs.
    if dry_run:
        try:
            # Coerce types to match the block's input schema, same as real execution.
            # This ensures the simulated preview is consistent with real execution
            # (e.g., "42" → 42, string booleans → bool, enum defaults applied).
            coerce_inputs_to_schema(input_data, block.input_schema)
            outputs: dict[str, list[Any]] = defaultdict(list)
            async for output_name, output_data in simulate_block(block, input_data):
                outputs[output_name].append(output_data)
            # simulator signals internal failure via ("error", "[SIMULATOR ERROR …]")
            sim_error = outputs.get("error", [])
            if (
                sim_error
                and isinstance(sim_error[0], str)
                and sim_error[0].startswith("[SIMULATOR ERROR")
            ):
                return ErrorResponse(
                    message=sim_error[0],
                    error=sim_error[0],
                    session_id=session_id,
                )

            return BlockOutputResponse(
                message=f"Block '{block.name}' executed successfully",
                block_id=block_id,
                block_name=block.name,
                outputs=dict(outputs),
                success=True,
                is_dry_run=True,
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


@dataclass
class BlockPreparation:
    """Result of successful block validation, ready for execution or task creation.

    Attributes:
        block: The resolved block instance (schema definition + execute method).
        block_id: UUID of the block being prepared.
        input_data: User-supplied input values after file-ref expansion.
        matched_credentials: Credential field name -> resolved credential metadata.
        input_schema: JSON Schema for the block's input, with credential
            discriminators resolved for the user's available providers.
        credentials_fields: Set of field names in the schema that are credential
            inputs (e.g. ``{"credentials", "api_key"}``).
        required_non_credential_keys: Schema-required fields minus credential
            fields — the fields the user must supply directly.
        provided_input_keys: Keys the user actually provided in ``input_data``.
        synthetic_graph_id: Auto-generated graph UUID used for CoPilot
            single-block executions (no real graph exists in the DB).
        synthetic_node_id: Auto-generated node UUID paired with
            ``synthetic_graph_id`` to form the execution context for the block.
    """

    block: AnyBlockSchema
    block_id: str
    input_data: dict[str, Any]
    matched_credentials: dict[str, CredentialsMetaInput]
    input_schema: dict[str, Any]
    credentials_fields: set[str]
    required_non_credential_keys: set[str]
    provided_input_keys: set[str]
    synthetic_graph_id: str
    synthetic_node_id: str


async def prepare_block_for_execution(
    block_id: str,
    input_data: dict[str, Any],
    user_id: str,
    session: ChatSession,
    session_id: str,
    dry_run: bool,
) -> "BlockPreparation | ToolResponseBase":
    """Validate and prepare a block for execution.

    Performs: block lookup, disabled/excluded-type checks, credential resolution,
    input schema generation, file-ref expansion, missing-credentials check, and
    unrecognized-field validation.

    Does NOT check for missing required fields (tools differ: run_block shows a
    schema preview) and does NOT run the HITL review check (use check_hitl_review
    separately).

    Args:
        block_id: Block UUID to prepare.
        input_data: Input values provided by the caller.
        user_id: Authenticated user ID.
        session: Current chat session (needed for file-ref expansion).
        session_id: Chat session ID (used in error responses).

    Returns:
        BlockPreparation on success, or a ToolResponseBase error/setup response.
    """
    # Lazy import: find_block imports from .base and .models (siblings), not
    # from helpers — no actual circular dependency exists today.  Kept lazy as a
    # precaution since find_block is the block-registry module and future changes
    # could introduce a cycle.
    from .find_block import COPILOT_EXCLUDED_BLOCK_IDS, COPILOT_EXCLUDED_BLOCK_TYPES

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

    matched_credentials, missing_credentials = await resolve_block_credentials(
        user_id, block, input_data
    )

    try:
        input_schema: dict[str, Any] = block.input_schema.jsonschema()
    except Exception as e:
        logger.warning("Failed to generate input schema for block %s: %s", block_id, e)
        return ErrorResponse(
            message=f"Block '{block.name}' has an invalid input schema",
            error=str(e),
            session_id=session_id,
        )

    # Expand @@agptfile: refs using the block's input schema so string/list
    # fields get the correct deserialization.
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

    credentials_fields = set(block.input_schema.get_credentials_fields().keys())

    if missing_credentials and not dry_run:
        credentials_fields_info = _resolve_discriminated_credentials(block, input_data)
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
                    "inputs": get_inputs_from_schema(
                        input_schema, exclude_fields=credentials_fields
                    ),
                    "execution_modes": ["immediate"],
                },
            ),
            graph_id=None,
            graph_version=None,
        )
    required_keys = set(input_schema.get("required", []))
    required_non_credential_keys = required_keys - credentials_fields
    provided_input_keys = set(input_data.keys()) - credentials_fields

    valid_fields = set(input_schema.get("properties", {}).keys()) - credentials_fields
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

    synthetic_graph_id = f"{COPILOT_SESSION_PREFIX}{session_id}"
    synthetic_node_id = f"{COPILOT_NODE_PREFIX}{block_id}"

    return BlockPreparation(
        block=block,
        block_id=block_id,
        input_data=input_data,
        matched_credentials=matched_credentials,
        input_schema=input_schema,
        credentials_fields=credentials_fields,
        required_non_credential_keys=required_non_credential_keys,
        provided_input_keys=provided_input_keys,
        synthetic_graph_id=synthetic_graph_id,
        synthetic_node_id=synthetic_node_id,
    )


async def check_hitl_review(
    prep: BlockPreparation,
    user_id: str,
    session_id: str,
) -> "tuple[str, dict[str, Any]] | ToolResponseBase":
    """Check for an existing or new HITL review requirement.

    If a review is needed, stores the review record and returns a
    ReviewRequiredResponse.  Otherwise returns
    ``(synthetic_node_exec_id, input_data)`` ready for execute_block.
    """
    block = prep.block
    block_id = prep.block_id
    synthetic_graph_id = prep.synthetic_graph_id
    synthetic_node_id = prep.synthetic_node_id
    input_data = prep.input_data

    # Reuse an existing WAITING review for identical input (LLM retry guard)
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
        f"{synthetic_node_id}{COPILOT_NODE_EXEC_ID_SEPARATOR}{uuid.uuid4().hex[:8]}"
    )

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

    return synthetic_node_exec_id, input_data


def _resolve_discriminated_credentials(
    block: AnyBlockSchema,
    input_data: dict[str, Any],
) -> dict[str, CredentialsFieldInfo]:
    """Resolve credential requirements, applying discriminator logic where needed.

    Handles two discrimination modes:
    1. **Provider-based** (``discriminator_mapping`` is set): the discriminator
       field value selects the provider (e.g. an AI model name -> provider).
    2. **URL/host-based** (``discriminator`` is set but ``discriminator_mapping``
       is ``None``): the discriminator field value (typically a URL) is added to
       ``discriminator_values`` so that host-scoped credential matching can
       compare the credential's host against the target URL.
    """
    credentials_fields_info = block.input_schema.get_credentials_fields_info()
    if not credentials_fields_info:
        return {}

    resolved: dict[str, CredentialsFieldInfo] = {}

    for field_name, field_info in credentials_fields_info.items():
        effective_field_info = field_info

        if field_info.discriminator:
            discriminator_value = input_data.get(field_info.discriminator)
            if discriminator_value is None:
                field = block.input_schema.model_fields.get(field_info.discriminator)
                if field and field.default is not PydanticUndefined:
                    discriminator_value = field.default

            if discriminator_value is not None:
                if field_info.discriminator_mapping:
                    # Provider-based discrimination (e.g. model -> provider)
                    if discriminator_value in field_info.discriminator_mapping:
                        effective_field_info = field_info.discriminate(
                            discriminator_value
                        )
                        effective_field_info.discriminator_values.add(
                            discriminator_value
                        )
                        # Model names are safe to log (not PII); URLs are
                        # intentionally omitted in the host-based branch below.
                        logger.debug(
                            "Discriminated provider for %s: %s -> %s",
                            field_name,
                            discriminator_value,
                            effective_field_info.provider,
                        )
                else:
                    # URL/host-based discrimination (e.g. url -> host matching).
                    # Deep copy to avoid mutating the cached schema-level
                    # field_info (model_copy() is shallow — the mutable set
                    # would be shared).
                    effective_field_info = field_info.model_copy(deep=True)
                    effective_field_info.discriminator_values.add(discriminator_value)
                    logger.debug(
                        "Added discriminator value for host matching on %s",
                        field_name,
                    )

        resolved[field_name] = effective_field_info

    return resolved
