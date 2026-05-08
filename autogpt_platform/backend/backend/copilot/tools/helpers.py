"""Shared helpers for chat tools."""

import asyncio
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
    MAX_TOOL_WAIT_SECONDS,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import FileRefExpansionError, expand_file_refs_in_args
from backend.data.credit import UsageTransactionMetadata
from backend.data.db_accessors import credit_db, review_db, workspace_db
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
from backend.executor.auto_credentials import (
    MissingAutoCredentialsError,
    acquire_auto_credentials,
)
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
    input_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract input field info from JSON schema.

    When *input_data* is provided, each field's ``value`` key is populated
    with the value the CoPilot already supplied — so the frontend can
    prefill the form instead of showing empty inputs.  Fields marked
    ``advanced`` in the schema are flagged so the frontend can hide them
    by default (matching the builder behaviour).
    """
    if not isinstance(input_schema, dict):
        return []

    exclude = exclude_fields or set()
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    provided = input_data or {}

    results: list[dict[str, Any]] = []
    for name, schema in properties.items():
        if name in exclude:
            continue
        # Pass the schema through verbatim so the frontend's generic custom
        # field dispatch (keyed on `format` / json_schema_extra) gets every
        # hint it needs — any block-specific whitelist here would silently
        # downgrade new picker/widget formats to plain text inputs.
        entry: dict[str, Any] = {
            **schema,
            "name": name,
            "title": schema.get("title", name),
            "type": schema.get("type", "string"),
            "description": schema.get("description", ""),
            "required": name in required,
            "default": schema.get("default"),
            "advanced": schema.get("advanced", False),
        }
        if name in provided:
            entry["value"] = provided[name]
        results.append(entry)
    return results


async def _charge_block_credits(
    _credit_db: Any,
    *,
    user_id: str,
    block_name: str,
    block_id: str,
    node_exec_id: str,
    cost: int,
    cost_filter: dict[str, Any],
    synthetic_graph_id: str,
    synthetic_node_id: str,
) -> None:
    """Charge credits for a block execution and log any billing leak.

    Centralised so the normal-path charge and the cancellation-recovery charge
    (see ``execute_block``'s finally) use the same metadata and the same
    leak-logging contract.
    """
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
                block=block_name,
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
        # Intentionally swallow. Block already executed with possible side
        # effects; the caller must still return BlockOutputResponse. The
        # BILLING_LEAK log above is the signal for reconciliation.


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
            async for output_name, output_data in simulate_block(
                block, input_data, user_id=user_id
            ):
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

        # Auto-credentials (picker-populated fields like GoogleDriveFileField).
        # If the picker hasn't been filled, surface the existing setup-card so
        # the user can pick inline via FormRenderer's google-drive-picker; the
        # LLM re-invokes this tool once input_data carries `_credentials_id`.
        auto_locks: list[Any] = []
        try:
            auto_extra_kwargs, auto_locks = await acquire_auto_credentials(
                input_model=block.input_schema,
                input_data=input_data,
                creds_manager=creds_manager,
                user_id=user_id,
            )
        except MissingAutoCredentialsError as e:
            input_schema = block.input_schema.jsonschema()
            credentials_fields = set(block.input_schema.get_credentials_fields().keys())
            return SetupRequirementsResponse(
                message=str(e),
                session_id=session_id,
                setup_info=SetupInfo(
                    agent_id=block_id,
                    agent_name=block.name,
                    user_readiness=UserReadiness(
                        has_all_credentials=True,
                        missing_credentials={},
                        ready_to_run=False,
                    ),
                    requirements={
                        "credentials": [],
                        "inputs": get_inputs_from_schema(
                            input_schema,
                            exclude_fields=credentials_fields,
                            input_data=input_data,
                        ),
                        "execution_modes": ["immediate"],
                    },
                ),
                graph_id=None,
                graph_version=None,
            )
        except ValueError as e:
            return ErrorResponse(message=str(e), error=str(e), session_id=session_id)

        # Everything from here owns the auto-cred locks; wrap so any early
        # return / exception (coerce, credit check, execution, etc.) still
        # releases them. Previously a raise between the acquire and the
        # inner wait_for try could strand locks until Redis TTL.
        try:
            exec_kwargs.update(auto_extra_kwargs)

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

            # Execute the block under the shared MCP wait cap. A block is
            # expected to finish in MAX_TOOL_WAIT_SECONDS; if it doesn't, the
            # MCP handler would block the stream close to the idle timeout.
            # wait_for cancels the generator on timeout, but the finally below
            # still settles billing via asyncio.shield — external side effects
            # may already have landed and the user should be charged for them.
            outputs: dict[str, list[Any]] = defaultdict(list)
            charge_handled = False
            try:
                await asyncio.wait_for(
                    _collect_block_outputs(block, input_data, exec_kwargs, outputs),
                    timeout=MAX_TOOL_WAIT_SECONDS,
                )

                # Normal (non-cancelled) path. Mark charge_handled BEFORE the
                # await so an outer cancellation landing mid-charge can't race
                # the finally block into a double-charge. asyncio.shield keeps
                # the spend running to completion even if the outer awaitable
                # is cancelled.
                if has_cost:
                    charge_handled = True
                    await asyncio.shield(
                        _charge_block_credits(
                            _credit_db,
                            user_id=user_id,
                            block_name=block.name,
                            block_id=block_id,
                            node_exec_id=node_exec_id,
                            cost=cost,
                            cost_filter=cost_filter,
                            synthetic_graph_id=synthetic_graph_id,
                            synthetic_node_id=synthetic_node_id,
                        )
                    )

                return BlockOutputResponse(
                    message=f"Block '{block.name}' executed successfully",
                    block_id=block_id,
                    block_name=block.name,
                    outputs=dict(outputs),
                    success=True,
                    session_id=session_id,
                )
            except asyncio.TimeoutError:
                # Structured record of tool-call timeouts (SECRT-2247 part 3).
                # Grep prod logs for `copilot_tool_timeout` to find tools that
                # keep hitting the cap — candidates for prompt tuning or
                # escalation to the async start+poll pattern.
                logger.warning(
                    "copilot_tool_timeout tool=run_block block=%s block_id=%s "
                    "input_keys=%s user=%s session=%s cap_s=%d",
                    block.name,
                    block_id,
                    sorted(input_data.keys()),
                    user_id,
                    session_id,
                    MAX_TOOL_WAIT_SECONDS,
                )
                return ErrorResponse(
                    message=(
                        f"Block '{block.name}' exceeded the "
                        f"{MAX_TOOL_WAIT_SECONDS}s single-tool wait cap and "
                        "was cancelled. Long-running work should go through "
                        "run_agent (graph executions) or run_sub_session "
                        "(sub-AutoPilot tasks) — those use async start+poll "
                        "so nothing blocks the chat stream."
                    ),
                    session_id=session_id,
                )
            finally:
                # Sentry r3105079148: asyncio.wait_for raises CancelledError
                # into the generator. Normal `except Exception` doesn't catch
                # it, so without this finally a cancelled block would skip
                # credit charging entirely while external side effects still
                # landed. Only run when the normal-path charge was NOT
                # reached (the flag is set before the await, so any
                # cancellation during charge still sets it and avoids
                # double-billing — r3105216985).
                if has_cost and outputs and not charge_handled:
                    await asyncio.shield(
                        _charge_block_credits(
                            _credit_db,
                            user_id=user_id,
                            block_name=block.name,
                            block_id=block_id,
                            node_exec_id=node_exec_id,
                            cost=cost,
                            cost_filter=cost_filter,
                            synthetic_graph_id=synthetic_graph_id,
                            synthetic_node_id=synthetic_node_id,
                        )
                    )
        finally:
            # Release auto-cred locks on every exit path so Redis doesn't hold them until TTL.
            for lock in auto_locks:
                try:
                    await lock.release()
                except Exception as release_exc:
                    logger.warning(
                        "Failed to release auto-credential lock: %s",
                        release_exc,
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


async def _collect_block_outputs(
    block: AnyBlockSchema,
    input_data: dict[str, Any],
    exec_kwargs: dict[str, Any],
    outputs: dict[str, list[Any]],
) -> None:
    """Drive ``block.execute`` and append each emitted pair to *outputs*.

    Extracted so ``asyncio.wait_for`` can wrap exactly the generator-
    consumption step; callers read ``outputs`` afterwards (including from
    the cancellation path) to decide whether the block produced enough
    side-effects to warrant billing.
    """
    async for output_name, output_data in block.execute(input_data, **exec_kwargs):
        outputs[output_name].append(output_data)


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
    validate_only: bool = False,
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
    required_keys = set(input_schema.get("required", []))
    required_non_credential_keys = required_keys - credentials_fields
    provided_input_keys = set(input_data.keys()) - credentials_fields

    # Picker-backed required fields that the caller hasn't filled surface the
    # same setup card as missing OAuth credentials — the frontend renders
    # the picker inline via FormRenderer's custom-field dispatch. Detecting
    # it here (instead of only inside execute_block's auto-creds layer)
    # saves a round trip when the caller omitted the field entirely.
    picker_fields_missing = [
        f
        for f in required_non_credential_keys - provided_input_keys
        if isinstance(input_schema.get("properties", {}).get(f), dict)
        and (
            input_schema["properties"][f].get("format") == "google-drive-picker"
            or "auto_credentials" in input_schema["properties"][f]
        )
    ]

    # validate_only suppresses the setup-card early-return — the caller is
    # doing static introspection, rendering a picker would violate the
    # documented no-side-effects contract of that mode.
    if (missing_credentials or picker_fields_missing) and not (
        dry_run or validate_only
    ):
        credentials_fields_info = _resolve_discriminated_credentials(block, input_data)
        missing_creds_dict = build_missing_credentials_from_field_info(
            credentials_fields_info, set(matched_credentials.keys())
        )
        missing_creds_list = list(missing_creds_dict.values())
        if missing_credentials:
            message = (
                f"Block '{block.name}' requires credentials that are not "
                "configured. Please set up the required credentials before "
                "running this block."
            )
        else:
            message = (
                f"Block '{block.name}' needs "
                f"{', '.join(repr(f) for f in picker_fields_missing)} "
                "picked before it can run. Select in the card below; the "
                "tool will re-run automatically."
            )
        return SetupRequirementsResponse(
            message=message,
            session_id=session_id,
            setup_info=SetupInfo(
                agent_id=block_id,
                agent_name=block.name,
                user_readiness=UserReadiness(
                    has_all_credentials=not missing_credentials,
                    missing_credentials=missing_creds_dict,
                    ready_to_run=False,
                ),
                requirements={
                    "credentials": missing_creds_list,
                    "inputs": get_inputs_from_schema(
                        input_schema,
                        exclude_fields=credentials_fields,
                        input_data=input_data,
                    ),
                    "execution_modes": ["immediate"],
                },
            ),
            graph_id=None,
            graph_version=None,
        )

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


# ---------------------------------------------------------------------------
# Agent-generation gate
# ---------------------------------------------------------------------------
#
# Tools that produce or modify agent JSON (create_agent, edit_agent,
# validate_agent_graph, fix_agent_graph) require the parent agent to have
# read the agent-building guide first — otherwise it tends to generate
# JSON that doesn't match the current block schemas, link semantics, or
# AgentExecutorBlock conventions, then waste turns fixing validation
# errors.  ``require_guide_read`` returns an ``ErrorResponse`` the caller
# should short-circuit with, or ``None`` when the guide has been read.


_AGENT_GUIDE_TOOL_NAME = "get_agent_building_guide"


def require_guide_read(session: ChatSession, tool_name: str):
    """Return an ErrorResponse if the guide hasn't been loaded this session.

    Import inline to keep ``helpers.py`` free of tool-response imports.
    Uses :meth:`ChatSession.has_tool_been_called` which checks both the
    persisted ``messages`` list (session-wide) and the in-flight
    announcement buffer — so a guide call dispatched earlier in the
    *current* turn (before ``session.messages`` flushes at turn end) is
    recognised too.  Otherwise a second tool in the same turn would
    re-fire this guard despite the guide having been called — seen on
    Kimi K2.6 in particular because its aggressive tool-call chaining
    exercises this path far more than Sonnet does.
    """
    from .models import ErrorResponse  # noqa: PLC0415 — avoid circular import

    # Builder-bound sessions always receive the guide inline via the
    # per-turn ``<builder_context>`` injection (see
    # ``backend.copilot.builder_context``), so no tool-call gate is needed —
    # requiring one would waste a round-trip every turn.
    if session.metadata.builder_graph_id:
        return None
    if session.has_tool_been_called(_AGENT_GUIDE_TOOL_NAME):
        return None
    return ErrorResponse(
        message=(
            f"Call get_agent_building_guide first, then retry {tool_name}. "
            "The guide documents required block ids, input/output schemas, "
            "link semantics, and AgentExecutorBlock / MCPToolBlock usage — "
            "generating agent JSON without it produces schema mismatches."
        ),
        session_id=session.session_id,
    )
