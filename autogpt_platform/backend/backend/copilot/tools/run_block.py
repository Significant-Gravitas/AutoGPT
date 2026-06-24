"""Tool for executing blocks directly."""

import logging
import uuid
from typing import Any

from backend.copilot.constants import COPILOT_NODE_EXEC_ID_SEPARATOR
from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession

from .base import BaseTool
from .helpers import (
    BlockPreparation,
    check_hitl_review,
    execute_block,
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
            "Pass `validate_only: true` to inspect a block without running it "
            "(safe pre-flight — returns schema + detected missing inputs). "
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
                "validate_only": {
                    "type": "boolean",
                    "description": (
                        "If true, describe what the block would do without "
                        "executing it or rendering any picker cards. Use this "
                        "as a safe pre-flight for blocks with no required "
                        "inputs (where empty input_data would otherwise "
                        "execute immediately) or to check what a call "
                        "_would_ need before committing."
                    ),
                    "default": False,
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
        *,
        block_id: str = "",
        input_data: dict | None = None,
        validate_only: bool = False,
        **kwargs,  # dry_run is intentionally not accepted; read from session.dry_run
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
        block_id = block_id.strip()
        if input_data is None:
            input_data = {}
        # Session-level flag drives dry-run mode — not exposed to the LLM.
        dry_run = session.dry_run
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

        logger.info("Preparing block %s for user %s", block_id, user_id)

        prep_or_err = await prepare_block_for_execution(
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session=session,
            session_id=session_id,
            dry_run=dry_run,
            validate_only=validate_only,
        )
        if isinstance(prep_or_err, ToolResponseBase):
            return prep_or_err
        prep: BlockPreparation = prep_or_err

        # Check block-level permissions before execution.
        perms = get_current_permissions()
        if perms is not None and not perms.is_block_allowed(block_id, prep.block.name):
            available_hint = (
                f"Allowed identifiers: {perms.blocks!r}. "
                if not perms.blocks_exclude and perms.blocks
                else (
                    f"Blocked identifiers: {perms.blocks!r}. "
                    if perms.blocks_exclude and perms.blocks
                    else ""
                )
            )
            return ErrorResponse(
                message=(
                    f"Block '{prep.block.name}' ({block_id}) is not permitted "
                    f"by the current execution permissions. {available_hint}"
                    "Use find_block to discover blocks that are allowed."
                ),
                session_id=session_id,
            )

        # Dry-run fast-path: skip credential/HITL checks — simulation never calls
        # the real service so credentials and review gates are not needed.
        # Input field validation (unrecognized fields) is already handled by
        # prepare_block_for_execution above.
        if dry_run:
            synthetic_node_exec_id = (
                f"{prep.synthetic_node_id}"
                f"{COPILOT_NODE_EXEC_ID_SEPARATOR}"
                f"{uuid.uuid4().hex[:8]}"
            )
            return await execute_block(
                block=prep.block,
                block_id=block_id,
                input_data=prep.input_data,
                user_id=user_id,
                session_id=session_id,
                node_exec_id=synthetic_node_exec_id,
                matched_credentials=prep.matched_credentials,
                dry_run=True,
            )

        # Show block details when required inputs are not yet provided
        # (two-step UX: first call returns the schema, second call actually
        # executes) or when the caller asked for introspection only via
        # validate_only — in both cases we return BlockDetailsResponse and
        # do not execute.
        if validate_only or not (
            prep.required_non_credential_keys <= prep.provided_input_keys
        ):
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
            missing = sorted(
                prep.required_non_credential_keys - prep.provided_input_keys
            )
            # Hide credential fields from the schema sent to the LLM — the
            # backend resolves them automatically from the user's connected
            # integrations, and exposing the picker shape (id/provider/type)
            # tempts the LLM to fabricate values it cannot construct.
            llm_input_schema = _strip_credentials_from_schema(
                prep.input_schema, prep.credentials_fields
            )
            if validate_only and not missing:
                detail_msg = (
                    f"Block '{prep.block.name}' — all required inputs "
                    f"provided, ready to run."
                )
            elif missing:
                detail_msg = (
                    f"Block '{prep.block.name}' — missing required input(s): "
                    f"{', '.join(repr(m) for m in missing)}."
                )
            else:
                detail_msg = (
                    f"Block '{prep.block.name}' details. Provide input_data "
                    f"matching the inputs schema to execute the block. "
                    f"Credentials are auto-resolved by the backend — do not "
                    f"include them in input_data."
                )
            return BlockDetailsResponse(
                message=detail_msg,
                session_id=session_id,
                block=BlockDetails(
                    id=block_id,
                    name=prep.block.name,
                    description=prep.block.description or "",
                    inputs=llm_input_schema,
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
            dry_run=dry_run,
        )


def _strip_credentials_from_schema(
    input_schema: dict[str, Any], credentials_fields: set[str]
) -> dict[str, Any]:
    """Return a copy of *input_schema* without credential properties.

    Credential fields are auto-resolved from the user's connected integrations;
    leaking the picker shape (``{provider, id, type, title}``) into the
    LLM-facing schema makes the model think it has to construct one and bail
    out when it can't (see the Replicate "Seed Dance 2.0" copilot session).
    """
    if not credentials_fields:
        return input_schema
    cleaned = dict(input_schema)
    properties = dict(cleaned.get("properties", {}))
    for field in credentials_fields:
        properties.pop(field, None)
    cleaned["properties"] = properties
    required = [r for r in cleaned.get("required", []) if r not in credentials_fields]
    if required:
        cleaned["required"] = required
    elif "required" in cleaned:
        del cleaned["required"]
    return cleaned
