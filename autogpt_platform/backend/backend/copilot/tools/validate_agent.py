"""ValidateAgentGraphTool - Validates agent JSON structure."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.validation import AgentValidator, get_blocks_as_dicts
from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, ValidationResultResponse

logger = logging.getLogger(__name__)


class ValidateAgentGraphTool(BaseTool):
    """Tool for validating agent JSON graphs."""

    @property
    def name(self) -> str:
        return "validate_agent_graph"

    @property
    def description(self) -> str:
        return (
            "Validate agent JSON for correctness: block_ids, links, required fields, "
            "type compatibility, nested sink notation, prompt brace escaping, "
            "and AgentExecutorBlock configs. On failure, use fix_agent_graph to auto-fix."
        )

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_json": {
                    "type": "object",
                    "description": "Agent JSON with 'nodes' and 'links' arrays.",
                },
            },
            "required": ["agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_json: dict | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None

        if not agent_json or not isinstance(agent_json, dict):
            return ErrorResponse(
                message="Please provide a valid agent JSON object.",
                error="Missing or invalid agent_json parameter",
                session_id=session_id,
            )

        nodes = agent_json.get("nodes", [])

        if not nodes:
            return ErrorResponse(
                message="The agent JSON has no nodes. An agent needs at least one block.",
                error="empty_agent",
                session_id=session_id,
            )

        try:
            blocks = get_blocks_as_dicts()
            validator = AgentValidator()
            is_valid, error_message = validator.validate(agent_json, blocks)
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Validation encountered an error: {str(e)}",
                error="validation_exception",
                session_id=session_id,
            )

        if is_valid:
            return ValidationResultResponse(
                message="Agent graph is valid! No issues found.",
                valid=True,
                errors=[],
                error_count=0,
                session_id=session_id,
            )

        # Parse individual errors from the validator's error list
        errors = validator.errors if hasattr(validator, "errors") else []
        if not errors and error_message:
            errors = [error_message]

        return ValidationResultResponse(
            message=f"Found {len(errors)} validation error(s). Fix them and re-validate.",
            valid=False,
            errors=errors,
            error_count=len(errors),
            session_id=session_id,
        )
