"""FixAgentGraphTool - Auto-fixes common agent JSON issues."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.validation import AgentFixer, AgentValidator, get_blocks_as_dicts
from .base import BaseTool
from .models import ErrorResponse, FixResultResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class FixAgentGraphTool(BaseTool):
    """Tool for auto-fixing common issues in agent JSON graphs."""

    @property
    def name(self) -> str:
        return "fix_agent_graph"

    @property
    def description(self) -> str:
        return (
            "Auto-fix common issues in an agent JSON graph. Applies fixes for:\n"
            "- Missing or invalid UUIDs on nodes and links\n"
            "- StoreValueBlock prerequisites for ConditionBlock\n"
            "- Double curly brace escaping in prompt templates\n"
            "- AddToList/AddToDictionary prerequisite blocks\n"
            "- CodeExecutionBlock output field naming\n"
            "- Missing credentials configuration\n"
            "- Node X coordinate spacing (800+ units apart)\n"
            "- AI model default parameters\n"
            "- Link static properties based on input schema\n"
            "- Type mismatches (inserts conversion blocks)\n"
            "- AgentExecutorBlock configuration\n\n"
            "Returns the fixed agent JSON plus a list of fixes applied. "
            "After fixing, the agent is re-validated. If still invalid, "
            "the remaining errors are included in the response."
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
                    "description": (
                        "The agent JSON to fix. Must contain 'nodes' and 'links' arrays."
                    ),
                },
            },
            "required": ["agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        agent_json = kwargs.get("agent_json")
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
            fixer = AgentFixer()
            fixed_agent = await fixer.apply_all_fixes(agent_json, blocks)
            fixes_applied = fixer.get_fixes_applied()
        except Exception as e:
            logger.error(f"Fixer error: {e}", exc_info=True)
            return ErrorResponse(
                message=f"Auto-fix encountered an error: {str(e)}",
                error="fix_exception",
                session_id=session_id,
            )

        # Re-validate after fixing
        try:
            validator = AgentValidator()
            is_valid, _ = validator.validate(fixed_agent, blocks)
            remaining_errors = validator.errors if not is_valid else []
        except Exception as e:
            logger.warning(f"Post-fix validation error: {e}", exc_info=True)
            remaining_errors = [f"Post-fix validation failed: {str(e)}"]
            is_valid = False

        if is_valid:
            return FixResultResponse(
                message=(
                    f"Applied {len(fixes_applied)} fix(es). "
                    "Agent graph is now valid!"
                ),
                fixed_agent_json=fixed_agent,
                fixes_applied=fixes_applied,
                fix_count=len(fixes_applied),
                valid_after_fix=True,
                remaining_errors=[],
                session_id=session_id,
            )

        return FixResultResponse(
            message=(
                f"Applied {len(fixes_applied)} fix(es), but "
                f"{len(remaining_errors)} issue(s) remain. "
                "Review the remaining errors and fix manually."
            ),
            fixed_agent_json=fixed_agent,
            fixes_applied=fixes_applied,
            fix_count=len(fixes_applied),
            valid_after_fix=False,
            remaining_errors=remaining_errors,
            session_id=session_id,
        )
