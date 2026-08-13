"""FixAgentGraphTool - Auto-fixes common agent JSON issues."""

import difflib
import json
import logging
from typing import Any

from backend.copilot.model import ChatSession

from .agent_generator.validation import AgentFixer, AgentValidator, get_blocks_as_dicts
from .agent_json_input import (
    AGENT_JSON_REF_SCHEMA,
    AGENT_JSON_SCHEMA,
    resolve_agent_json_or_error,
    write_agent_json_to_workspace,
)
from .base import BaseTool
from .helpers import require_guide_read
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
            "Auto-fix common agent JSON issues (invalid UUIDs, brace "
            "escaping, block prerequisites, credentials, model defaults, "
            "type mismatches). Returns fixed JSON + fixes applied. "
            "Requires the building guide first (refuses otherwise)."
        )

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_json": AGENT_JSON_SCHEMA,
                "agent_json_ref": AGENT_JSON_REF_SCHEMA,
                "write_to": {
                    "type": "string",
                    "description": (
                        "Workspace filename (no directories) to write the "
                        "fixed JSON to (pretty-printed, overwrites). The "
                        "response then returns an @@agptfile ref to pass to "
                        "create_agent/edit_agent instead of the full JSON."
                    ),
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_json: dict | str | None = None,
        agent_json_ref: str | None = None,
        write_to: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None

        guide_gate = require_guide_read(session, "fix_agent_graph")
        if guide_gate is not None:
            return guide_gate

        agent_json, resolve_error = await resolve_agent_json_or_error(
            agent_json=agent_json,
            agent_json_ref=agent_json_ref,
            user_id=user_id,
            session=session,
            session_id=session_id,
            missing_message=(
                "Please provide a valid agent JSON object via agent_json (inline "
                'or an "@@agptfile:<path>" string), or agent_json_ref pointing '
                "at the workspace agent file."
            ),
            missing_error="Missing or invalid agent_json parameter",
            invalid_error="Missing or invalid agent_json parameter",
        )
        if resolve_error is not None:
            return resolve_error
        assert agent_json is not None  # narrowed: resolve_error covers the None case

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
            fixed_agent = fixer.apply_all_fixes(agent_json, blocks)
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
            message = f"Applied {len(fixes_applied)} fix(es). Agent graph is now valid!"
        else:
            message = (
                f"Applied {len(fixes_applied)} fix(es), but "
                f"{len(remaining_errors)} issue(s) remain. "
                "Review the remaining errors and fix manually."
            )

        fixed_ref: str | None = None
        fix_diff: str | None = None
        if write_to := write_to.strip():
            fixed_ref, write_note = await write_agent_json_to_workspace(
                fixed_agent,
                write_to,
                user_id,
                session_id,
                label="Fixed JSON",
                pass_to="create_agent/edit_agent",
                fallback_note="returning the fixed JSON inline instead.",
            )
            message += write_note
            if fixed_ref and fixes_applied:
                fix_diff = _build_fix_diff(agent_json, fixed_agent, write_to)

        return FixResultResponse(
            message=message,
            fixed_agent_json=None if fixed_ref else fixed_agent,
            fixed_agent_ref=fixed_ref,
            fix_diff=fix_diff,
            fixes_applied=fixes_applied,
            fix_count=len(fixes_applied),
            valid_after_fix=is_valid,
            remaining_errors=remaining_errors if not is_valid else [],
            session_id=session_id,
        )


_MAX_DIFF_CHARS = 4000


def _build_fix_diff(
    original: dict[str, Any], fixed: dict[str, Any], filename: str
) -> str | None:
    """Unified diff of the applied fixes against the pretty-printed JSON.

    Lets the model keep its mental copy of the (just-overwritten) workspace
    file current without re-reading it. Both sides are normalized with
    ``json.dumps(indent=2)`` — the "after" side is byte-identical to what
    ``_write_fixed_agent`` wrote.
    """
    diff_lines = difflib.unified_diff(
        json.dumps(original, indent=2).splitlines(keepends=True),
        json.dumps(fixed, indent=2).splitlines(keepends=True),
        fromfile=f"{filename} (before fixes)",
        tofile=f"{filename} (after fixes)",
        n=2,
    )
    diff = "".join(diff_lines)
    if not diff:
        return None
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + (
            "\n... [diff truncated — re-read the file before making " "targeted edits]"
        )
    return diff
