"""FixAgentGraphTool - Auto-fixes common agent JSON issues."""

import difflib
import json
import logging
import os
import re
from typing import Any

from backend.copilot.context import get_workspace_manager
from backend.copilot.model import ChatSession

from .agent_generator.validation import AgentFixer, AgentValidator, get_blocks_as_dicts
from .base import BaseTool
from .helpers import coerce_agent_json, require_guide_read
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
                "agent_json": {
                    "type": ["object", "string"],
                    "description": (
                        "The agent JSON to fix ('nodes' + 'links' arrays), or "
                        'the string "@@agptfile:<path>" to a JSON file '
                        "(preferred for large graphs)."
                    ),
                },
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
            "required": ["agent_json"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        agent_json: dict | str | None = None,
        write_to: str = "",
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None

        guide_gate = require_guide_read(session, "fix_agent_graph")
        if guide_gate is not None:
            return guide_gate

        agent_json = coerce_agent_json(agent_json)
        if not agent_json:
            return ErrorResponse(
                message=(
                    "Please provide a valid agent JSON object, or the string "
                    '"@@agptfile:<path>" referencing a JSON file.'
                ),
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
            fixed_ref, write_note = await _write_fixed_agent(
                fixed_agent, write_to, user_id, session_id
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


async def _write_fixed_agent(
    fixed_agent: dict[str, Any],
    write_to: str,
    user_id: str | None,
    session_id: str | None,
) -> tuple[str | None, str]:
    """Write the fixed agent JSON to a workspace file, pretty-printed.

    Returns (file reference to pass to create_agent/edit_agent, message
    note). On any failure the reference is None and the note explains the
    fallback to inline JSON.
    """
    if not re.fullmatch(r"[\w][\w.-]*", write_to):
        return None, (
            f" NOTE: write_to must be a plain filename (got {write_to!r}); "
            "returning the fixed JSON inline instead."
        )
    if not user_id or not session_id:
        return None, (
            " NOTE: write_to requires an authenticated session; "
            "returning the fixed JSON inline instead."
        )
    try:
        manager = await get_workspace_manager(user_id, session_id)
        rec = await manager.write_file(
            content=json.dumps(fixed_agent, indent=2).encode("utf-8"),
            filename=write_to,
            overwrite=True,
            metadata={"origin": "agent-created"},
        )
    except Exception as e:
        logger.warning(f"fix_agent_graph: failed to write {write_to!r}: {e}")
        return None, (
            f" NOTE: could not write {os.path.basename(write_to)}; "
            "returning the fixed JSON inline instead."
        )
    ref = f"@@agptfile:workspace://{rec.path}"
    return ref, (
        f" Fixed JSON written to workspace file {rec.path} — pass "
        f'agent_json="{ref}" to create_agent/edit_agent (do not re-emit '
        "the JSON)."
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
