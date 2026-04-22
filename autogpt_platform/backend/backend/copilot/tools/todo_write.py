"""Task-list tool for baseline copilot mode.

Mirrors the schema and UX of Claude Code's built-in ``TodoWrite`` tool so
the frontend's generic tool renderer draws baseline-emitted checklists the
same way it draws SDK-emitted ones. The tool is stateless: the model's
latest ``todos`` argument IS the canonical list, replayed from transcript
on subsequent turns.

Baseline needs this as a platform tool because OpenAI-compatible providers
(Kimi, GPT, Grok, Gemini) do not ship a built-in equivalent. The SDK path
continues to use the CLI's native ``TodoWrite`` — the MCP-wrapped version
of this tool is filtered out of SDK's allowed_tools list (see
``sdk/tool_adapter.py``) to avoid name shadowing.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, TodoItem, TodoWriteResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class TodoWriteTool(BaseTool):
    """Maintain a step-by-step task checklist visible to the user."""

    @property
    def name(self) -> str:
        # Capitalised to match the frontend's switch on ``"TodoWrite"``
        # (see ``copilot/tools/GenericTool/helpers.ts``).
        return "TodoWrite"

    @property
    def description(self) -> str:
        return (
            "Plan and track multi-step work as a visible checklist. Send "
            "the full list every call; exactly one item in_progress at a time."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "Full updated task list (not a delta).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Imperative (e.g. 'Run tests').",
                            },
                            "activeForm": {
                                "type": "string",
                                "description": (
                                    "Present-continuous (e.g. 'Running tests')."
                                ),
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "default": "pending",
                            },
                        },
                        "required": ["content", "activeForm"],
                    },
                },
            },
            "required": ["todos"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id
        raw_todos = kwargs.get("todos")
        if raw_todos is None:
            return ErrorResponse(
                message="`todos` is required.",
                session_id=session.session_id,
            )
        if not isinstance(raw_todos, list):
            return ErrorResponse(
                message="`todos` must be an array.",
                session_id=session.session_id,
            )

        try:
            parsed = [TodoItem.model_validate(item) for item in raw_todos]
        except Exception as exc:
            return ErrorResponse(
                message=f"Invalid todo entry: {exc}",
                session_id=session.session_id,
            )

        in_progress = sum(1 for t in parsed if t.status == "in_progress")
        if in_progress > 1:
            return ErrorResponse(
                message=(
                    "Only one todo may be 'in_progress' at a time "
                    f"(found {in_progress})."
                ),
                session_id=session.session_id,
            )

        return TodoWriteResponse(
            message="Task list updated.",
            session_id=session.session_id,
            todos=parsed,
        )
