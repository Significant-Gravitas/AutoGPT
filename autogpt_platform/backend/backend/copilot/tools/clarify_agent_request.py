"""ClarifyAgentRequestTool - Ask the user one clarifying question before generating an agent."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    ClarificationNeededResponse,
    ClarifyingQuestion,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class ClarifyAgentRequestTool(BaseTool):
    """Present a clarifying question with concrete platform options to the user.

    Called when the user's agent-building goal is ambiguous (missing output
    format, delivery channel, data source, or trigger). The caller must first
    use ``find_block`` to discover real platform options, then pass them here
    so the question is grounded in what the platform actually supports.

    Skipped when the goal already specifies all dimensions.
    """

    @property
    def name(self) -> str:
        return "clarify_agent_request"

    @property
    def description(self) -> str:
        return (
            "Ask the user a clarifying question grounded in real platform "
            "options before building an agent. Call find_block first to "
            "discover available options for the ambiguous dimension (output "
            "format, delivery channel, data source, or trigger), then call "
            "this tool with a concrete question listing those options. "
            "Skip this tool when the user's goal is already specific."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The single concrete question to ask, listing the real "
                        "platform options discovered via find_block."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Real platform options discovered via find_block for the "
                        "user to choose from (e.g. ['Email', 'Slack', 'Google Docs'])."
                    ),
                },
            },
            "required": ["question", "options"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        del user_id  # unused; required by BaseTool contract
        question = str(kwargs.get("question", "")).strip()
        options: list[str] = kwargs.get("options", [])
        session_id = session.session_id if session else None

        if not question:
            return ErrorResponse(
                message="clarify_agent_request requires a non-empty question.",
                error="missing_question",
                session_id=session_id,
            )

        if not options:
            return ErrorResponse(
                message="clarify_agent_request requires at least one option.",
                error="missing_options",
                session_id=session_id,
            )

        clarifying_question = ClarifyingQuestion(
            question=question,
            keyword="",
            example=", ".join(options),
        )
        return ClarificationNeededResponse(
            message=question,
            session_id=session_id,
            questions=[clarifying_question],
        )
