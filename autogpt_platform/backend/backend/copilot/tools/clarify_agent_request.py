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
    """Confirm the user's intent before starting agent generation.

    Always called as step 1 of the agent generation workflow
    (agent_generation_guide.md) to confirm output format, delivery channel,
    data source, and trigger. Call find_block first to discover real platform
    options, then call this tool with a concrete question listing those options.
    """

    @property
    def name(self) -> str:
        return "clarify_agent_request"

    @property
    def description(self) -> str:
        return (
            "Confirm the user's intent before building an agent. "
            "Call this as the first step of agent generation. "
            "Call find_block first to discover real platform options for the "
            "relevant dimension (output format, delivery channel, data source, "
            "trigger), then call this tool with a concrete question listing "
            "those options."
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
                    "description": (
                        "Real platform options discovered via find_block for the "
                        "user to choose from (e.g. ['Email', 'Slack', 'Google Docs'])."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": "The find_block search term used to discover the options.",
                },
            },
            "required": ["question"],
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
        keyword: str = kwargs.get("keyword", "")
        session_id = session.session_id if session else None

        if not question:
            return ErrorResponse(
                message="clarify_agent_request requires a non-empty question.",
                error="missing_question",
                session_id=session_id,
            )

        example = ", ".join(options) if options else None
        clarifying_question = ClarifyingQuestion(
            question=question,
            keyword=keyword,
            example=example,
        )
        return ClarificationNeededResponse(
            message=question,
            session_id=session_id,
            questions=[clarifying_question],
        )
