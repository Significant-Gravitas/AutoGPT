"""AskQuestionTool - Ask the user a clarifying question before proceeding."""

from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ClarificationNeededResponse, ClarifyingQuestion, ToolResponseBase


class AskQuestionTool(BaseTool):
    """Ask the user a clarifying question and wait for their answer.

    Use this tool when the user's request is ambiguous and you need more
    information before proceeding. Call find_block or other discovery tools
    first to ground your question in real platform options, then call this
    tool with a concrete question listing those options.
    """

    @property
    def name(self) -> str:
        return "ask_question"

    @property
    def description(self) -> str:
        return (
            "Ask the user a clarifying question. Use when the request is "
            "ambiguous and you need to confirm intent, choose between options, "
            "or gather missing details before proceeding."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The concrete question to ask the user. Should list "
                        "real options when applicable."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Options for the user to choose from "
                        "(e.g. ['Email', 'Slack', 'Google Docs'])."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": "Short label identifying what the question is about.",
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
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id  # unused; required by BaseTool contract
        question_raw = kwargs.get("question")
        if not isinstance(question_raw, str) or not question_raw.strip():
            raise ValueError("ask_question requires a non-empty 'question' string")
        question = question_raw.strip()
        raw_options = kwargs.get("options", [])
        if not isinstance(raw_options, list):
            raw_options = []
        options: list[str] = [str(o) for o in raw_options if o]
        raw_keyword = kwargs.get("keyword", "")
        keyword: str = str(raw_keyword) if raw_keyword else ""
        session_id = session.session_id if session else None

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
