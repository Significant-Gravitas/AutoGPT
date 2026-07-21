"""AskQuestionTool - Ask the user one or more clarifying questions."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ClarificationNeededResponse, ClarifyingQuestion, ToolResponseBase

logger = logging.getLogger(__name__)


class AskQuestionTool(BaseTool):
    """Ask the user one or more clarifying questions and wait for answers.

    Use this tool when the user's request is ambiguous and you need more
    information before proceeding.  Call find_block or other discovery tools
    first to ground your questions in real platform options, then call this
    tool with concrete questions listing those options.
    """

    @property
    def name(self) -> str:
        return "ask_question"

    @property
    def description(self) -> str:
        return (
            "Ask the user one or more clarifying questions. Use when the "
            "request is ambiguous and you need to confirm intent, choose "
            "between options, or gather missing details before proceeding."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question text.",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Options for this question.",
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Short label for this question.",
                            },
                        },
                        "required": ["question"],
                    },
                    "description": (
                        "One or more clarifying questions. Each item has "
                        "'question' (required), 'options', and 'keyword'."
                    ),
                },
            },
            "required": ["questions"],
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
        del user_id
        raw_questions = kwargs.get("questions", [])
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("ask_question requires a non-empty 'questions' array")

        questions = _parse_questions(raw_questions)
        if not questions:
            raise ValueError(
                "ask_question requires at least one valid question in 'questions'"
            )

        return ClarificationNeededResponse(
            message="; ".join(q.question for q in questions),
            session_id=session.session_id if session else None,
            questions=questions,
        )


def _parse_questions(raw: list[Any]) -> list[ClarifyingQuestion]:
    """Parse and validate raw question dicts into ClarifyingQuestion objects."""
    return [
        q for idx, item in enumerate(raw) if (q := _parse_one(item, idx)) is not None
    ]


def _parse_one(item: Any, idx: int) -> ClarifyingQuestion | None:
    """Parse a single question item, returning None for invalid entries."""
    if not isinstance(item, dict):
        logger.warning("ask_question: skipping non-dict item at index %d", idx)
        return None

    text = item.get("question")
    if not isinstance(text, str) or not text.strip():
        logger.warning(
            "ask_question: skipping item at index %d with missing/empty question",
            idx,
        )
        return None

    raw_keyword = item.get("keyword")
    keyword = (
        str(raw_keyword).strip()
        if raw_keyword is not None and str(raw_keyword).strip()
        else f"question-{idx}"
    )

    raw_options = item.get("options")
    options = (
        [str(o) for o in raw_options if o is not None and str(o).strip()]
        if isinstance(raw_options, list)
        else []
    )

    return ClarifyingQuestion(
        question=text.strip(),
        keyword=keyword,
        example=", ".join(options) if options else None,
    )
