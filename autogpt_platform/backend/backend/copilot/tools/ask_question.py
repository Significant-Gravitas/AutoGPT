"""AskQuestionTool - Ask the user one or more clarifying questions."""

import logging
from datetime import UTC, datetime
from typing import Any

from backend.copilot.model import ChatSession, PendingQuestion
from backend.data.db_accessors import chat_db

from .base import BaseTool
from .models import ClarificationNeededResponse, ClarifyingQuestion, ToolResponseBase

logger = logging.getLogger(__name__)

# The model can be steered by untrusted page content (agent_browser, MCP tools),
# and every option is replayed into context on each subsequent turn.
MAX_QUESTIONS = 10
MAX_OPTIONS = 25
MAX_OPTION_LENGTH = 200
MAX_OPTION_SCAN = 500


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
                                "description": (
                                    "Options for this question, offered to the "
                                    "user as choices; they can always type a "
                                    "custom answer instead. Exactly one option "
                                    "is chosen, so split a question that needs "
                                    "several answers into separate questions."
                                ),
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

        text = "; ".join(q.question for q in questions)
        if session:
            await _mark_pending(session, text)
        return ClarificationNeededResponse(
            message=text,
            session_id=session.session_id if session else None,
            questions=questions,
        )


async def _mark_pending(session: ChatSession, text: str) -> None:
    """Park the question on the session so Home can surface it.

    Best-effort: a failure here costs the user a "Needs You" row, and must
    never cost them the answer they were about to be asked for.
    """
    asked_at = datetime.now(UTC)
    session.metadata.pending_question = PendingQuestion(text=text, asked_at=asked_at)
    try:
        await chat_db().set_session_pending_question(
            session.session_id, session.user_id, text, asked_at
        )
    except Exception as e:
        logger.warning(
            "Could not record pending question for session %s: %s",
            session.session_id,
            e,
        )


def _parse_questions(raw: list[Any]) -> list[ClarifyingQuestion]:
    """Parse and validate raw question dicts into ClarifyingQuestion objects."""
    return [
        q
        for idx, item in enumerate(raw[:MAX_QUESTIONS])
        if (q := _parse_one(item, idx)) is not None
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

    options = _parse_options(item.get("options"))

    return ClarifyingQuestion(
        question=text.strip(),
        keyword=keyword,
        example=", ".join(options) if options else None,
        options=options,
    )


def _parse_options(raw: Any) -> list[str]:
    """Strip the option strings and drop repeats, keeping the model's order.

    Repeats would otherwise reach `example` as "Yes, Yes". Non-strings are
    dropped rather than coerced: the schema declares strings, and `str()` would
    surface a Python repr like "['a']" as a tappable choice. Capped because the
    options are LLM-controlled and get replayed into context every turn.
    """
    if not isinstance(raw, list):
        return []
    unique: dict[str, None] = {}
    # Collect up to the cap rather than normalizing everything and slicing
    # after: a huge array would otherwise cost a strip per entry. Blanks and
    # repeats drop out, so the scan needs its own bound to stay finite.
    for option in raw[:MAX_OPTION_SCAN]:
        if not isinstance(option, str):
            continue
        if trimmed := option.strip()[:MAX_OPTION_LENGTH]:
            unique[trimmed] = None
            if len(unique) == MAX_OPTIONS:
                break
    return list(unique)
