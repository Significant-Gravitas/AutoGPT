"""AskQuestionTool - Ask the user one or more clarifying questions."""

from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ClarificationNeededResponse, ClarifyingQuestion, ToolResponseBase


class AskQuestionTool(BaseTool):
    """Ask the user one or more clarifying questions and wait for answers.

    Use this tool when the user's request is ambiguous and you need more
    information before proceeding. Call find_block or other discovery tools
    first to ground your questions in real platform options, then call this
    tool with concrete questions listing those options.

    Supports two calling conventions (backward-compatible):
      1. Single question: ``question="Which channel?"``
      2. Multiple questions: ``questions=[{...}, {...}]``

    When *questions* (plural) is provided, the *question*, *options*, and
    *keyword* top-level parameters are ignored.
    """

    @property
    def name(self) -> str:
        return "ask_question"

    @property
    def description(self) -> str:
        return (
            "Ask the user one or more clarifying questions. Use when the "
            "request is ambiguous and you need to confirm intent, choose "
            "between options, or gather missing details before proceeding. "
            "Pass a single question via 'question' or multiple via 'questions'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A single concrete question to ask the user. "
                        "Ignored when 'questions' is provided."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Options for the single question "
                        "(e.g. ['Email', 'Slack', 'Google Docs']). "
                        "Ignored when 'questions' is provided."
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "Short label for the single question. "
                        "Ignored when 'questions' is provided."
                    ),
                },
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
                        "Ask multiple questions at once. Each item has "
                        "'question' (required), 'options', and 'keyword'."
                    ),
                },
            },
            "required": [],
            "anyOf": [
                {"required": ["question"]},
                {"required": ["questions"]},
            ],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_question(
        question: str,
        options: list[str] | None = None,
        keyword: str = "",
    ) -> ClarifyingQuestion:
        """Build a single ``ClarifyingQuestion`` from raw inputs."""
        safe_options = (
            [str(o) for o in options if o] if isinstance(options, list) else []
        )
        return ClarifyingQuestion(
            question=question,
            keyword=keyword,
            example=", ".join(safe_options) if safe_options else None,
        )

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id  # unused; required by BaseTool contract
        session_id = session.session_id if session else None

        raw_questions = kwargs.get("questions")
        if isinstance(raw_questions, list) and raw_questions:
            return self._execute_multi(raw_questions, session_id)

        return self._execute_single(kwargs, session_id)

    def _execute_single(
        self,
        kwargs: dict[str, Any],
        session_id: str | None,
    ) -> ClarificationNeededResponse:
        """Original single-question path (backward-compatible)."""
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

        clarifying_question = self._build_question(question, options, keyword)
        return ClarificationNeededResponse(
            message=question,
            session_id=session_id,
            questions=[clarifying_question],
        )

    def _execute_multi(
        self,
        raw_questions: list[Any],
        session_id: str | None,
    ) -> ClarificationNeededResponse:
        """New multi-question path."""
        clarifying_questions: list[ClarifyingQuestion] = []
        for idx, item in enumerate(raw_questions):
            if not isinstance(item, dict):
                continue
            q_text = item.get("question")
            if not isinstance(q_text, str) or not q_text.strip():
                continue
            keyword = str(item.get("keyword", "")) or f"question-{idx}"
            clarifying_questions.append(
                self._build_question(
                    question=q_text.strip(),
                    options=item.get("options"),
                    keyword=keyword,
                )
            )

        if not clarifying_questions:
            raise ValueError(
                "ask_question requires at least one valid question in 'questions'"
            )

        message = clarifying_questions[0].question
        return ClarificationNeededResponse(
            message=message,
            session_id=session_id,
            questions=clarifying_questions,
        )
