"""ExpertOnboardingTool - a freshly hired expert's first-turn intake card."""

import json
import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    ErrorResponse,
    ExpertOnboardingResponse,
    ExpertOnboardingStep,
    ResponseType,
)

logger = logging.getLogger(__name__)

TOOL_NAME = "expert_onboarding"

# The card is a one-question-at-a-time pager, so its length is the user's
# patience, not a context budget: five steps is already a long walk before
# anyone has seen the expert do anything.
MIN_STEPS = 3
MAX_STEPS = 5
MAX_OPTIONS = 6
MAX_QUESTION_LENGTH = 200
MAX_OPTION_LENGTH = 120
MAX_GREETING_LENGTH = 600
# Both arrays are model-authored and the model can be steered by an expert's
# own user-supplied identity text, so every scan needs its own bound.
MAX_SCAN = 200


class ExpertOnboardingTool(BaseTool):
    """Open the onboarding card a newly hired expert greets the user with.

    Called once, on the expert's very first turn, instead of guessing what
    the user wants or starting work unasked. The card asks a handful of
    multiple-choice questions and returns the answers as a normal user
    message, so everything after it is an ordinary conversation.
    """

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Open your onboarding card on your first turn after being hired: "
            "a short greeting plus 3-5 multiple-choice questions that settle "
            "what the user wants from you and which services you need. Ask "
            "for nothing else on that turn, and do not start work until the "
            "answers come back."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "greeting": {
                    "type": "string",
                    "description": (
                        "One or two sentences introducing yourself, in your "
                        "own voice. Shown above the questions."
                    ),
                },
                "steps": {
                    "type": "array",
                    "minItems": MIN_STEPS,
                    "maxItems": MAX_STEPS,
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
                                    "Up to 6 answers the user can tap. Make "
                                    "them concrete and specific to your role "
                                    "— the point is that answering costs no "
                                    "typing. The user can always write their "
                                    "own answer instead. Exactly one is "
                                    "chosen, so split a question that needs "
                                    "several answers into separate steps."
                                ),
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Short label for this step.",
                            },
                        },
                        "required": ["question", "options"],
                    },
                    "description": (
                        "3-5 questions, shown one per step. Order them from "
                        "what you most need to know to what is merely nice "
                        "to know."
                    ),
                },
            },
            "required": ["greeting", "steps"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ExpertOnboardingResponse | ErrorResponse:
        del user_id
        expert_id = session.expert_id if session else None
        if not expert_id:
            return ErrorResponse(
                message=(
                    "expert_onboarding only works in an expert's own chat. "
                    "Use ask_question instead."
                ),
                session_id=session.session_id if session else None,
            )

        if _has_onboarded(session):
            return ErrorResponse(
                message=(
                    "This expert has already opened its onboarding card in "
                    "this chat. Use ask_question for anything further."
                ),
                session_id=session.session_id,
            )

        greeting = kwargs.get("greeting")
        if not isinstance(greeting, str) or not greeting.strip():
            raise ValueError("expert_onboarding requires a non-empty 'greeting'")

        raw_steps = kwargs.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("expert_onboarding requires a non-empty 'steps' array")

        steps = _parse_steps(raw_steps)
        if not steps:
            raise ValueError(
                "expert_onboarding requires at least one valid step in 'steps'"
            )

        return ExpertOnboardingResponse(
            message="; ".join(step.question for step in steps),
            session_id=session.session_id,
            expert_id=expert_id,
            greeting=greeting.strip()[:MAX_GREETING_LENGTH],
            steps=steps,
        )


def _has_onboarded(session: ChatSession) -> bool:
    """Whether this chat already opened an onboarding card successfully.

    The card belongs to the hire, not to the conversation: without this a
    model that re-read its kickoff instruction — or was talked into it — would
    interrupt an ongoing chat with a fresh intake form.

    Keyed off the persisted tool *result* rather than the assistant's tool
    call. The call row is appended before the tool runs (see
    ``BaselineToolPersistence.begin``), so matching on it would let one
    malformed first attempt — bad arguments, a refusal — permanently convince
    this guard the hire was already onboarded, with no way to recover.
    """
    for message in session.messages:
        if message.role != "tool" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except ValueError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("type") == ResponseType.EXPERT_ONBOARDING.value
        ):
            return True
    return False


def _parse_steps(raw: list[Any]) -> list[ExpertOnboardingStep]:
    """Parse raw step dicts, dropping invalid entries and duplicate keywords."""
    steps: list[ExpertOnboardingStep] = []
    seen: set[str] = set()
    for index, item in enumerate(raw[:MAX_SCAN]):
        step = _parse_one(item, index)
        if step is None:
            continue
        if step.keyword in seen:
            continue
        seen.add(step.keyword)
        steps.append(step)
        if len(steps) == MAX_STEPS:
            break
    return steps


def _parse_one(item: Any, index: int) -> ExpertOnboardingStep | None:
    if not isinstance(item, dict):
        logger.warning("expert_onboarding: skipping non-dict step at index %d", index)
        return None

    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        logger.warning(
            "expert_onboarding: skipping step at index %d with no question", index
        )
        return None

    raw_keyword = item.get("keyword")
    keyword = (
        str(raw_keyword).strip()
        if raw_keyword is not None and str(raw_keyword).strip()
        else f"step-{index}"
    )

    return ExpertOnboardingStep(
        question=question.strip()[:MAX_QUESTION_LENGTH],
        keyword=keyword,
        options=_parse_options(item.get("options")),
    )


def _parse_options(raw: Any) -> list[str]:
    """Strip the option strings and drop repeats, keeping the model's order."""
    if not isinstance(raw, list):
        return []
    unique: dict[str, None] = {}
    for option in raw[:MAX_SCAN]:
        if not isinstance(option, str):
            continue
        if trimmed := option.strip()[:MAX_OPTION_LENGTH]:
            unique[trimmed] = None
            if len(unique) == MAX_OPTIONS:
                break
    return list(unique)
