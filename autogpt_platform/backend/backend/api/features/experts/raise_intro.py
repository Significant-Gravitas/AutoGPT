"""A new expert's opening line and setup questions, from its name and role.

The raise dialog asks this while the user is still typing, so it runs on the
fast model with a short timeout. Any failure answers with a template: the
expert page renders whatever comes back and must never wait on a model.
"""

import logging
import os

from openai import OpenAIError
from pydantic import BaseModel, Field, field_validator

from backend.api.features.onboarding_dump.structured import request_json
from backend.util.clients import get_openai_client

logger = logging.getLogger(__name__)

_MODEL = os.environ.get("RAISE_INTRO_MODEL", "anthropic/claude-haiku-5.5")
_TIMEOUT_SECONDS = 8

MAX_QUESTIONS = 3
MAX_OPTIONS = 3
MAX_INTRO_CHARS = 200
MAX_QUESTION_CHARS = 90
MAX_OPTION_CHARS = 40

_PROMPT = """You are a new AI teammate the user just created. They gave you \
only a name and a role. Work out from those alone what your job is, the way \
a sharp new hire reads their own job title.

Name: {name}
Role: {role}

Return ONLY valid JSON with exactly two keys:
- "intro": one sentence, first person, at most 140 characters, that says \
concretely what you will do for them. Be specific to the name and role \
(a "Deps merger" keeps dependency update pull requests reviewed and merged). \
No greeting, no dashes, no exclamation marks.
- "questions": an array of 2 or 3 objects, each with "question" (at most 70 \
characters) and "options" (exactly 3 short answers, at most 28 characters \
each). Ask only what you need to start: which project or system, how much \
freedom you have, and what matters most. Options must be realistic answers \
for this role, not generic placeholders.

Never use the words "agent", "graph", "AI" or "assistant". Call automations \
"workflows".
"""


class RaiseIntroRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    job_title: str | None = Field(default=None, max_length=100)

    @field_validator("name", "job_title", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RaiseIntroQuestion(BaseModel):
    question: str
    options: list[str]


class RaiseIntro(BaseModel):
    intro: str
    questions: list[RaiseIntroQuestion]
    # False when the template answered, so the client knows the copy is
    # generic rather than read from the name.
    is_generated: bool


async def generate_intro(request: RaiseIntroRequest) -> RaiseIntro:
    client = get_openai_client(prefer_openrouter=True)
    if client is None:
        return fallback_intro(request)
    prompt = _PROMPT.format(name=request.name, role=request.job_title or "not given")
    try:
        reply = await request_json(
            client,
            model=_MODEL,
            prompt=prompt,
            temperature=0.4,
            max_tokens=600,
            timeout=_TIMEOUT_SECONDS,
        )
    except (OpenAIError, TimeoutError) as e:  # the template answers instead
        logger.warning(f"Raise intro generation failed: {e}")
        return fallback_intro(request)
    parsed = parse_intro(reply.data)
    if parsed is None:
        logger.warning(f"Raise intro: unusable output ({reply.describe()})")
        return fallback_intro(request)
    return parsed


def parse_intro(data: object) -> RaiseIntro | None:
    """The model's answer, clipped to what the page can show, or ``None``."""
    if not isinstance(data, dict):
        return None
    intro = data.get("intro")
    if not isinstance(intro, str) or not intro.strip():
        return None
    questions = [
        question
        for question in (
            _parse_question(item) for item in _as_list(data.get("questions"))
        )
        if question is not None
    ][:MAX_QUESTIONS]
    if not questions:
        return None
    return RaiseIntro(
        intro=intro.strip()[:MAX_INTRO_CHARS],
        questions=questions,
        is_generated=True,
    )


def _parse_question(item: object) -> RaiseIntroQuestion | None:
    if not isinstance(item, dict):
        return None
    question = item.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    options = [
        option.strip()[:MAX_OPTION_CHARS]
        for option in _as_list(item.get("options"))
        if isinstance(option, str) and option.strip()
    ][:MAX_OPTIONS]
    if len(options) < 2:
        return None
    return RaiseIntroQuestion(
        question=question.strip()[:MAX_QUESTION_CHARS], options=options
    )


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def fallback_intro(request: RaiseIntroRequest) -> RaiseIntro:
    role = request.job_title
    intro = (
        f"I'll take on the {role} work for you, starting with what matters most."
        if role
        else "I'll take on whatever you point me at, starting with what matters most."
    )
    return RaiseIntro(
        intro=intro,
        questions=[
            RaiseIntroQuestion(
                question="What should I start with?",
                options=["A specific project", "Day-to-day tasks", "Research first"],
            ),
            RaiseIntroQuestion(
                question="How much should I check with you?",
                options=["Ask before acting", "Use your judgment", "Just get it done"],
            ),
            RaiseIntroQuestion(
                question="How should I report back?",
                options=["Short summaries", "Full detail", "Only when needed"],
            ),
        ],
        is_generated=False,
    )
